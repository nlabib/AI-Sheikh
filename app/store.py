"""SQLite corpus store with an FTS5 index over normalised text.

One row per (ref, edition): Quran 2:255 has an Arabic row, an English row and a
Bangla row, all sharing ref "quran:2:255". Retrieval works on rows and then
groups by ref, so a Bangla question can match the Bangla row and still return
the Arabic text alongside it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.text import BANGLA_TOKENCHARS, normalized_for_index, tokenize

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY,
    ref TEXT NOT NULL,
    collection TEXT NOT NULL,
    lang TEXT NOT NULL,
    edition TEXT NOT NULL,
    text TEXT NOT NULL,
    source_id TEXT NOT NULL,
    allow_training INTEGER NOT NULL DEFAULT 0,
    meta TEXT NOT NULL DEFAULT '{{}}',
    UNIQUE (ref, edition)
);
CREATE INDEX IF NOT EXISTS passages_ref ON passages(ref);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    norm,
    tokenize = "porter unicode61 remove_diacritics 2 tokenchars '{BANGLA_TOKENCHARS}'"
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    info TEXT NOT NULL
);
"""


@dataclass
class Passage:
    ref: str
    collection: str
    lang: str
    edition: str
    text: str
    source_id: str
    allow_training: bool = False
    meta: dict = field(default_factory=dict)
    id: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Passage":
        return cls(
            id=row["id"],
            ref=row["ref"],
            collection=row["collection"],
            lang=row["lang"],
            edition=row["edition"],
            text=row["text"],
            source_id=row["source_id"],
            allow_training=bool(row["allow_training"]),
            meta=json.loads(row["meta"]),
        )


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- writing -----------------------------------------------------------

    def upsert(self, passages: Iterable[Passage]) -> int:
        n = 0
        with self.conn:
            for p in passages:
                old = self.conn.execute(
                    "SELECT id FROM passages WHERE ref=? AND edition=?", (p.ref, p.edition)
                ).fetchone()
                if old:
                    self.conn.execute("DELETE FROM passages_fts WHERE rowid=?", (old["id"],))
                    self.conn.execute("DELETE FROM passages WHERE id=?", (old["id"],))
                cur = self.conn.execute(
                    "INSERT INTO passages(ref, collection, lang, edition, text, source_id, allow_training, meta)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (p.ref, p.collection, p.lang, p.edition, p.text, p.source_id,
                     int(p.allow_training), json.dumps(p.meta, ensure_ascii=False)),
                )
                self.conn.execute(
                    "INSERT INTO passages_fts(rowid, norm) VALUES (?, ?)",
                    (cur.lastrowid, normalized_for_index(p.text)),
                )
                n += 1
        return n

    def record_source(self, source_id: str, info: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sources(source_id, info) VALUES (?, ?)",
                (source_id, json.dumps(info, ensure_ascii=False)),
            )

    # -- reading -----------------------------------------------------------

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def stats(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT source_id, collection, lang, edition, COUNT(*) AS n, MAX(allow_training) AS train"
            " FROM passages GROUP BY source_id ORDER BY collection, lang"
        ).fetchall()
        return [dict(r) for r in rows]

    def sources(self) -> dict[str, dict]:
        return {r["source_id"]: json.loads(r["info"]) for r in self.conn.execute("SELECT * FROM sources")}

    def by_ref(self, ref: str) -> list[Passage]:
        rows = self.conn.execute("SELECT * FROM passages WHERE ref=? ORDER BY lang", (ref,)).fetchall()
        return [Passage.from_row(r) for r in rows]

    def ref_exists(self, ref: str) -> bool:
        return self.conn.execute("SELECT 1 FROM passages WHERE ref=? LIMIT 1", (ref,)).fetchone() is not None

    def refs_by_alt_number(self, collection: str, number: str) -> list[str]:
        """Refs whose alternative number is `number` or `number` plus a letter (Muslim 394 -> 394, 394a...)."""
        rows = self.conn.execute(
            "SELECT DISTINCT ref FROM passages WHERE collection=? AND json_extract(meta, '$.alt_number') "
            "IN (SELECT ? UNION SELECT ? || char(c) FROM (SELECT 97 AS c UNION SELECT 98 UNION SELECT 99 "
            "UNION SELECT 100 UNION SELECT 101 UNION SELECT 102 UNION SELECT 103 UNION SELECT 104)) ORDER BY id",
            (collection, number, number)).fetchall()
        return [r[0] for r in rows]

    def by_ids(self, ids: list[int]) -> dict[int, Passage]:
        out: dict[int, Passage] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = f"SELECT * FROM passages WHERE id IN ({','.join('?' * len(chunk))})"
            for r in self.conn.execute(q, chunk):
                out[r["id"]] = Passage.from_row(r)
        return out

    def iter_passages(self, where: str = "1=1", params: tuple = ()) -> Iterable[Passage]:
        for r in self.conn.execute(f"SELECT * FROM passages WHERE {where} ORDER BY id", params):
            yield Passage.from_row(r)

    def all_ids_and_texts(self) -> list[tuple[int, str]]:
        return [(r[0], r[1]) for r in self.conn.execute("SELECT id, text FROM passages ORDER BY id")]

    def collections_by_id(self) -> dict[int, str]:
        return dict(self.conn.execute("SELECT id, collection FROM passages"))

    def fts_search(self, query: str, limit: int = 50,
                   collections: list[str] | None = None) -> list[tuple[int, float]]:
        """BM25 search; any query term may match. Returns (passage id, score) best first."""
        terms = tokenize(query, drop_stopwords=True) or tokenize(query)
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in dict.fromkeys(terms))
        sql = "SELECT f.rowid, bm25(passages_fts) AS s FROM passages_fts f"
        params: list = []
        if collections:
            sql += f" JOIN passages p ON p.id = f.rowid AND p.collection IN ({','.join('?' * len(collections))})"
            params += collections
        sql += " WHERE passages_fts MATCH ? ORDER BY s LIMIT ?"
        rows = self.conn.execute(sql, [*params, match, limit]).fetchall()
        return [(r[0], -r[1]) for r in rows]
