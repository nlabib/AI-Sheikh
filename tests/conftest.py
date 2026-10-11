from pathlib import Path

import pytest

from app.store import Store
from tools.ingest import jsonl_passages

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample" / "fixture.jsonl"


@pytest.fixture()
def store(tmp_path) -> Store:
    s = Store(tmp_path / "corpus.sqlite")
    s.upsert(jsonl_passages(FIXTURE))
    yield s
    s.close()
