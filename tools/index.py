"""Build the vector index over every passage in the corpus.

    python -m tools.index                                     # multilingual-e5-small (default)
    python -m tools.index --model intfloat/multilingual-e5-base
    python -m tools.index --embedder hashing                  # offline fallback, lexical only

The E5 model downloads from Hugging Face on first use (~470 MB for -small).
Expect roughly 10-20 minutes for the full corpus on a recent NVIDIA GPU or Apple
Silicon, and much longer on CPU. Long hadith are truncated by the model at 512
tokens; the start of a hadith (the matn usually follows the isnad) is kept.
"""

from __future__ import annotations

import argparse
import time

from app.config import get_settings
from app.embed import build_index, make_embedder
from app.store import Store


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", choices=["e5", "hashing"], default=settings.embedder)
    ap.add_argument("--model", default=settings.embed_model)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args(argv)

    store = Store(settings.db_path)
    items = store.all_ids_and_texts()
    if not items:
        raise SystemExit("corpus is empty; run `python -m tools.ingest` first")
    embedder = make_embedder(args.embedder, args.model)
    print(f"embedding {len(items)} passages with {embedder.name}")
    t0 = time.perf_counter()
    index = build_index(embedder, items, batch=args.batch, progress=True)
    index.save(settings.index_dir)
    print(f"saved {len(items)} vectors to {settings.index_dir} in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
