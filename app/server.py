"""FastAPI app.

    uvicorn app.server:app --host 127.0.0.1 --port 8000

Endpoints: POST /ask, GET /search, GET /ref/{ref}, GET /health, GET /sources.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.citations import check_answer
from app.config import get_settings
from app.embed import VectorIndex, make_embedder
from app.llm import ChatClient, LLMError
from app.prompts import NOT_SURE, build_messages
from app.retrieve import Retriever
from app.store import Store
from app.text import detect_lang


def build_retriever(settings) -> tuple[Retriever, str]:
    store = Store(settings.db_path)
    index = VectorIndex.load(settings.index_dir)
    if index is None:
        return Retriever(store), "fts only (no vector index; run `python -m tools.index`)"
    embedder_kind = "hashing" if index.embedder_name.startswith("hashing") else "e5"
    if embedder_kind == "hashing" and settings.embedder != "hashing":
        # The hashing fallback makes hybrid results worse than FTS alone (docs/RESULTS.md).
        return Retriever(store), "fts only (ignoring hashing index; set EMBEDDER=hashing to use it)"
    embedder = make_embedder(embedder_kind, index.embedder_name if embedder_kind == "e5" else "")
    return Retriever(store, embedder, index), f"hybrid (fts + {index.embedder_name})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not hasattr(app.state, "retriever"):
        app.state.retriever, app.state.retrieval_mode = build_retriever(settings)
    if not hasattr(app.state, "llm"):
        app.state.llm = ChatClient(settings)
    app.state.top_k = settings.top_k
    app.state.strict = getattr(app.state, "strict", settings.strict_citations)
    yield


app = FastAPI(title="Islamic Qwen AI", version="0.2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)
    collections: list[str] | None = Field(default=None, description='e.g. ["quran"] or ["bukhari", "muslim"]')
    mode: Literal["hybrid", "fts", "vector"] = "hybrid"


@app.get("/health")
def health(request: Request) -> dict:
    st = request.app.state
    return {"passages": st.retriever.store.count(), "retrieval": st.retrieval_mode, "llm_model": st.llm.model}


@app.get("/sources")
def sources(request: Request) -> dict:
    store = request.app.state.retriever.store
    return {"sources": store.sources(), "counts": store.stats()}


@app.get("/search")
def search(request: Request, q: str, k: int = 10, mode: str = "hybrid") -> dict:
    retriever: Retriever = request.app.state.retriever
    if mode == "vector" and retriever.index is None:
        mode = "fts"
    return {"query": q, "hits": [h.to_dict() for h in retriever.search(q, k=k, mode=mode)]}


@app.get("/ref/{ref:path}")
def get_ref(request: Request, ref: str) -> dict:
    passages = request.app.state.retriever.store.by_ref(ref)
    if not passages:
        raise HTTPException(404, f"{ref} is not in the corpus")
    return {"ref": ref, "texts": [{"lang": p.lang, "edition": p.edition, "text": p.text, "meta": p.meta}
                                  for p in passages]}


@app.post("/ask")
def ask(request: Request, body: AskRequest) -> dict:
    st = request.app.state
    retriever: Retriever = st.retriever
    mode = body.mode if (body.mode != "vector" or retriever.index is not None) else "fts"
    hits = retriever.search_balanced(body.question, k=body.k or st.top_k, mode=mode, collections=body.collections)
    response = {"question": body.question, "sources": [h.to_dict() for h in hits]}
    if not hits:
        response.update(answer=NOT_SURE[detect_lang(body.question)],
                        citation_check={"status": "pass", "cited": [], "issues": []})
        return response
    try:
        answer = st.llm.chat(build_messages(body.question, hits))
    except LLMError as e:
        response.update(answer=None, error=str(e), citation_check=None)
        return response
    report = check_answer(answer, [h.ref for h in hits], retriever.store)
    response.update(answer=answer, citation_check=report.to_dict())
    if report.status == "fail" and st.strict:
        # An invented reference or a misquote: don't show it as an answer.
        response.update(answer=NOT_SURE[detect_lang(body.question)], unverified_answer=answer)
    return response
