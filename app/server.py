"""FastAPI app.

    uvicorn app.server:app --host 127.0.0.1 --port 8000

Endpoints: POST /ask, GET /search, GET /ref/{ref}, GET /health, GET /sources.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.citations import check_answer
from app.config import get_settings
from app.embed import VectorIndex, make_embedder
from app.llm import ChatClient, LLMError
from app.prompts import INTRO, NO_SOURCES, NOT_SURE, SALAM_REPLY, build_messages
from app.rerank import make_reranker
from app.retrieve import Retriever
from app.store import Store
from app.text import detect_lang


def build_retriever(settings) -> tuple[Retriever, str]:
    store = Store(settings.db_path)
    reranker = make_reranker(settings.reranker)
    rerank_note = f", reranked by {settings.reranker}" if reranker else ""
    index = VectorIndex.load(settings.index_dir)
    if index is None:
        return (Retriever(store, reranker=reranker, rerank_pool=settings.rerank_pool),
                f"fts only (no vector index; run `python -m tools.index`){rerank_note}")
    embedder_kind = "hashing" if index.embedder_name.startswith("hashing") else "e5"
    if embedder_kind == "hashing" and settings.embedder != "hashing":
        # The hashing fallback makes hybrid results worse than FTS alone (docs/RESULTS.md).
        return (Retriever(store, reranker=reranker, rerank_pool=settings.rerank_pool),
                f"fts only (ignoring hashing index; set EMBEDDER=hashing to use it){rerank_note}")
    embedder = make_embedder(embedder_kind, index.embedder_name if embedder_kind == "e5" else "")
    return (Retriever(store, embedder, index, reranker=reranker, rerank_pool=settings.rerank_pool),
            f"hybrid (fts + {index.embedder_name}){rerank_note}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not hasattr(app.state, "retriever"):
        app.state.retriever, app.state.retrieval_mode = build_retriever(settings)
    if not hasattr(app.state, "llm"):
        app.state.llm = ChatClient(settings)
    app.state.top_k = settings.top_k
    app.state.strict = getattr(app.state, "strict", settings.strict_citations)
    app.state.min_relevance = getattr(app.state, "min_relevance", settings.relevance_floor)
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


_SALAM = re.compile(r"(salaa?m|assalamu|as-salamu|السلام عليكم|সালাম)", re.I)
_SMALL_TALK = re.compile(
    r"^\W*(?:(?:hi|hello|hey|salaa?m|(?:as)?-?salaa?mu?\s*(?:'?alaikum|alaykum)(?:\s*wa\s*rahmatullah\w*)?|"
    r"السلام عليكم(?: ورحمة الله)?|মারহাবা|হ্যালো|আসসালামু আলাইকুম|সালাম|مرحبا|أهلا)\W*)?"
    r"(?:who are you|what are you|what can you do|তুমি কে|আপনি কে|আপনি কী করতে পারেন|من أنت|ماذا تستطيع)?\W*$",
    re.I)


def small_talk_reply(question: str) -> str | None:
    """Greetings and "who are you": answered with a fixed introduction instead of the model, which
    otherwise makes up a persona."""
    q = question.strip()
    if not q or not _SMALL_TALK.match(q) or not re.search(r"\w", q):
        return None
    lang = detect_lang(q)
    return (SALAM_REPLY[lang] if _SALAM.search(q) else "") + INTRO[lang]


def tidy(answer: str, lang: str) -> str:
    # The Bangla full stop leaks into English and Arabic answers from the training data.
    return answer.replace("।", ".") if lang != "bn" else answer


def answer_question(retriever: Retriever, llm, question: str, k: int, collections: list[str] | None = None,
                    mode: str = "hybrid", strict: bool = True, min_relevance: float = 0.0) -> dict:
    """The whole /ask pipeline: retrieve, drop passages that aren't about the question, ask the model,
    check its citations, and replace a failing answer with "I'm not sure". tools.eval_answers uses it too,
    so evaluation sees exactly what a user sees.

    min_relevance is the floor for the relevance score in use: the reranker's (0-1) when the retriever has
    one, otherwise E5 similarity."""
    lang = detect_lang(question)
    intro = small_talk_reply(question)
    if intro:
        return {"question": question, "sources": [], "answer": intro,
                "citation_check": {"status": "pass", "cited": [], "issues": []}}
    mode = mode if (mode != "vector" or retriever.index is not None) else "fts"
    hits = retriever.search_balanced(question, k=k, mode=mode, collections=collections,
                                     min_rerank=min_relevance if retriever.reranker is not None else 0.0)
    if retriever.reranker is not None:
        relevance = {h.ref: h.rerank for h in hits if h.rerank is not None}
    else:
        relevance = retriever.relevance(question, [h.ref for h in hits])
    for h in hits:
        if "reference" in h.matched_by:
            relevance[h.ref] = 1.0  # the user asked for it by number
    if min_relevance > 0 and relevance:
        # Passages that aren't about the question only invite the model to cite them anyway.
        hits = [h for h in hits if relevance.get(h.ref, 1.0) >= min_relevance]
    response = {"question": question, "sources": [dict(h.to_dict(), relevance=relevance.get(h.ref))
                                                   for h in hits]}
    if not hits:
        response.update(answer=NO_SOURCES[lang] if min_relevance > 0 else NOT_SURE[lang],
                        citation_check={"status": "pass", "cited": [], "issues": []})
        return response
    try:
        answer = tidy(llm.chat(build_messages(question, hits)), lang)
    except LLMError as e:
        response.update(answer=None, error=str(e), citation_check=None)
        return response
    report = check_answer(answer, [h.ref for h in hits], retriever.store, question_lang=lang,
                          relevance=relevance, min_relevance=min_relevance)
    response.update(answer=answer, citation_check=report.to_dict())
    if report.status == "fail" and strict:
        # An invented reference, a misquote, an answer from memory: don't show it as an answer.
        response.update(answer=NOT_SURE[lang], unverified_answer=answer)
    return response


@app.post("/ask")
def ask(request: Request, body: AskRequest) -> dict:
    st = request.app.state
    return answer_question(st.retriever, st.llm, body.question, k=body.k or st.top_k,
                           collections=body.collections, mode=body.mode, strict=st.strict,
                           min_relevance=st.min_relevance)
