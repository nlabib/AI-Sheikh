"""Prompt construction. The training data builder uses the same functions, so a
fine-tuned adapter learns exactly the format the server sends at inference time."""

from __future__ import annotations

from app.text import detect_lang

SYSTEM_PROMPT = """You answer questions about Islam using ONLY the numbered sources provided.

Rules:
1. Every factual sentence ends with the marker of the source that supports it, copied exactly, e.g. [quran:2:255] or [bukhari:1].
2. Quote source text only word for word, inside double quotes, followed by its marker.
3. Never cite a marker that is not in the sources. Never invent verse or hadith numbers.
4. If the sources do not answer the question, say so plainly and suggest asking a qualified scholar. Do not answer from memory.
5. Do not issue fatwas. For rulings that depend on circumstances or differ between schools, present what the sources say and recommend a qualified scholar.
6. Answer in the language of the question ({lang_name})."""

LANG_NAMES = {"en": "English", "bn": "Bangla", "ar": "Arabic"}


def format_evidence(hits: list, max_chars: int = 1200) -> str:
    """hits: retrieve.Hit objects or dicts from Hit.to_dict()."""
    blocks = []
    for h in hits:
        d = h.to_dict() if hasattr(h, "to_dict") else h
        lines = [f"[{d['ref']}] {d['label']}"]
        meta = d.get("meta") or {}
        if meta.get("grades"):
            lines.append("Grading: " + "; ".join(f"{g['grade']} ({g['name']})" for g in meta["grades"][:3]))
        for t in d["texts"]:
            text = t["text"] if len(t["text"]) <= max_chars else t["text"][:max_chars] + " …"
            lines.append(f"({t['lang']}) {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list) -> list[dict]:
    lang = detect_lang(question)
    system = SYSTEM_PROMPT.format(lang_name=LANG_NAMES.get(lang, "English"))
    user = f"Sources:\n\n{format_evidence(hits) or '(no sources found)'}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
