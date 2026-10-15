import json
import re

from app.citations import check_answer
from tools.build_sft import build


def test_template_examples_are_trainable_and_pass_citation_checks(store):
    examples = build(store, per_kind=12, seed=0)
    assert {e["meta"]["kind"] for e in examples} == {"quran_lookup", "hadith_lookup", "no_answer", "missing_ref"}
    for e in examples:
        assert all(store_source.endswith(("-ar", "-tanzil", "-pickthall")) for store_source in e["meta"]["sources"])
        user, answer = e["messages"][1]["content"], e["messages"][-1]["content"]
        evidence = re.findall(r"^\[([^\]]+)\]", user, re.M)
        report = check_answer(answer, evidence, store)
        assert report.status == "pass", (answer, report.to_dict())


def test_eval_questions_never_leak_into_training(store):
    gold = {json.loads(l)["query"] for l in open("data/eval/gold.jsonl", encoding="utf-8") if l.strip()}
    examples = build(store, per_kind=12, seed=1)
    questions = {e["messages"][1]["content"].rsplit("Question: ", 1)[1] for e in examples}
    assert not questions & gold


def test_gold_set_references_are_well_formed():
    from app.refs import MARKER

    rows = [json.loads(l) for l in open("data/eval/gold.jsonl", encoding="utf-8") if l.strip()]
    assert len(rows) >= 90 and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["lang"] in {"en", "bn", "ar"} and r["relevant"]
        assert all(MARKER.fullmatch(f"[{ref}]") for ref in r["relevant"])


def test_honesty_examples_say_not_sure_and_never_contain_the_asked_verse(store):
    from app.citations import says_not_sure

    examples = build(store, per_kind=12, seed=2)
    honest = [e for e in examples if e["meta"]["kind"] in ("no_answer", "missing_ref")]
    assert honest and all(says_not_sure(e["messages"][-1]["content"]) for e in honest)
    for e in honest:
        if e["meta"]["kind"] == "missing_ref":
            asked = re.search(r"(\d+:\d+)", e["messages"][1]["content"].rsplit("Question: ", 1)[1]).group(1)
            assert f"[quran:{asked}]" not in e["messages"][1]["content"]


def test_unanswerable_set_is_well_formed():
    rows = [json.loads(l) for l in open("data/eval/unanswerable.jsonl", encoding="utf-8") if l.strip()]
    assert len(rows) >= 10 and {r["lang"] for r in rows} == {"en", "bn", "ar"}


def test_examples_citing_held_out_sources_are_detected():
    from tools.build_sft import cites_any
    ex = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": 'Text [quran:2:255] and [bukhari:1].'}]}
    assert cites_any(ex, {"bukhari:1"})
    assert not cites_any(ex, {"bukhari:2"})
    assert cites_any({"messages": [{"role": "assistant", "content": "x [quran:2:255-256]"}]}, {"quran:2:256"})


def test_tafsir_on_a_held_out_verse_counts_as_citing_it():
    from types import SimpleNamespace
    from tools.build_sft import cites_any

    class FakeStore:
        def by_ref(self, ref):
            return [SimpleNamespace(meta={"surah": 2, "ayah": 255, "ayah_to": 256})]

    ex = {"messages": [{"role": "assistant", "content": "See [jalalayn:2:255]."}]}
    assert cites_any(ex, {"quran:2:256"}, FakeStore())
    assert not cites_any(ex, {"quran:2:257"}, FakeStore())
    assert not cites_any(ex, {"quran:2:256"})  # without a store only the cited refs themselves are compared
