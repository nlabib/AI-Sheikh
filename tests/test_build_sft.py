import json
import re

from app.citations import check_answer
from tools.build_sft import build


def test_template_examples_are_trainable_and_pass_citation_checks(store):
    examples = build(store, per_kind=12, seed=0)
    assert {e["meta"]["kind"] for e in examples} == {"quran_lookup", "hadith_lookup", "no_answer"}
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
