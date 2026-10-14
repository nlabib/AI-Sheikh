from app.prompts import build_messages


def test_prompt_text_is_nfc_normalized_like_qwens_own_tokenizer():
    # The corpus stores য় as one code point (U+09DF). Qwen's tokenizer normalizes it to য + nukta before
    # tokenizing; llama.cpp doesn't, so the prompt must already be normalized (see build_messages).
    hit = {"ref": "quran:2:275", "label": "Quran 2:275", "meta": {},
           "texts": [{"lang": "bn", "text": "যারা সুদ খায়"}]}
    user = build_messages("সুদ কি হারাম?", [hit])[1]["content"]
    assert "য়" not in user and "য়" in user
