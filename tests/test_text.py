from app.text import detect_lang, normalize, tokenize


def test_arabic_normalisation_strips_harakat_and_folds_letters():
    assert normalize("إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ") == "انما الاعمال بالنيات"
    assert normalize("رَحْمَة") == "رحمه"


def test_bangla_words_stay_whole():
    # Python's \w and SQLite's default tokenizer both split these on vowel signs.
    assert tokenize("আল্লাহর রাসূল বলেছেন নিয়্যাত") == ["আল্লাহর", "রাসূল", "বলেছেন", "নিয়্যাত"]


def test_digits_are_unified():
    assert normalize("২:২৫৫") == "2:255"
    assert normalize("٢:٢٥٥") == "2:255"


def test_language_detection():
    assert detect_lang("What is zakat?") == "en"
    assert detect_lang("যাকাত কী?") == "bn"
    assert detect_lang("ما هي الزكاة؟") == "ar"


def test_stopwords_dropped_only_when_asked():
    assert "the" in tokenize("the verse of the throne")
    assert tokenize("what does the quran say about fasting", drop_stopwords=True) == ["fasting"]
