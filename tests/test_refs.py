from app.refs import expand_marker, find_explicit_refs, find_markers, hadith_ref, parse_ref


def test_explicit_references_in_three_languages():
    assert find_explicit_refs("Explain 2:255 please") == ["quran:2:255"]
    assert find_explicit_refs("আয়াত ২:২৫৫") == ["quran:2:255"]
    assert find_explicit_refs("Bukhari 6018 and Sahih Muslim 12") == ["bukhari:6018", "muslim:12"]
    assert find_explicit_refs("বুখারী ১") == ["bukhari:1"]


def test_surah_numbers_out_of_range_are_ignored():
    assert find_explicit_refs("at 10:30 tomorrow, 200:1") == ["quran:10:30"]


def test_markers_and_ranges():
    text = "First [quran:2:255]. Then [bukhari:1] and [quran:94:5-6]. Not [Q 2:255]."
    assert find_markers(text) == ["quran:2:255", "bukhari:1", "quran:94:5-6"]
    assert expand_marker("quran:94:5-6") == ["quran:94:5", "quran:94:6"]


def test_fractional_hadith_numbers():
    assert hadith_ref("bukhari", 402.2) == "bukhari:402.2"
    assert hadith_ref("bukhari", 7.0) == "bukhari:7"
    assert find_markers("see [bukhari:402.2]") == ["bukhari:402.2"]
    assert parse_ref("muslim:4927").label == "Sahih Muslim 4927"


def test_reference_phrasings_people_use():
    assert find_explicit_refs("What is Sahih Bukhari hadith number 1?") == ["bukhari:1"]
    assert find_explicit_refs("Sunan Abi Dawud 4031") == ["abudawud:4031"]
    assert find_explicit_refs("مسلم حديث رقم ٣٩٤") == ["muslim:394"]


def test_short_surahs_by_name_or_number_are_given_whole():
    ikhlas = ["quran:112:1", "quran:112:2", "quran:112:3", "quran:112:4"]
    assert find_explicit_refs("Show me Surah Al-Ikhlas.") == ikhlas
    assert find_explicit_refs("surah 112") == ikhlas
    assert find_explicit_refs("সূরা ১১২") == ikhlas
    assert len(find_explicit_refs("is surah fathia mandatory in every prayer")) == 7  # typo, Al-Faatiha
    assert len(find_explicit_refs("surah ikhlas and surah nas")) == 10
    assert find_explicit_refs("surah al-baqarah") == []  # too long to hand over whole
    assert find_explicit_refs("which surah talks about patience") == []
    assert find_explicit_refs("surah 2:255") == ["quran:2:255"]
