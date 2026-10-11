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
