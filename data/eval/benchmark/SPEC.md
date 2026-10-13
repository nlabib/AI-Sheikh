# Benchmark item spec (one JSON object per line)

{"id": "<part>-<nnn>", "lang": "en|bn|ar", "category": "<one of below>", "difficulty": "easy|medium|hard",
 "question": "...", "expected": "answer|not_sure",
 "gold_answer": "2-4 sentence reference answer in the question's language, every claim backed by `sources`",
 "sources": ["quran:2:255", "bukhari:1", ...],          // refs that MUST be cited; verified to exist and to say what gold_answer claims
 "acceptable_sources": [...],                           // other refs that also correctly support it (optional)
 "must_mention": ["key fact or phrase", ...],           // 1-3 short checks a grader can look for
 "must_not": ["wrong claim to watch for", ...],         // optional: common mistakes / fabrications
 "needs_scholar_review": true|false,                    // true for any ruling (halal/haram/fard), disputed issue, or anything you are <95% sure of
 "notes": "why this is the right answer, differing views, numbering caveats"}

Categories: quran_locate (which verse says X), quran_content (what does verse/surah say), hadith_locate,
hadith_content, fiqh_ruling (single clear evidence), fiqh_disputed (scholars differ: gold answer must present
the views and their evidence, not pick one), aqeedah, tafsir (meaning / context of a verse), seerah (life of the
Prophet, answerable from Quran/hadith), akhlaq (manners/ethics), dua_dhikr, trick (fabricated hadith,
misattributed quote, false premise, nonexistent ref, out-of-scope) -> expected "not_sure", sources [].

Refs: quran:<surah>:<ayah>; hadith <collection>:<hadithnumber> with collection in
bukhari, muslim, abudawud, tirmidhi, nasai, ibnmajah (numbering = the corpus's `hadithnumber`; Muslim is the
sequential number, its Abd al-Baqi number is in meta alt_number).

Verify every ref against the local corpus before writing it:
  cd /home/claude/ai-sheikh && python3 -c "from app.store import Store; s=Store('data/corpus.sqlite'); [print(p.lang, p.meta, p.text[:600]) for p in s.by_ref('bukhari:1')]"
  Full-text search: Store.fts_search(query, limit=20) (see app/store.py), or app.retrieve.Retriever(store).search_balanced(q, k=10, mode='fts').
Never write a ref you have not read in the corpus. If you cannot find a supporting passage, drop the question.
Do not copy long passages into gold_answer; paraphrase briefly and cite.
