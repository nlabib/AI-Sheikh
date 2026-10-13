# Test: questions about Surah al-Fatiha in prayer

2026-10-12. Model `ai-sheikh-ft` (Qwen3-4B fine-tuned for 144 steps, f16 GGUF) in LM Studio, called
through the app's `/ask` endpoint: hybrid retrieval, 6 sources (3 Quran + 3 hadith), non-thinking
mode, `STRICT_CITATIONS=1` (an answer that fails the citation check is replaced by "I'm not sure").
I checked every quote against the stored text of the cited hadith.

## Retrieval: `/search?q=Is Surah al-Fatiha mandatory in every prayer?`

Top 10 (whole corpus, unscoped): `muslim:883`, `bukhari:772`, `muslim:884`, `bukhari:776`, `bukhari:778`,
`bukhari:759`, `bukhari:762`, **`bukhari:756`** (rank 8), `ibnmajah:839`, `muslim:882`.

The top 5 are all on topic (Abu Huraira on reciting in every prayer; the Prophet's recitation in Zuhr),
but the clearest hadith, Bukhari 756 "Whoever does not recite Al-Fatiha in his prayer, his prayer is
invalid", only comes 8th. **Muslim 394 is not in the top 10.** This corpus numbers Muslim in its own
sequence, with the familiar Abd al-Baqi number in the metadata (`alt_number`). Muslim 394a–d is
`muslim:874`–`muslim:877` here (Ubada: "He who does not recite Fatihat al-Kitab is not credited with
having observed the prayer"). `muslim:882`–`884`, which did come up, are Muslim 396a–c (Abu Huraira).

## The seven questions

| # | Question | Sources /ask gave the model | Cited | Check | Replaced? |
|---|---|---|---|---|---|
| 1 | Is Surah al-Fatiha mandatory in every prayer? | quran:3:30, muslim:883, quran:6:108, bukhari:772, quran:22:27, muslim:884 | muslim:883, bukhari:772 | warn | no |
| 2 | Is it obligatory to recite al-Fatiha in every rak'ah of salah? | quran:26:222, muslim:883, quran:3:30, bukhari:776, quran:22:27, muslim:884 | muslim:883 | warn | no |
| 3 | What did the Prophet say about a prayer without al-Fatiha? | quran:25:77, muslim:882, quran:24:58, bukhari:1335, quran:10:89, **bukhari:756** | bukhari:756 | **pass** | no |
| 4 | Should someone praying behind an imam recite al-Fatiha? | quran:4:9, **abudawud:823**, quran:9:108, bukhari:1335, quran:45:8, muslim:883 | muslim:883 | warn | no |
| 5 | প্রতি নামাজে কি সূরা ফাতিহা পড়া ফরজ? | quran:89:10, tirmidhi:1027, quran:2:23, bukhari:1335, quran:95:8, tirmidhi:311 | none | fail | **yes** |
| 6 | هل قراءة الفاتحة واجبة في كل صلاة؟ | quran:36:83, ibnmajah:841, quran:21:96, ibnmajah:840, quran:39:62, nasai:351 | ibnmajah:841 | pass | no |
| 7 | Which hadith says Fatiha is optional? (trick) | quran:39:23, muslim:884, quran:5:97, bukhari:776, quran:5:19, muslim:883 | none | pass | no |

### Answers

**1.** "Yes, Surah al-Fatiha is mandatory in every prayer. According to Sahih Muslim 883, Abu Huraira
said that one should recite Surah al-Fatiha in every rak'ah of prayer [muslim:883]. Similarly, Sahih
al-Bukhari 772 states that the Qur'an is recited in every prayer ... If you recite "Al-Fatiha" only it is
sufficient [bukhari:772]."
Faithful to both sources; the quote from Bukhari 772 is exact. Note that both are Abu Huraira's own
words, not the Prophet's. "warn" because the opening "Yes, ... mandatory" sentence has no citation.
Muslim 884, the most explicit source it had ("Recitation (of Surat al-Fatiha) in every (rak'ah) of
prayer is essential"), was not used. Three of the six source slots were unrelated Quran verses.

**2.** "Yes, according to Sahih Muslim 883, Abu Huraira said that one should recite (al-Fatiha) in
every rak'ah of prayer ... [muslim:883]." Faithful. "warn" for the same uncited opening.

**3.** "The Prophet Muhammad (ﷺ) said: "Whoever does not recite Al-Fatiha in his prayer, his prayer is
invalid" [bukhari:756]." **Exact quote, correct source, the best answer of the set.**

**4.** "Yes, a person praying behind an imam should recite al-Fatiha. According to Sahih Muslim 883 ...
one should recite al-Fatiha in every rak'ah of prayer [muslim:883]."
Weak. Muslim 883 is general and says nothing about praying behind an imam. The source that answers
it directly was in front of the model and ignored: Abu Dawud 823 (the Prophet: "Do not do so except
when it is Fatihat al-Kitab"). The answer also gives a flat "Yes" on a question the schools of law
disagree on, without mentioning that.

**5.** (Bangla) Shown: "আমি নিশ্চিত নই। আমাকে দেওয়া উৎসগুলোতে এর উত্তর নেই, তাই আমি অনুমান করব না। ..." (I'm not sure.)
The model's own answer, which the server hid: "হ্যাঁ, প্রতি নামাজে সূরা ফাতিহা পড়া ফরজ। এটি আমাকে দেওয়া উৎসগুলোতে অনুমান করব না।"
("Yes, reciting al-Fatiha is fard in every prayer." followed by a garbled sentence), with no citation.
Retrieval failed here: none of the sources is about al-Fatiha except tirmidhi:311, at the end. The
model answered from memory, the citation check failed it (`unsupported_answer`), and the safety net
turned it into "I'm not sure". **The safety net did its job; the search did not.**

**6.** (Arabic) "نعم، فقراءة الفاتحة واجبة في كل صلاة [ibnmajah:841]۔"
Ibn Majah 841: "Every prayer in which Fatihat al-Kitab is not recited is deficient (خداج), deficient."
The source is right and on point, but "deficient" is softer than "واجبة" (obligatory), so the answer
says more than the source does. It ends with a stray Bangla full stop "।".

**7.** "I'm not sure. The sources I was given do not include a hadith that states Surah al-Fatiha is
optional, so I cannot quote or explain it. ..." **Correct: there is no such hadith, and it didn't invent one.**

## How to ask: what works better

I tried three rephrasings:

| Question | Result |
|---|---|
| সূরা ফাতিহা ছাড়া নামাজ সম্পর্কে রাসূল (সা.) কী বলেছেন? ("What did the Prophet say about prayer without al-Fatiha?") | **pass**, cites tirmidhi:247: "যে ব্যক্তি সূরা ফাতিহা পাঠ করেনি তার নামায হয়নি" (exact) |
| Q1 with `"collections": ["bukhari","muslim","abudawud","tirmidhi","nasai","ibnmajah"]` | all 6 sources on topic (muslim:883, bukhari:772, muslim:884, bukhari:776, bukhari:778, bukhari:759); answer same as Q2, warn |
| What did the Prophet say about reciting behind the imam? | pass, cites ibnmajah:850: "Whoever has an Imam, the recitation of the Imam is his recitation." (exact). But abudawud:823, which says the opposite (recite al-Fatiha even behind the imam), was also among the sources and not mentioned |

1. **Ask what the sources say, not for a ruling.** "What did the Prophet say about a prayer without
   al-Fatiha?" got exact, cited quotes in English and Bangla. "Is it fard/mandatory?" invites a
   yes/no verdict with an uncited opening sentence, and in Bangla the word "ফরজ" sent the search off course.
2. **Name the collection when you know it's hadith.** `"collections": [...]` stops three Quran
   verses from taking half the source slots on a question the Quran doesn't address directly.
3. **On questions where scholars differ** (reciting behind the imam), the model quotes one narration
   and ignores the other. Ask "What narrations are there about ...?", and read the `sources` list
   yourself. This is the single-source habit the training templates taught; see `docs/RESULTS.md`.

## What this shows about the model

- It never invented a reference in these 10 questions, and quotes were word for word every time
  it quoted.
- It says "I'm not sure" when the sources don't support an answer (Q7), and when it does guess
  without a source (Q5), the citation check catches it.
- Weak points: it uses one source even when several are relevant or they disagree; it opens with an
  uncited "Yes"; and Bangla retrieval for ruling-style questions is poor.
