# Why is the non-English typographic attack weak? — Mechanism upgrade

*This extends `TYPOGRAPHIC_ATTACK_PROJECT.md`. The original project **reported** an
asymmetry (English text hijacks the model in any language; other languages barely work).
This upgrade **explains why** — turning a "look what happens" project into a "here's the
mechanism" project, which is much stronger for a conference. Every number below was run
and verified.*

---

## 0. The question (and the precise answer)

> *"Even with different languages, is the encoder the same? Is that why the attack is
> not very effective?"*

**Yes — there is exactly one image encoder, and it is identical no matter what language
you query in.** CLIP has two encoders: an **image** encoder (pixels → one vector) and a
**text** encoder (label words → vectors). Choosing a "language" only swaps the *text
labels*; it never changes how the image is processed. So a doctored image produces **one**
embedding, which is then compared to labels in whatever language you ask.

But that shared encoder is **why the English attack *transfers* across query languages —
not why the non-English attack fails.** Two distinct facts:

1. **Shared encoder ⇒ the query language barely matters.** In the original attack matrix,
   each *row* (the language you *write*) is roughly flat across *columns* (the language you
   *ask in*). Writing English "dog" gives ~36–55% success whether you ask in English,
   Korean, or Japanese. The variation is **between rows, not within rows** — the fingerprint
   of a single shared image encoder feeding cross-lingually-aligned labels.
2. **The non-English attack is weak for a different reason:** the single image encoder
   **can't read non-English text in pixels well.** It has an English-dominant "OCR" ability.

The rest of this document proves point 2 with three experiments.

---

## 1. The improved experiment: an "OCR probe"

**Idea.** To test what the encoder can *read*, remove the object entirely: render **only
the word** on a blank gray background and classify that. If the encoder reads the word, the
blank-text image lands on that word's concept. This isolates *reading ability* from the
object-vs-text competition.

**Result — reading ability by writing language** (multilingual CLIP, STL-10 words, 3 font
sizes each; "→ own-language label" = does the rendered word match its own language's label;
chance = 10%):

| writing language | script | reads → English concept | reads → own-language label |
|---|---|---|---|
| **English** | Latin | **100%** | 100% |
| German | Latin | 67% | 100% |
| French | Latin | 60% | 100% |
| Spanish | Latin | 40% | 93% |
| Russian | Cyrillic | 17% | 90% |
| Chinese | CJK | 23% | 30% |
| Korean | CJK | 20% | 13% |
| Japanese | CJK | 13% | 30% |
| Hindi | Devanagari | 7% | 13% |
| *(blank, no text)* | — | 10% | — |

> The encoder **reads European alphabetic scripts** (Latin + Cyrillic: 90–100% against their
> own labels) but **barely reads CJK / Devanagari** (13–30%). And it maps text to the shared
> *concept* most strongly for **English** (100%), partially for other European languages
> (40–67%), and weakly for everything else. (Fonts were visually verified — no missing-glyph
> boxes — so the low scores are real.)

---

## 2. Is it the SCRIPT or the WORD? (romanization test)

If we write a Korean/Japanese/Chinese word in **Latin letters** (e.g. 犬 → "inu", 개 → "gae",
狗 → "gou"), does the Latin script alone make it readable?

| condition | script | reads → English concept |
|---|---|---|
| English real words | Latin | **100%** |
| Spanish/French/German real words | Latin | 40–67% |
| **Romanized Korean/Japanese/Chinese** | Latin | **20–23%** |
| Original CJK | CJK | 13–23% |

**Conclusion: it's mostly the WORD, not the script.** Romanized non-English words (Latin
letters, but not real English words) read no better than the original CJK (~20%), and far
worse than English (100%). Latin script alone buys almost nothing; being a **real,
frequently-seen word — especially an English one — is what matters.** (European languages
sit in between because words like "auto", "chat", "Hund" are real text the model saw on the
web; invented romanizations like "gae" are not.)

---

## 3. Reading ability → attack strength (override dose-response)

Reading a word *in isolation* isn't enough for an attack — the written word must be strong
enough to **override the real object** in the photo. We sweep font size on real images and
measure attack success (prediction flips to the written word, English query):

| font size | English | Spanish | German | Korean | Japanese |
|---|---|---|---|---|---|
| 16 | 39% | 2% | 2% | 1% | 1% |
| 24 | 47% | 2% | 2% | 0% | 1% |
| 32 | 50% | 3% | 3% | 1% | 1% |
| 40 | 61% | 3% | 4% | 1% | 1% |
| 52 | **71%** | 3% | 4% | 1% | 2% |

> **Even huge non-English text never overrides the object.** English climbs 39% → 71% with
> size; everything else stays flat at ≤4%. So the weakness is **not** about text size — it's
> that only English text is "read" strongly enough to beat the real image content. (Note
> Spanish/French are 40–60% readable *in isolation* yet ~3% as attackers: readable but too
> weak to win the tug-of-war against a real object.)

---

## 4. The full mechanism (the upgraded story)

1. **One shared image encoder.** The query language only swaps text labels; it never changes
   image processing. ⇒ attack strength depends on the **written** language, not the queried
   one (flat matrix rows).
2. **The encoder's text-reading is English-dominant.** OCR probe: English 100% → other
   European 40–67% → CJK/Cyrillic/Devanagari/romanized 7–23%.
3. **It's vocabulary/frequency, not just script.** Romanizing into Latin doesn't help;
   real (especially English) words do.
4. **Attacks need to override the object.** Only English text is read strongly enough to
   win against the real image content — even giant non-English text fails.

**So the answer to the original question:** the shared encoder is why the *English* attack
works in every language; the non-English attack fails because that *same* single encoder
was trained on web images whose in-image text is overwhelmingly English, so it learned to
read English (and to a lesser degree other major European languages) but not other scripts.

---

## 5. Why this is a better project

The original project said *"English text hijacks multilingual AI in any language."* The
upgrade adds the **causal explanation** and rules out alternatives:

- It's **not** the query language (shared encoder; flat rows).
- It's **not** the script per se (romanization doesn't help).
- It's **not** text size (giant non-English text still fails).
- It **is** the encoder's English-biased, frequency-driven text-reading ability.

That is a genuine, testable, mechanistic finding — exactly what lifts a science-fair entry
from "demo" to "explanation."

### New questions this opens (great extensions)
- Does an encoder trained with more multilingual web data (e.g. a SigLIP or a CJK-heavy
  CLIP) read CJK text better? (Predicts the asymmetry shrinks.)
- Does the effect track each language's **frequency of in-image text** on the web?
- Can you **defend** by OCR-detecting and masking text regions before classifying?
  (Unlike the multilingual ensemble, which we showed does *not* defend.)

---

## 6. Reproducibility

| file | what it runs |
|---|---|
| `typographic_attack.py` | original cross-lingual attack matrix + font-size sweep |
| `mechanism_experiment.py` | the OCR probe + romanization test (Section 1–2) |
| `override_doseresponse.py` | reading-vs-attack override sweep (Section 3) |
| `results/ocr_probe_strip.png` | visual proof the fonts render (no missing glyphs) |
| `results/mechanism.json`, `results/override.json` | the raw numbers above |

All run in minutes on one GPU (or free Colab); the OCR probe needs **no gradients**.
