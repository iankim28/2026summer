# Does AI Read English Better Than Your Language?
### Cross-Lingual Typographic Attacks on Multilingual Vision-Language AI

*A high-school research project proposal. All "preliminary results" below were already
run and verified, so the project is known to be feasible.*

---

## 1. One-paragraph summary

Modern image-recognition AI (like CLIP) can label a picture using words in *many*
languages, because it turns the picture into a single set of numbers (an "embedding")
and compares it to the words. We ask a simple, visual question: **if you write a wrong
word onto a picture, does the AI believe the word instead of its own eyes — and does
this work in every language?** We find a striking, lopsided answer: **English text
written on an image hijacks the AI's prediction even when you ask it in Korean,
Japanese, Spanish, or French — but the same word written in those languages barely
fools it at all.** The AI "reads" English in images and mostly ignores other scripts.

---

## 2. Background (the two ideas you need)

**Vision-language AI (CLIP).** CLIP looks at an image and a list of candidate labels
("a photo of a *dog*", "a photo of a *cat*", …) and picks the label whose meaning is
closest to the image. A **multilingual** CLIP can do this with labels written in ~100
languages, because one shared "image brain" (the image encoder) is matched against
labels in any language.

**Typographic attacks.** CLIP has a famous weakness: if you *write a word on the image*,
CLIP often reads the word and reports it instead of what's actually pictured. The classic
example (OpenAI, 2021): tape a paper saying "iPod" on an apple, and CLIP says "iPod."
This is called a **typographic attack** — an attack that needs **no math, no gradients,
just a marker**. (See Related Work for the research lineage.)

**The gap we fill.** Typographic attacks are studied mostly in English. Nobody has
cleanly asked, on a *multilingual* model: *which written language is the strongest
attacker, and does an attack in one language carry over to questions asked in another?*
That cross-lingual question is our contribution.

---

## 3. Research questions & hypotheses

- **RQ1.** Does writing a misleading word on an image flip a multilingual CLIP's
  prediction to that word? *(H1: yes — the known typographic weakness.)*
- **RQ2.** Is the attack **cross-lingual** — does English text fool the model when we ask
  in Korean/Japanese/etc., and vice-versa? *(H2: there will be an asymmetry; English
  text will be the strongest attacker because the AI saw mostly English text in images
  during training.)*
- **RQ3.** Does the attack get **stronger as the text gets bigger**? *(H3: yes — a clean
  dose-response curve.)*
- **RQ4.** Can a simple **defense** stop it — e.g., averaging the answers from five
  languages (an "ensemble")? *(H4: no — if English text fools every language, averaging
  languages won't help.)*

---

## 4. Preliminary results (already run — proof it works)

**Setup.** Model: `open_clip xlm-roberta-base-ViT-B-32` (a real multilingual CLIP).
Images: 300 photos from the STL-10 dataset (10 everyday classes: airplane, bird, car,
cat, deer, dog, horse, monkey, ship, truck). For each image we wrote the name of a
*different* (wrong) class onto it, in white text on a white box, then asked CLIP to
classify it. **"Attack success" = the prediction flips to the written (wrong) word.**

**4.1 The AI is accurate before any attack** (sanity check):

| en | ko | es | fr | ja |
|----|----|----|----|----|
| 96.0% | 87.7% | 96.7% | 95.7% | 95.3% |

**4.2 The cross-lingual attack matrix** — attack-success rate by *written-text language*
(rows) × *language we asked in* (columns):

| written ↓ / asked → | en | ko | es | fr | ja |
|---|---|---|---|---|---|
| **English** | **55%** | **51%** | **41%** | **36%** | **49%** |
| Korean | 1% | 2% | 1% | 1% | 2% |
| Spanish | 3% | 4% | 5% | 3% | 2% |
| French | 4% | 4% | 4% | 5% | 3% |
| Japanese | 1% | 2% | 2% | 2% | 2% |

> **The headline:** the English row is huge across *every* column; every other row is
> tiny. Writing **English** "dog" on a cat photo fools the model even when we ask in
> Korean (51%). Writing "개" (Korean for dog) fools almost nothing (≤2%) — *even when we
> ask in Korean.* The model reads English in images and largely ignores other scripts.
> (We verified the Korean/Japanese fonts render correctly, so this is a real effect, not
> a broken-font artifact.)

**4.3 Bigger text = stronger attack** (controlled-variable experiment). Attack success
vs. font size, for English text:

| font size | 10 | 16 | 22 | 28 | 36 | 44 |
|---|---|---|---|---|---|---|
| asked in English | 18% | 39% | 43% | 50% | 56% | 66% |
| 5-language ensemble | 13% | 30% | 33% | 42% | 49% | 56% |

**4.4 A simple defense does not work.** Averaging the predictions of all five languages
(the "ensemble" row above) gets fooled almost as much as English alone — because the
English text fools every language at once. So multilingual voting is *not* a defense.

---

## 5. Why this matters

- **Security.** Anyone can attack this AI with a *printer and tape* — no coding, no
  access to the model. A sticker with an English word could mislead an image-tagging or
  content-moderation system.
- **Fairness / language bias.** The AI behaves as if **English is the "real" language**
  inside images. Speakers of other languages get an AI that both understands their words
  less *and* can be attacked *through* English regardless of the language they use. This
  is a concrete, measurable form of language inequity in AI.
- **It's surprising and clear.** A one-sentence takeaway ("English text can hijack the AI
  in any language") backed by a clean matrix and a dose-response curve.

---

## 6. Method — how to run it (accessible, no calculus)

You do **not** need gradient descent or GPUs for this project; it runs on **free Google
Colab** (or even a laptop CPU, just slower).

1. **Load a multilingual CLIP** with the `open_clip` library
   (`xlm-roberta-base-ViT-B-32`).
2. **Get images** with clear labels (STL-10 or CIFAR-10 — both download in one line).
3. **Write a word on an image** using Python's Pillow (`PIL`) library: `ImageDraw.text(...)`
   with a font that supports all scripts (`NotoSansCJK`, which is free).
4. **Classify**: encode the image, compare to the label words in each language (cosine
   similarity), take the closest — all a few lines with `open_clip`.
5. **Measure** the attack-success rate and make the matrix / curves with `matplotlib`.

*(A working reference implementation already exists: `typographic_attack.py`. It can be
turned into a clean, student-friendly Colab notebook on request.)*

### The four experiments to run
1. **Baseline & matrix (RQ1–RQ2):** the 5×5 (or more) language matrix above.
2. **Dose-response (RQ3):** sweep font size; plot the curve.
3. **Defense test (RQ4):** compare single-language vs. 5-language-ensemble attack success.
4. *(Optional, advanced)* **Where/how does it work:** vary text color, position, or use
   a real-world phone-photo of a printed sticker to test physical-world robustness.

---

## 7. Suggested timeline (≈ 4–6 weeks, part-time)

| Week | Goal |
|---|---|
| 1 | Set up Colab; load CLIP; classify clean images in English; reproduce ~96% accuracy. |
| 2 | Add multilingual labels (start with en/ko/es/fr/ja); confirm per-language accuracy. |
| 3 | Write text on images with Pillow; reproduce the single-language typographic attack. |
| 4 | Build the full **cross-lingual matrix**; verify fonts render; find the asymmetry. |
| 5 | Font-size dose-response + the ensemble-defense test; make all figures. |
| 6 | Write up: poster/paper, example images, related-work section, conclusions. |

---

## 8. What makes it a *good* science-fair project

- **A clear hypothesis** with a surprising, falsifiable prediction (the asymmetry).
- **Controlled variables** (font size, written language, query language).
- **Honest negative result** (the ensemble defense fails) — graders love this.
- **Reproducible** and cheap (free Colab, public model, public data).
- **Great visuals** (show the actual doctored images next to the AI's wrong answers).

---

## 9. Limitations & honest caveats (put these in the paper)

- Results are on one model (`xlm-roberta-base-ViT-B-32`) and easy 10-class datasets;
  a stretch goal is to test a second multilingual model to show it generalizes.
- The white-box-on-image text is deliberately easy to read; real attacks may be subtler.
- Translations of class names must be checked by a fluent speaker (a confound if wrong).
- "Attack success" here means the prediction flips to the *written* word; also report
  plain "fooled" (prediction ≠ true class) so the framing is transparent.

---

## 10. Related work (for the paper's background section)

- Goh et al., *Multimodal Neurons in Artificial Neural Networks* (Distill, 2021) — the
  original typographic attack on CLIP (the "iPod apple").
- Azuma & Matsui / Materzyńska et al. — typographic-attack analyses and defenses for CLIP.
- *Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks* (2024) —
  https://arxiv.org/abs/2402.00626
- *Typographic Attacks in a Multi-Image Setting* (2025) — https://arxiv.org/abs/2502.08193
- *Towards Mechanistic Defenses Against Typographic Attacks in CLIP* (2025) —
  https://arxiv.org/html/2508.20570v1
- *Web Artifact Attacks Disrupt Vision-Language Models* (2025) — notes that multilingual
  text artifacts (Arabic, Latin, Hindi) can shift predictions.

*Our angle vs. prior work:* we run a **controlled cross-lingual study on a multilingual
model**, measuring which written language is the strongest attacker and whether attacks
transfer across the language you ask in — and we test multilingual ensembling as a
(failed) defense.

---

## 11. One-line elevator pitch

> *"We showed that multilingual AI reads English text inside images and can be hijacked
> by it in any language — a printable, no-code attack that also reveals a hidden English
> bias in 'multilingual' AI."*
