# Reading is Utility *and* Vulnerability
### A cross-lingual fairness↔security tradeoff in multilingual vision-language models

*Research-direction proposal, extending our findings in `TYPOGRAPHIC_MECHANISM_UPGRADE.md`
and `SHARED_VS_SEPARATE.md`. Reframes a security result as a fairness/policy result:
making a "multilingual" image encoder read more scripts (a fairness goal) necessarily
enlarges its typographic attack surface (a security cost) — and the natural defense
inherits the same language bias.*

---

## 1. One-line thesis

> **The very capability that makes a multilingual VLM useful to non-English speakers —
> reading text in their script — is exactly what an attacker exploits. So a language's
> current robustness to typographic attacks is just the encoder's *inability to read it*,
> and closing that reading gap (fairness) enlarges the attack surface (security), while
> OCR-based defenses protect high-resource scripts more than low-resource ones.**

---

## 2. Why this holds — grounding in what we already measured

The causal chain **reading ability → typographic vulnerability** is visible in our data:

| language | OCR-probe reading (→ concept) | typographic attack success (ASR) |
|---|---|---|
| English | 100% | 36–55% |
| German / French / Spanish | 40–67% | 3–5% |
| Korean / Japanese / Chinese | 7–23% | 1–5% |
| Hindi (Devanagari) | 7% | ~1% |

- **The strongest attacker is English, the best-read language.** Non-English text is a weak
  attacker *because the encoder barely reads it* — not because those languages are safe.
- On **separate** per-language encoders, the **Japanese encoder is immune to English text
  (ASR 5%)** — its "robustness" is literally its refusal/inability to read English (it was
  trained on Japanese web data). Robustness = a reading limitation.
- Romanization test: writing non-English words in Latin letters does **not** make them
  strong attackers → it is **vocabulary/training-frequency**, not script.

So "which languages are safe from typographic attacks?" has the uncomfortable answer:
**the ones the model can't yet read for its legitimate users.**

---

## 3. The two halves of the tradeoff

### 3a. Improving multilingual reading *enlarges* the attack surface
Reading text-in-images is a real utility goal (signs, menus, documents, accessibility) and
a known fairness gap for non-English scripts. But because typographic ASR is *mediated by*
reading ability, any progress on multilingual reading turns more scripts into usable attack
vectors. You cannot buy the utility without buying the vulnerability — they are the same
capability.

### 3b. The OCR-mask defense inherits the same bias
The natural defense is **OCR-detect-and-mask**: find in-image text, inpaint it, then
classify (cf. Azuma & Matsui, *Defense-Prefix*, ICCVW 2023 — monolingual). But OCR engines
are themselves English/Latin-biased, so the defense:
- removes **English** text well → protects English-based attacks,
- **misses** low-resource-script text → leaves those attacks intact.

**Double asymmetry ("safety debt"):** non-English contexts get *both* a less capable model
*and* weaker protection once defenses ship. Improving inclusivity on the model side without
co-improving defense coverage **transfers risk onto the very users being included.**

---

## 4. Research questions & hypotheses

- **RQ1.** Does OCR-reading ability *mediate* typographic ASR across encoders and languages?
  *(H1: yes — ASR is a monotone function of reading ability; the JA-immune point anchors it.)*
- **RQ2.** Does making an encoder read a script better *causally* raise its ASR for that
  script? *(H2: yes — "reads L" predicts "vulnerable to L-script typographic text.")*
- **RQ3.** Do OCR-mask defenses have a per-language coverage gap (high detection recall +
  low residual ASR for Latin/English; the reverse for low-resource scripts)? *(H3: yes.)*
- **RQ4.** What is the per-language **safety debt** (attack surface minus defense coverage),
  and how large is the English-vs-low-resource gap?

---

## 5. Experiments (all feasible with existing code + off-the-shelf OCR)

1. **Mediation scatter (the core figure).** Across the 4 separate encoders × writing
   languages: plot OCR-reading-ability (x) vs typographic ASR (y). Establishes RQ1.
   Reuses `mechanism_experiment.py` (OCR probe) + `shared_vs_separate.py` / `perlang_models.py`.
2. **Reading ⇒ attack-surface (causal-comparative).** Compare encoders that differ in whether
   they read a given script (e.g. an encoder that reads Korean vs one that doesn't); show the
   Korean-typographic ASR follows reading ability. *Cheap cross-encoder version;* the strongest
   (heavier) version is a within-encoder fine-tune that adds Korean OCR ability and shows ASR
   rise.
3. **OCR-defense bias.** Run **two** detectors (EasyOCR + PaddleOCR) over the already-attacked
   images; per written language, measure **detection recall** (found the text?) and
   **post-mask residual ASR** (masking restored accuracy?). Establishes RQ3, tool-independently.
4. **Safety-debt table (the deliverable).** One row per language:
   `reading-ability | attack ASR (surface) | OCR detection recall (coverage) | residual ASR after defense`.
   The English-vs-low-resource gap is the quantified safety debt (RQ4).

---

## 6. Novelty positioning (honest)

The bare "reading → ASR" link is partly known — Goh et al. (*Multimodal Neurons*, Distill
2021) show CLIP reads rendered text; a 2026 preprint *"Reading Between the Pixels"* links
image–text embedding alignment to typographic ASR; Udandarao et al. (*"No Zero-Shot Without
Exponential Data"*) own the concept-frequency→performance story. **Do not** pitch "frequency
predicts vulnerability." The genuinely new contributions here are:

1. framing it as an explicit **fairness↔security tradeoff / design constraint** (closing the
   multilingual reading gap enlarges the attack surface);
2. the **defense-side** fairness result (OCR-mask defenses inherit language bias);
3. the per-language **safety-debt** quantification.

That trio was not present in the literature we surveyed.

---

## 7. Bridge to language-agnostic attacks

OCR-masking defends only the **typographic** (text-in-image) family. **Language-agnostic
pixel/embedding attacks have no text to detect**, so they bypass masking entirely. This
cleanly **delimits** where the reading/fairness story applies: typographic attacks are
language-structured (reading-dependent, hence unequal across languages); pixel attacks are
not. That boundary result is itself a contribution and connects to the team's planned
language-agnostic threat model.

---

## 8. Risks & things to design around

- **Confound:** reading ability tracks training frequency / resource tier, so "language"
  effects are really "training-diet" effects — acceptable, and it *strengthens* the
  "it's the training data, not the language" thesis; log per-string tokenizer subword-count
  and external word-frequency as covariates.
- **Tool-specificity:** OCR engines have their own biases distinct from CLIP's → test ≥2
  detectors so the defense-bias result isn't an artifact of one tool.
- **Complex-script rendering:** Arabic (RTL/joining), Hebrew, Thai (shaping) silently
  mis-render in plain PIL without `libraqm`/`arabic_reshaper`/`python-bidi`, producing
  garbage that *fakes* low ASR ("robustness"). The tofu strip catches empty boxes but not
  mis-shaping — verify with a real OCR pass or a native-speaker check.
- **Prior-work overlap:** position explicitly against *Reading Between the Pixels* (2026)
  and *Defense-Prefix* (2023, monolingual OCR-mask).
- **Small class vocabulary:** STL-10/CIFAR-10 (10 classes) can compress dynamic range and
  cause ASR floor/ceiling effects — consider a larger label set for the scaling claims.

---

## 9. Recommended first step

Run **Experiments 1 + 3 together** (mediation scatter + OCR-mask bias) on the four separate
encoders and the already-attacked images. One pass yields the paper's core figure ("reading
mediates vulnerability") and the defense-bias / safety-debt table — the two artifacts that
make the fairness↔security tradeoff concrete.

---

## 10. Reproducibility pointers

| need | reuse |
|---|---|
| OCR reading probe (text-on-blank) | `mechanism_experiment.py` (`render_textonly`, cosine-to-concept) |
| typographic attack + ASR matrix | `typographic_attack.py` |
| per-language / separate encoders | `perlang_models.py`, `shared_vs_separate.py` |
| new: OCR-mask defense | add EasyOCR + PaddleOCR detect→inpaint→re-classify over attacked images |
