# Paper tables (final)

Manuscript Tables 1–4 (full columns, not slim candidates).  
**Protocol:** frozen dual-box CIFAR-10, n = 1000, attack = `multi`, thr ≥ 0.95, CUDA.  
**Ours:** gated Attn-last EN∩L black fill.  
Working numbers: `[tables_index.md](tables_index.md)`.

JHSS layout: **Table #.** heading above; short caption below.

**Reading the tables.** Language codes: EN / ZH / KO / JA = English / Chinese / Korean / Japanese. **acc** = top-1 accuracy (%). Unless noted, accuracy columns are measured **under attack**. Language cells in Table 1 are **acc / mixed acc**. **mixed acc** (MIXED2000) = 0.5 × attacked acc + 0.5 × clean-policy acc on the same 1000 images. **Clean Δ EN** = Δ (delta / change) in English clean (unattacked) accuracy under the defense, in percentage points (pp). **ASR** = attack success rate. **Cost** = passes per image. Em dash (—) = not evaluated or intentionally skipped (see schedule below).

**Scoped means:** Mean acc (attacked) / mean mixed acc average only languages scored in that row (4-lang when filled; 2-lang for each Ours EN∩L row).

---

**Table 1.** Baseline and proposed defenses under dual-box typographic attack.


| Method                | EN acc / mixed acc | ZH acc / mixed acc | KO acc / mixed acc | JA acc / mixed acc | Mean acc (attacked) | Mean mixed acc | Clean Δ EN  | Cost  |
| --------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------- | -------------- | ----------- | ----- |
| No defense (Raw CLIP) | 4.5 / 45.2         | 6.4 / 48.9         | 11.6 / 50.6        | 6.0 / 49.3         | **7.1%**            | **48.5%**      | 0.0 pp      | 1     |
| 4×4 grid occlusion    | 47.8 / 65.3        | 49.2 / 66.8        | 52.8 / 65.8        | 55.1 / 70.8        | 51.2%               | 67.2%          | −3.2 pp     | 62    |
| OCR + blur            | 72.8 / 79.1        | 74.7 / 82.7        | 80.2 / 84.5        | 87.6 / 89.7        | **78.8%**           | **84.0%**      | −0.6 pp     | 3     |
| Defense-Prefix        | 73.8 / 81.7        | 81.4 / 86.2        | 69.1 / 79.4        | 84.8 / 89.2        | 77.3%               | 84.1%          | +0.5 pp     | 2     |
| SamplingTAR + blur    | 67.3 / 72.5        | 68.9 / 79.3        | 59.0 / 62.9        | 51.5 / 72.1        | 61.7%               | 71.7%          | −8.3 pp     | 3     |
| Dyslexify + blur      | 66.9 / 72.4        | 68.2 / 79.6        | 62.9 / 67.1        | 43.2 / 67.0        | 60.3%               | 71.5%          | −8.1 pp     | 3     |
| **Ours (EN∩ZH)**      | **72.9 / 79.4**    | **76.5 / 84.0**    | —                  | —                  | **74.7%**           | **81.7%**      | **−0.1 pp** | **4** |
| **Ours (EN∩KO)**      | **65.6 / 75.5**    | —                  | **73.1 / 81.3**    | —                  | **69.4%**           | **78.4%**      | **−0.6 pp** | **4** |
| **Ours (EN∩JA)**      | **68.9 / 77.4**    | —                  | —                  | **82.8 / 87.7**    | **75.9%**           | **82.5%**      | **0.0 pp**  | **4** |


Caption: Language cells are attacked acc / mixed acc (%). Δ = change (pp). Mean acc (attacked) and Mean mixed acc average only languages present in the row (4-lang for Raw CLIP, OCR + blur, grid, Defense-Prefix, and head hybrids; 2-lang for each Ours EN∩L pairing). Clean Δ EN is the change in English clean accuracy under that method. Cost is passes per image. Bold marks production Ours rows and the Raw CLIP floor. All baseline cells are n=1000 finals.

---

**Table 2.** Attack-component and geometry ablations with gated EN∩ZH black occlusion.


| Ablation          | Setting          | EN acc (no defense) | EN acc (gated) | ZH acc (gated) |
| ----------------- | ---------------- | ------------------- | -------------- | -------------- |
| Font size         | 12               | 9.8%                | 76.0%          | 83.9%          |
| Font size         | **24**           | 4.5%                | **72.9%**      | **76.5%**      |
| Font size         | 40               | 3.0%                | 58.3%          | 56.3%          |
| Sticker type | white pad only   | 75.5%               | 70.9%          | 75.0%          |
| Sticker type | letters only     | 9.6%                | 75.8%          | 82.7%          |
| Sticker type | **full sticker** | 4.5%                | **72.9%**      | **76.5%**      |
| Number of boxes   | 1                | 5.7%                | 78.2%          | 82.9%          |
| Number of boxes   | **2**            | 4.5%                | **72.9%**      | **76.5%**      |
| Number of boxes   | 3                | 2.4%                | 45.9%          | 56.1%          |


Caption: Top-1 accuracy (%) under attack for gated EN∩ZH black fill (n = 1000). Bold marks production settings (font 24, full sticker, two boxes) and the matching gated accuracies.

---

**Table 3.** Defense recovery on text, hybrid, and animal-sticker overlays.


| Overlay                | EN acc (no defense) | EN∩ZH acc | EN-only acc | ASR (EN∩ZH) | Clean Δ EN (EN∩ZH) |
| ---------------------- | ------------------- | --------- | ----------- | ----------- | ------------------ |
| All text (typographic) | 4.5%                | **72.9%** | 68.0%       | 3.7%        | −0.1 pp            |
| Text + animal (hybrid) | 1.3%                | **43.5%** | 40.2%       | 27.3%       | −0.1 pp            |
| Animal only            | 11.0%               | **20.6%** | 32.9%       | 54.2%       | −1.8 pp            |


Caption: English top-1 accuracy (%) and attack success rate (ASR) for gated black fill (n = 1000). No defense is the never-defend arm. Bold marks the production EN∩ZH pairing. Clean Δ EN is the Δ (change) in clean English accuracy relative to no-defense CLIP (pp).

---

**Table 4.** Occlusion-algorithm ablations: languages used, gate, and fill type.


| Ablation             | Setting                  | EN acc    | Clean Δ EN              | Mixed acc                            |
| -------------------- | ------------------------ | --------- | ----------------------- | ------------------------------------ |
| Language used | English only             | 68.0%     | −0.1 pp                 | —                                    |
| Language used | **EN∩ZH**                | **72.9%** | **−0.1 pp**             | **81.7%** bilingual                  |
| Language used | EN∩ZH∩KO∩JA              | 72.0%     | −0.6 pp                 | 83.4% mean (4-lang)                  |
| Gate                 | always on (ZH / KO / JA) | —         | −2.2 / −12.1 / −13.6 pp | EN mixed acc 76.8 / 69.6 / 70.5%     |
| Gate                 | **gated** (ZH / KO / JA) | —         | **0.0 / −0.3 / 0.0 pp** | EN mixed acc **77.9 / 75.3 / 77.2%** |
| Fill type            | no fill                  | 60.5%     | ≈0.0 pp                 | 73.2% EN                             |
| Fill type            | blur                     | 69.9%     | ≈0.0 pp                 | 77.9% EN                             |
| Fill type            | mean color               | 70.4%     | ≈0.0 pp                 | 78.2% EN                             |
| Fill type            | **black**                | **72.9%** | ≈0.0 pp                 | **79.4%** EN                         |


Caption: Ablations of the occlusion recipe (n = 1000). Bold marks production choices: EN∩ZH, gated, and black fill. EN acc = English accuracy under attack. Mixed acc = MIXED2000 for the languages named in Setting. Clean Δ EN = Δ (change) in English clean accuracy (pp). Gate rows report English Clean Δ EN and English mixed acc from Phase-C blur logs (always on vs gated). Fill and language-count rows use production black.

---



## Fill-in schedule (Table 1 gaps)


| Priority | Run                                  | Goal                                     | Status                                                                             |
| -------- | ------------------------------------ | ---------------------------------------- | ---------------------------------------------------------------------------------- |
| 1        | Defense-Prefix ZH retrain/tune       | ZH attacked acc ≥ 60%                    | **done** — 81.4% (EN+ZH multi train, 20 ep)                                        |
| 2        | OCR + blur KO/JA                     | Fill KO/JA acc / mixed acc               | **done** — KO 80.2/84.5, JA 87.6/89.7; 4-lang mean acc 78.8%, mean mixed acc 84.0% |
| 3        | Defense-Prefix KO/JA (after ZH ≥60%) | Fill or document skip                    | **done** — KO 69.1/79.4, JA 84.8/89.2 (n=1000); 4-lang mean atk 77.3%, mean mixed 84.1% |
| 4        | Grid Clean Δ EN (+ KO/JA if cheap)   | Fill grid mixed acc / Clean Δ EN         | **done** — EN/ZH/KO/JA 47.8/49.2/52.8/55.1; mean atk 51.2%, mean mixed 67.2%; CleanΔEN −3.2pp |
| 5        | SamplingTAR / Dyslexify KO/JA/ZH     | Only if head hooks transfer; else keep — | **done** — TAR ZH/KO/JA 68.9/59.0/51.5; Dys ZH/KO/JA 68.2/62.9/43.2 (n=1000 hybrids) |


After each final n=1000 result: recompute `[mixed_2000_summary.json](../lib/notebooks/paper_baselines/results/mixed_2000_summary.json)` and back-fill this file + `[tables_index.md](tables_index.md)`.