# Ablation study — method + attack components

**Protocol (production):** gated Attn-last EN∩L `cc_bbox_black`, thr ≥ 0.95, dilate 3, top-2, frozen dual-box CIFAR-10 n=1000 (`CIFAR10_BALANCED_1000_SAMPLE` + `attack_pos`), attack = `multi`, CUDA.  
**Main results:** [`4_lang_table.md`](4_lang_table.md). **Narrative:** [`research_diary.md`](research_diary.md).

This file is the paper appendix source for component ablations. **Do not re-run** fill / gate / text-component / animal occlusion — numbers below are frozen. Font-size and box-count sweeps live under [`lib/notebooks/attack_geometry_ablation/`](../lib/notebooks/attack_geometry_ablation/).

---

## A. Method ablations

### A.1 Fill type (neglect / blur / mean / black)

**Question:** After `cc_bbox` shaping, which fill recovers attacked accuracy with least clean damage under the gate?

**Setup:** Gated Attn-last EN∩L `cc_bbox`; fills = neglect (ViT patch-token zero), blur, mean, black. EN: [`en_neglect_vs_blur/`](../lib/notebooks/en_neglect_vs_blur/). Partners: [`partner_fill_ablation/`](../lib/notebooks/partner_fill_ablation/).

**EN gated (n=1000)**

| Fill | EN atk | EN clean | EN MIXED2000 |
|------|-------:|---------:|-------------:|
| neglect | 60.5% | 85.9% | 73.20% |
| blur | 69.9% | 85.9% | 77.90% |
| mean | 70.4% | 85.9% | 78.15% |
| **black** | **72.9%** | **85.8%** | **79.35%** |

Ranking: **black > mean > blur > neglect**. Oracle GT+black EN atk ceiling ≈ **74.6%**.

**Partner bilingual MIXED2000 (gated)**

| Partner | neglect | blur | mean | **black** | Winner |
|---------|--------:|-----:|-----:|----------:|--------|
| ZH | 77.38% | 81.28% | 81.30% | **81.65%** | black |
| KO | 74.18% | **78.50%** | 78.23% | 78.35% | blur (+0.15pp) |
| JA | 80.45% | 82.45% | 82.20% | **82.53%** | black |

**Verdict:** Production freezes **black for all langs**. KO blur +0.15pp is ablation-only — not a protocol fork.

---

### A.2 Gate on vs always-on

**Question:** Does the Attn-last detector gate recover Clean Δ without losing attacked accuracy?

**Setup:** Phase-C detector; always-on vs gated occlusion on `multi` (partners ZH/KO/JA). Always-on numbers below use blur fill (Phase-C logs); production gated fill is black. Clean Δ pattern is what motivates the gate.

| Partner | Always Clean Δ | Gated Clean Δ | Always MIXED | Gated MIXED | Gated−always |
|---------|---------------:|--------------:|-------------:|------------:|-------------:|
| ZH | −1.45 pp | **0.00 pp** | 80.60% | 81.28% | +0.68 |
| KO | −11.25 pp | **−0.20 pp** | 73.20% | 78.50% | +5.30 |
| JA | −11.50 pp | **0.00 pp** | 76.80% | 82.45% | +5.65 |

Gate fire (gated black, EN∩ZH/KO/JA `multi`): atk ≈ **99.4–99.8%**, clean ≈ **0.3–2.5%**.

**Verdict:** Always-on destroys KO/JA clean accuracy; the gate is a **core** pipeline stage, not an optional bolt-on.

Sources: [`attack_detector/`](../lib/notebooks/attack_detector/), [`paper_draft.md`](paper_draft.md) §3.2 / §3.8.

---

### A.3 Single-language occlusion (EN + EN-only vs EN∩L)

**Question:** How does monolingual EN occlusion compare to bilingual EN∩L under production black fill?

**EN gated black (typographic `multi`, production):** atk **72.9%** / clean **85.8%** / MIXED2000 **79.35%** ([`en_neglect_vs_blur/results/gated_n1000.json`](../lib/notebooks/en_neglect_vs_blur/results/gated_n1000.json)).

**Animal / mixed / text recovery (n=1000, black)** — EN∩ZH vs EN-only ([`animal_sticker_ablation/results/occlusion_n1000.json`](../lib/notebooks/animal_sticker_ablation/results/occlusion_n1000.json)):

| Mode | Arm | EN acc | EN ASR | Clean Δ EN (gated) |
|------|-----|-------:|-------:|-------------------:|
| all_text | gated EN∩ZH | **72.9%** | 3.7% | −0.1 pp |
| all_text | gated EN-only | 68.0% | 3.6% | −0.1 pp |
| all_sticker | gated EN∩ZH | 20.6% | 54.2% | −1.8 pp |
| all_sticker | gated EN-only | **32.9%** | **28.5%** | −6.1 pp |
| mixed | gated EN∩ZH | **43.5%** | 27.3% | −0.1 pp |
| mixed | gated EN-only | 40.2% | 27.4% | −0.1 pp |

**Verdict:** Bilingual EN∩ZH is best for **typographic** stickers. EN-only partially transfers to **animal** stickers but costs clean accuracy and still trails oracle GT black (animal EN **56.9%**). Localization—not fill—is the animal bottleneck.

---

### A.4 Four-language results (method transfer)

**Question:** Does gated `cc_bbox_black` transfer across ZH / KO / JA partners? Does a true 4-way EN∩ZH∩KO∩JA mask work?

Full pairwise tables: [`4_lang_table.md`](4_lang_table.md). Quote set:

| Claim | Number |
|-------|-------:|
| Ours avg bilingual MIXED (ZH/KO/JA black) | **80.84%** |
| Partner bilingual MIXED black (ZH / KO / JA) | **81.65 / 78.35 / 82.53%** |
| Ours EN MIXED (gated black) | **79.35%** |
| Ours EN atk / clean (gated black) | **72.9% / 85.8%** |
| Gate fire atk / clean (EN∩ZH) | **99.8% / 0.4%** |
| **Four-way gated mean MIXED / mean atk** | **83.40% / 77.4%** |

**Four-way occlusion** (EN ∩ ZH ∩ KO ∩ JA Attn-last → gated black; EN+ZH `multi` attack; score all four): [`four_way_occlusion/`](../lib/notebooks/four_way_occlusion/). Gated EN/ZH/KO/JA atk **72.0 / 76.4 / 76.5 / 84.5%**; Clean Δ ≈ **−0.3 to −0.6 pp**; mean MIXED **83.40%**. Always-on Clean Δ ≈ **−20 pp** — gate required. Always-on 4-lang blur transfer (design history): [`four_lang_cc_bbox_blur/`](../lib/notebooks/four_lang_cc_bbox_blur/).

**Verdict:** Production claim is **cross-language spatial defense** under a frozen gate+black stack—not EN-only peer matching. Pairwise EN∩L is the default cost-4 recipe; 4-way is the stronger all-language occlusion variant.

---

## B. Attack ablations

### B.1–B.2 Sticker content: white pads, glyphs, animals, hybrid

**Question:** Is the hijack from glyphs or from the white pad? Does production gated black repair animal stickers, typographic text, and mixed hybrid?

**Setup:** Same GT boxes where applicable. Modes include `white_only` (blank white pad), `all_text` / `full` (glyphs), `mixed`, `all_sticker` (animal). Attack+localization from [`attack_component_ablation/results/summary_n1000.json`](../lib/notebooks/attack_component_ablation/results/summary_n1000.json) and [`animal_sticker_ablation/results/summary_n1000.json`](../lib/notebooks/animal_sticker_ablation/results/summary_n1000.json).

**Localization**

| Mode | EN ASR | ZH ASR | EN∩ZH IoU | EN∩ZH det@0.1 |
|------|-------:|-------:|----------:|--------------:|
| white_only | 2.3% | 2.1% | 0.083 | 41.1% |
| all_sticker | **76.6%** | **87.7%** | 0.117 | 51.1% |
| mixed | **98.7%** | **97.6%** | 0.310 | **99.8%** |
| all_text | **95.3%** | **93.6%** | **0.691** | **100%** |

Undefended EN acc: white_only **75.5%** vs text_only **9.6%** vs full **4.5%**. Letters drive ASR and EN∩ZH localization; blank white pads do not. Threat model = **readable text**.

**Gated black recovery** (see A.3 table): `all_text` → EN **72.9%**; `mixed` → **43.5%**; `all_sticker` → **20.6%** (EN-only **32.9%**).

**Verdict:** Method is strong on typographic dual-box; hybrid is partial; animal-only is a limitation (intersection is text-favoring). Glyphs—not white pads—cause the hijack.

---

### B.3 Font size

**Question:** How sensitive is undefended ASR and gated `cc_bbox_black` recovery to sticker font size?

**Setup:** Frozen dual-box `attack_pos`, EN∩ZH, `multi`, font ∈ {12, 24, 40}. Protocol default = **24**. Arms: never / always / gated black. Code: [`attack_geometry_ablation/`](../lib/notebooks/attack_geometry_ablation/). Results: [`font_n1000.json`](../lib/notebooks/attack_geometry_ablation/results/font_n1000.json).

**Gated black (n=1000)**

| Font | never EN (acc/ASR) | gated EN | gated ZH | Clean Δ EN | fire atk / clean |
|------|-------------------:|---------:|---------:|-----------:|-----------------:|
| 12 | 9.8% / 89.9% | **76.0%** | **83.9%** | +0.0 pp | 98.6% / 0.3% |
| **24** | 4.5% / 95.3% | **72.9%** | **76.5%** | −0.1 pp | 99.8% / 0.4% |
| 40 | 3.0% / 96.9% | 58.3% | 56.3% | −0.4 pp | 99.9% / 2.7% |

Always-on EN: 76.7% / **72.9%** / 58.4% (12 / 24 / 40). Font-24 gated matches production EN **72.9%**.

**Verdict:** Larger glyphs are a harder spatial threat — font 40 drops gated EN by **−14.6 pp** vs protocol 24. Smaller font 12 is easier (gated EN **76.0%**) but weaker as an attack (ASR 89.9% vs 95.3%). **Main results stay at font 24.**

---

### B.4 Number of boxes

**Question:** How does attack strength and gated recovery change with 1 / 2 / 3 text boxes?

**Setup:** `FONT_SIZE=24`, EN∩ZH black. **1** = frozen EN anchor only; **2** = frozen dual EN+ZH (protocol); **3** = dual + seeded third EN box (not in frozen JSON). **Mask capacity matches the threat:** `top_k = num_boxes` (1→1, 2→2, 3→3). Results: [`boxes_n1000.json`](../lib/notebooks/attack_geometry_ablation/results/boxes_n1000.json).

**Gated black (n=1000)**

| Boxes | top_k | never EN (acc/ASR) | gated EN | gated ZH | Clean Δ EN | fire atk / clean |
|------:|------:|-------------------:|---------:|---------:|-----------:|-----------------:|
| 1 | 1 | 5.7% / 94.1% | **78.2%** | **82.9%** | −0.1 pp | 100% / 0.6% |
| **2** | 2 | 4.5% / 95.3% | **72.9%** | **76.5%** | −0.1 pp | 99.8% / 0.4% |
| 3 | 3 | 2.4% / 97.5% | **45.9%** | **56.1%** | −0.8 pp | 99.5% / 4.3% |

Always-on EN: 78.2% / 72.9% / 46.0% (1 / 2 / 3). Font-24 dual-box gated matches production EN **72.9%**.

**Verdict:** Dual-box (protocol) is the intended threat — strong ASR with recoverable gated black. One EN box is slightly easier to defend. **Three boxes with matched `top_k=3` recover gated EN to 45.9%** (vs **8.5%** when production `top_k=2` discarded the third CC). Still harder than dual-box; main results stay at **NUM_BOXES=2** / production `top_k=2`.

---

## C. Status board

| Ablation | Kind | Status |
|----------|------|--------|
| Fill type | method | Done |
| Gate on/off | method | Done |
| Single-lang occlusion | method | Done |
| Four-lang results | method | Done (pairwise + 4-way) |
| Text vs white / sticker / hybrid | attack | Done |
| Font size | attack | **Done** (`attack_geometry_ablation`) |
| Number of boxes | attack | **Done** (`attack_geometry_ablation`) |
