# JHSS prose draft — writing steps 1–5

**Purpose:** Full body prose for JHSS order: Abstract → Introduction → Materials and Methods → Results → Discussion → Conclusion.  
**Style:** Formal scientific prose; third person; past tense for completed work. Citation numbers follow [`references.md`](references.md).  
**Numbers:** Locked to [`paper_tables_final.md`](paper_tables_final.md) / [`tables_index.md`](tables_index.md). MIXED2000 = `0.5 × attacked accuracy + 0.5 × clean-policy accuracy` over the same 1000 images. Table 1 baseline cells are n = 1000 finals.  
**Last synced:** 2026-08-05 — Table 1 baseline † cells promoted to n=1000 finals (DP KO/JA, hybrids ZH/KO/JA, grid 4-lang).

---

## Title page (fill before Word export)

**Title:** Cross-Lingual Attention Intersection as a Spatial Defense Against Typographic Attacks on Multilingual CLIP  
**Authors:** [Name]  
**Affiliation / school:** [School name and address]  
**Corresponding author:** [Name, email, phone]

---

## Abstract

Typographic attacks overlay an adversarial class name on an image and alter the prediction of CLIP-style classifiers even when the depicted object is unchanged. Prior defenses often depend on training against a particular attack family or on an external text recognizer, which can limit transfer or inherit recognizer failures. This study developed a spatial defense that leaves CLIP frozen and exploits text bias in separately trained multilingual vision encoders. Last-layer attention maps from English and a partner-language encoder were intersected; the agreed region was shaped into bounding boxes and filled with solid black only when a classifier on attention features predicted an attack. On a frozen dual-box CIFAR-10 protocol (n = 1000), undefended mean accuracy was 7.1%. The method recovered bilingual MIXED2000 scores of 81.7%, 78.4%, and 82.5% for the Chinese, Korean, and Japanese pairings, with English clean-image costs (Clean Δ EN) of −0.1, −0.6, and 0.0 percentage points. Ablations indicated that letterforms, rather than blank pads, accounted for both attack success and localization. Recovery was strong on typographic stickers, partial on hybrid text–animal overlays, and limited on animal-only stickers. Overall, the results support a defense that uses cross-lingual attention agreement for glyph localization, restores accuracy under dual-box attack at low clean cost, does not retrain CLIP, and does not require OCR; the only learned component is the image-level attack gate.

**Keywords:** Typographic Attack; CLIP; Vision–Language Models; Cross-Lingual Attention; Attention Intersection; Spatial Defense; Occlusion; Multilingual CLIP; Zero-Shot Classification; Attack Detection; CIFAR-10; Glyph Localization

---

## 1. Introduction

CLIP-style models perform zero-shot image classification by comparing an image embedding with text embeddings for candidate class names (1). Because these models are trained on large collections of captioned web images, they are also sensitive to written text appearing in the image. Overlaying an adversarial class name, such as printing “dog” on a photograph of a cat, is often sufficient to change the predicted label, even though the depicted object is unchanged (2, 3). Such overlays are termed typographic attacks. They require no gradients, no access to model weights, and no specialized hardware; a rendered text box or physical sticker is sufficient. Systems that classify photographs containing signs, packaging, or on-screen text are therefore exposed to this failure mode.

A natural hypothesis is that a second language could mitigate the attack: an English overlay might not affect a model trained primarily on Chinese captions, and disagreement between models could indicate an attack. That hypothesis is weaker when multiple languages share a single image encoder, because a corrupted encoder yields the same incorrect image embedding for every text tower. The present study instead used four independently pretrained models that do not share an encoder: English (EN), Chinese (ZH), Korean (KO), and Japanese (JA). Each model retains some English exposure from pretraining, but each also reflects priors from its own image–text corpus. The defense exploits that difference. An overlay that misleads two independently trained models must occupy regions attended by both, so the overlap of their attention maps provides a candidate location for the sticker. All pairings in this paper take the form EN ∩ L for L ∈ {ZH, KO, JA}.

English was included in every pair for an empirical reason. In prior attack-language × model experiments under the same protocol, English overlays produced high attack success across all four models, whereas native-script stickers transferred poorly to other languages, and Korean-only or Japanese-only stickers were often weak. The primary evaluation cases were therefore English-only stickers (Pure E) and English paired with the partner language (E + L). Partner-only stickers (Pure L) were retained mainly to characterize the limits of English attention on non-Latin glyphs.

Controlled ablations further indicated that the relevant stimulus is the letterform rather than a generic rectangular anomaly. Last-layer attention concentrates on the characters themselves. Blank white pads at the same coordinates produced little change in predictions and were localized poorly, whereas black letters without a pad approached the attack success and localization quality of the full sticker. Animal cutouts, which are visually salient but non-textual, could still reduce accuracy, yet the two models rarely agreed on their location. Cross-lingual attention intersection is therefore best interpreted as a typographic (glyph) localizer rather than a general anomaly detector.

Prior defenses against typographic attacks fall into several categories. Prediction-level detectors flag disagreement or anomalous confidence but do not restore the correct label. Blind spatial search, such as covering one-sixteenth patches in a 4×4 grid, is model-agnostic but coarse and computationally expensive. OCR-based repair detects text with an external recognizer and blurs it; performance is strong when recognition succeeds but inherits the recognizer’s failure modes. Defense-Prefix learns a soft text token while leaving CLIP frozen (4); on the present protocol it was strong on English, Chinese, and Japanese, with a weaker Korean score. Dyslexify (5) and SamplingTAR (6) suppress attention heads associated with typographic information. On the dual-box protocol used here, head-only interventions were ineffective, and hybrids that added blur recovered only part of the lost accuracy.

The remaining gap is spatial repair after detection. Localization is useful only if the identified region is removed and the image is reclassified. Unconditional masking is also undesirable, because applying the mask to clean images reduces accuracy, with the largest clean costs observed for the Korean and Japanese pairings. The defense therefore combines two stages: localization by cross-lingual attention agreement, and a gate that applies masking only when an attack is predicted.

Concretely, last-layer CLS→patch attention (7) was extracted from both models in a pair, the maps were intersected, surviving regions were converted to bounding boxes, and those boxes were filled with solid black. The full pipeline is referred to as gated `cc_bbox_black`. It was evaluated on three dual-box attack types for each partner language and compared with 4×4 grid occlusion, OCR+blur, Defense-Prefix (4), and Dyslexify / SamplingTAR hybrids (5, 6).

Without defense, the four models achieved a mean accuracy of 7.1% under dual-box attack. Gated `cc_bbox_black` recovered bilingual MIXED2000 scores of 81.7%, 78.4%, and 82.5% for the Chinese, Korean, and Japanese pairings, with Clean Δ EN of −0.1, −0.6, and 0.0 percentage points (Table 1). For the EN∩ZH pairing, English attacked accuracy was 72.9% and English MIXED2000 was 79.4%. Localization and reclassification require four forward passes and no gradients; CLIP is not retrained. The only learned component is the image-level gate. Limitations are equally specific: recovery is strong for printed text, partial for hybrid overlays, and weak for animal-only stickers, and the method requires a partner language model.

---

## 2. Materials and Methods

### 2.1 Dataset and protocol

Experiments used the CIFAR-10 test split (8), which comprises the classes airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. A balanced sample of 1000 images was drawn once (100 per class, seed 0, then shuffled) and reused for all comparisons. Sticker positions were stored in the same sample file. Each image contained fixed top-left anchors for two boxes — one for the English sticker and one for the partner-language sticker — so that every method attacked identical pixels. Images were upscaled to 224×224 before overlay and inference.

Percentile thresholds for attention masks were tuned on a 100-image subset (10 per class) and then floored: full runs used `max(tuned threshold, 0.95)`. Clean-image cost was measured by applying the same gated policy to the same 1000 images with no stickers. In Table 1, Clean Δ EN denotes the change in English clean accuracy under that defense.

Five quantities were reported: (i) top-1 accuracy after attack and after defense, per model and averaged over the languages scored in that row; (ii) attack success rate, defined as the frequency with which the model predicted the class written on the sticker; (iii) Clean Δ EN, the change in English clean accuracy when the defense was run on unattacked images; (iv) MIXED2000, defined as `0.5 × attacked accuracy + 0.5 × clean-policy accuracy` over the same indices, which penalizes defenses that improve attacked accuracy by degrading clean performance; and (v) cost, counted as forward, OCR, or intervention passes per image. MIXED2000 is always identified as such and is never reported as attacked accuracy. Mean attacked accuracy and mean MIXED2000 average only the languages present in each table row (four languages when filled; two languages for each Ours EN∩L pairing).

### 2.2 Models

Four separate checkpoints were used rather than language adapters on a shared backbone (1).

| Language | Checkpoint | Role |
|---|---|---|
| EN | OpenAI ViT-B/32 through OpenCLIP (1, 9) | anchor in every pair |
| ZH | Chinese-CLIP ViT-B/16 (10) | partner |
| KO | `Bingsu/clip-vit-base-patch32-ko` (11) | partner |
| JA | `llm-jp/llm-jp-clip-vit-base-patch16` (12) | partner |

Clean top-1 accuracy on the balanced sample was approximately 85.9% for English, 91.4% for Chinese, 89.6% for Korean, and 92.5% for Japanese. Because each model was trained on a different caption distribution, each contributes a distinct prior; the intersection defense depends on that difference. English was retained in every pair.

### 2.3 Attacks

Stickers were generated by rendering an adversarial class name onto the image (2, 3). No gradients were used. The primary threat model placed two non-overlapping white text boxes at the frozen coordinates, with font size 24 on the 224×224 canvas. Three language combinations were tested for each partner.

| Name | Box 0 | Box 1 |
|---|---|---|
| Pure E | English | English |
| E + L | English | partner language |
| Pure L | partner language | partner language |

Slot 0 always corresponded to the English position, because English was the strongest attack language across all four models. Three diagnostic variants reused the same coordinates: full sticker versus white-pad-only versus letters-only; all-text versus mixed text and animal cutouts versus animal-only; and font sizes 12, 24, and 40 crossed with one, two, or three boxes. Results for these variants are reported in Section 3.

### 2.4 Defense

For partner language L, the pipeline proceeded as follows.

1. A classifier on the attention features predicted clean or attacked (Section 2.5). Clean images were reclassified without masking.
2. For predicted attacks, last-layer CLS→patch attention was extracted from English and from L (Section 2.6).
3. Both maps were resized to 224×224 and combined by elementwise minimum, retaining only regions attended by both models.
4. A percentile threshold of at least 0.95 was applied.
5. A 3×3×3 dilation was applied by default; tighter settings were examined for Korean and Japanese.
6. The two largest connected components were retained and each was converted to an axis-aligned bounding box, matching the rectangular geometry of the stickers.
7. The boxes were filled with solid black.
8. Both models classified the edited image.

This stack is gated `cc_bbox_black`. Blur, mean-color, and neglect fills were also implemented and are compared in Section 3; black was the production setting.

### 2.5 Detector gate

Unconditional masking was not viable. Always-on masking reduced English clean accuracy by 12.1 and 13.6 percentage points on the Korean and Japanese pairings in the gate ablation (Table 4), so a mechanism was required to leave clean images unmodified. Twenty-six scalar features were computed from the English map, the partner map, and their intersection, including entropy, top-k mass, Gini coefficient, kurtosis, the area covering 95% of the mass, connected-component count, and the correlation and IoU between the two maps. A logistic classifier and a calibrated linear SVM (14) were trained on an image-level 70/15/15 split. The operating point was selected on the validation split to keep attack recall at or above 0.99, which limited the loss in attacked accuracy relative to always-on masking to less than one percentage point. All headline results in this paper use the gated policy.

### 2.6 Attention maps

Each image tower is a Vision Transformer (7): the image is partitioned into patches, a CLS token is prepended, and the final CLS vector serves as the image embedding. The saliency map used here is derived from that token. After one forward pass, the self-attention tensor of the final block was averaged over heads, and the CLS→patch row was reshaped to the patch grid — 7×7 for the ViT-B/32 English and Korean towers, and 14×14 for the ViT-B/16 Chinese and Japanese towers. Each map was min–max normalized and bilinearly resized to 224×224; the two maps were then combined by elementwise minimum. No backward pass is required, because attention weights are available after the forward pass. One implementation detail: the OpenCLIP English and Japanese towers hardcode `need_weights=False`, so attention was recomputed from the query and key projections inside a forward hook, whereas the Hugging Face Chinese and Korean towers return attentions directly. Localization and reclassification for one pair require four forward passes.

### 2.7 Baselines

All baselines used the same frozen stickers on the same 1000 images. Table 1 baseline cells are n = 1000 finals.

**4×4 grid occlusion.** The image was partitioned into 16 equal patches. One or two patches were covered greedily by selecting the cover that most reduced the model’s pre-defense top-1 confidence, after which the image was reclassified. Approximate cost was 62 passes. Search used English and Chinese confidence drop; Korean and Japanese were scored on the chosen occlusions. This baseline provides a model-agnostic search floor without attention or text detection.

**OCR + blur.** EasyOCR (13) localized text boxes, which were Gaussian-blurred at radius 12 before reclassification. Cost was 3. Scores were reported for all four languages at n = 1000. When recognition succeeds, this baseline approximates a spatial upper bound based on near ground-truth boxes.

**Defense-Prefix (4).** A prefix token was trained on CIFAR-10 training images with the same sticker style, with CLIP frozen and evaluation images held out. Separate tokens were trained for English, Chinese (20 epochs under an EN+ZH multi-sticker schedule), Korean, and Japanese. Cost was 2.

**Dyslexify (5) and SamplingTAR (6) hybrids.** Typographic heads or circuits were identified and suppressed, then combined with attention-guided patch blur. Cost was 3. English and partner-language hybrid cells in Table 1 are n = 1000 finals. Head-only variants were retained as negative results. Both implementations follow the style of the original papers and are not reruns of the authors’ exact checkpoints.

---

## 3. Results

Under the E + L attack, accuracy fell to 4.5% for English, 6.4% for Chinese, 11.6% for Korean, and 6.0% for Japanese, for a mean of 7.1%, below the 10% expected from random guessing among ten classes. Clean accuracy for the same models ranged from 85.9% to 92.5%. Subsequent results are reported relative to that undefended baseline.

### 3.1 Comparison with baselines (Table 1)

Scope differs across methods and must be read with the scores. Raw CLIP, OCR + blur, grid occlusion, Defense-Prefix, and the head hybrids report four-language finals. Ours rows are bilingual EN∩L pairings.

Grid occlusion recovered a mean attacked accuracy of 51.2% (47.8% English, 49.2% Chinese, 52.8% Korean, 55.1% Japanese) and 67.2% mean MIXED2000 at approximately 62 passes per image, with Clean Δ EN of −3.2 pp. OCR + blur reached 78.8% mean attacked accuracy and 84.0% mean MIXED2000 across four languages at cost 3, with Clean Δ EN of −0.6 pp. Defense-Prefix reached 73.8% on English, 81.4% on Chinese, 69.1% on Korean, and 84.8% on Japanese (MIXED2000 81.7%, 86.2%, 79.4%, and 89.2%); the four-language means were 77.3% attacked and 84.1% MIXED2000. The SamplingTAR and Dyslexify hybrids reached 61.7% and 60.3% four-language mean attacked accuracy (67.3% and 66.9% on English), with Clean Δ EN of −8.3 and −8.1 pp.

Gated `cc_bbox_black` reached mean attacked accuracy of 74.7%, 69.4%, and 75.9% for the Chinese, Korean, and Japanese pairings, with bilingual MIXED2000 of 81.7%, 78.4%, and 82.5%. Clean Δ EN was −0.1, −0.6, and 0.0 percentage points. Under EN∩ZH, English recovered from 4.5% to 72.9% under attack (English MIXED2000 79.4%), and attack success rate fell from 95.3% to 3.7%. The gate fired on 99.4–99.8% of attacked images and on 0.3–2.5% of clean images, with the highest clean false-positive rate on Korean.

Two comparisons are particularly relevant. Defense-Prefix retained a higher English MIXED2000 (81.7% versus 79.4%), so occlusion alone did not surpass the learned English prefix on English. OCR + blur retained a higher four-language mean MIXED2000 (84.0%) than any bilingual Ours row. On the English–Chinese pair alone, however, gated black reached bilingual MIXED2000 81.7%, close to the English–Chinese OCR average of 80.9%, without an external recognizer and without retraining CLIP.

### 3.2 Attack-component and geometry ablations (Table 2)

Letterforms, rather than blank pads, accounted for attack success. Full stickers reached 95.3% attack success on English and 93.6% on Chinese, with tight cross-model attention overlap (IoU 0.691; nonzero overlap on 100% of images). Letters without a pad remained strong (89.8% and 84.5%; IoU 0.669). White pads alone reached 2.3% and 2.1% and were localized poorly (IoU 0.083; overlap on 41.1% of images). In Table 2, gated black recovered 75.8% English and 82.7% Chinese accuracy in the letters-only condition, compared with 72.9% and 76.5% for full stickers, consistent with letters carrying both the attack and the localization signal. White-pad-only images remained largely correct without defense (75.5% English) and did not benefit from occlusion.

Font size and sticker count controlled difficulty. At font 12, gated English and Chinese accuracies were 76.0% and 83.9%. At the production size of 24, they were 72.9% and 76.5%. At font 40, they fell to 58.3% and 56.3%, as larger glyphs exceeded the coverage of two bounding boxes. Sticker count showed the same pattern: one box yielded 78.2% and 82.9%, two boxes 72.9% and 76.5%, and three boxes 45.9% and 56.1%, even after the number of retained components was increased to match. Font 24 with two boxes was retained as the reported threat model.

### 3.3 Text, hybrid, and non-text stickers (Table 3)

The same pipeline was evaluated on three overlay types with positions and settings held fixed.

| Overlay | No defense (EN) | EN∩ZH black | EN-only black | EN ASR after | Clean Δ EN |
|---|---:|---:|---:|---:|---:|
| all text | 4.5% | 72.9% | 68.0% | 3.7% | −0.1 pp |
| text + animal | 1.3% | 43.5% | 40.2% | 27.3% | −0.1 pp |
| animal only | 11.0% | 20.6% | 32.9% | 54.2% | −1.8 pp |

Oracle masks constructed from true sticker coordinates reached 74.6%, 64.3%, and 56.9% on the same three conditions. On printed text, the defense was within approximately two points of that ceiling. On animal cutouts, recovery was minimal, and removing the partner model increased English accuracy from 20.6% to 32.9%, because the two models rarely agreed on non-text regions. Because the oracle mask itself reached only 56.9% in that condition, the shortfall is only partly attributable to localization error. The gate also performed worse for animal overlays, firing on approximately 26% of clean images compared with roughly 0–1% under printed text. These results support describing the method as a defense against printed text rather than against arbitrary visual stickers.

### 3.4 Design choices (Table 4)

Three settings were varied while the remainder of the pipeline was held fixed.

Adding the partner model improved recovery on text. English-only masking recovered 68.0%, whereas English–Chinese intersection recovered 72.9% on English, for a bilingual MIXED2000 of 81.7%. Extending the intersection to all four models left English at 72.0% and raised the four-language mean MIXED2000 to 83.4%, but increased Clean Δ EN to −0.6 pp; the two-model configuration was therefore retained as the default.

The gate was required for usable clean performance on Korean and Japanese. In the gate ablation, always-on masking cost 2.2, 12.1, and 13.6 percentage points of English clean accuracy for the Chinese, Korean, and Japanese pairings. Gating reduced those Clean Δ EN values to 0.0, −0.3, and 0.0 while raising English MIXED2000 from 76.8% to 77.9% (ZH), 69.6% to 75.3% (KO), and 70.5% to 77.2% (JA). Headline Clean Δ EN for production gated black in Table 1 remained −0.1, −0.6, and 0.0 for the three pairings.

Fill type mattered less than the gate but was not neutral. On English, MIXED2000 ranked black (79.4%), mean color (78.2%), blur (77.9%), and no fill (73.2%). Black improved attacked accuracy by 3.0 percentage points relative to blur (72.9% versus 69.9%). Black was used for all languages to keep tables comparable.

The localization recipe transferred across partners. Partner-only stickers remained an exception: they were already weak on Korean and Japanese before defense, and masking sometimes reduced accuracy further.

### 3.5 Grid search and qualitative cases

Scoring grid patches by confidence drop rather than raw confidence raised mean attacked accuracy from approximately 11% to the Table 1 grid floor (51.2%), indicating that the scoring rule, rather than search alone, accounted for most of the gain. Exhaustive search over all 120 patch pairs did not close the remaining gap, because a sticker straddling a patch boundary cannot be covered cleanly by a 1/16 tile. Patch-hit analysis showed that covering the English box contributed more than covering the partner box.

In qualitative examples, successful cases were consistent across partners: both attention maps concentrated on the lettering, the bounding boxes aligned with the two stickers, and probability mass returned to the true class after occlusion. Failures were typically partial coverage, in which a box clipped a sticker edge and left sufficient glyph visible for the adversarial class to remain competitive.

### 3.6 Recommended setting

For this protocol, the reported configuration is Attn-last EN ∩ L, `cc_bbox` shaping, black fill, and the detector gate, with threshold floored at 0.95 and threat model fixed at font 24 with two boxes. That configuration yielded bilingual MIXED2000 of 81.7%, 78.4%, and 82.5% for the Chinese, Korean, and Japanese pairings, with Clean Δ EN of −0.1, −0.6, and 0.0 percentage points.

---

## 4. Discussion

The detector gate was essential for practical deployment. Clean and attacked attention maps were nearly linearly separable on the twenty-six heatmap features (test AUC ≈ 1.0 for Chinese and Japanese; 0.999 for Korean), so a logistic classifier or calibrated linear SVM was sufficient (14). Attacked maps concentrated mass on compact glyph-shaped regions, whereas clean maps were more diffuse. In the Table 4 gate ablation, always-on masking reduced English clean accuracy by 12.1 and 13.6 percentage points on Korean and Japanese; gating restored Clean Δ EN to 0.0, −0.3, and 0.0 for ZH/KO/JA while improving English MIXED2000 on each pairing. The gate is therefore a core pipeline stage rather than an optional post-processing step. CLIP parameters remain frozen: localization uses only forward passes — four for one English–partner pair — and the only learned component is the image-level gate.

The effectiveness of localization follows from the training structure of the models. Independently trained English and partner towers share some English exposure but retain distinct caption priors. An overlay that misleads both must occupy patches where those priors still co-attend, which in practice are letterforms. Intersection converts that agreement into a spatial cue stronger than either map alone: English-only masking recovered 68.0%, whereas adding Chinese raised English accuracy to 72.9% (bilingual MIXED2000 81.7%). On English, fill types ranked black > mean color > blur > no fill. Soft blur remains a useful ablation but was not the production setting.

Comparisons with other defenses should be interpreted with scope. Grid occlusion provides a weak floor (51.2% mean attacked accuracy, 67.2% mean MIXED2000 at ~62 passes). OCR + blur is the strongest four-language mean in Table 1 (78.8% attacked, 84.0% MIXED2000) when recognition succeeds, and gated black approaches the English–Chinese OCR average without OCR. Defense-Prefix retains a higher English MIXED2000 (81.7% versus 79.4%), so occlusion alone does not surpass a learned English prefix on English; Chinese and Japanese Defense-Prefix are also strong (81.4% / 86.2% and 84.8% / 89.2%), while Korean is weaker (69.1% / 79.4%). Head-only Dyslexify and SamplingTAR interventions were ineffective on dual-box attacks; hybrids that added blur recovered partially (four-language means 61.7% and 60.3%; English 67.3% and 66.9%) but incurred large English clean costs (−8.3 and −8.1 pp) and did not match gated black. Spatial occlusion, rather than head ablation, accounted for most of the recovery.

The animal-sticker results delimit the claim. Printed text recovered to 72.9%, within approximately two points of an oracle mask from true boxes. Hybrid overlays reached 43.5%, and animal cutouts reached only 20.6%; removing the partner model increased animal recovery to 32.9% because bilingual agreement was weak for non-text patches. Even the oracle mask reached only 56.9% on animals, so the shortfall is only partly a localization error. Cross-lingual attention intersection should therefore be described as a typographic localizer, not as a general visual-anomaly defense. Extension to arbitrary stickers remains an open question.

Additional limitations follow. The method requires a partner language model; a single monolingual tower does not instantiate EN ∩ L. Evaluation used CIFAR-10 with frozen sticker positions rather than ImageNet-scale scenes or adaptive attackers that place text to evade attention or the gate. Gated headline numbers correspond to E + L; Pure E and Pure L under the same gate remain open. Partner-only stickers remain more difficult for the English half of the intersection, particularly Chinese-only text. Three simultaneous stickers reduced gated English accuracy to 45.9% even after increasing the number of retained components, and font 40 reduced gated English accuracy by 14.6 percentage points relative to font 24 (58.3% versus 72.9%). A residual gap to clean performance under attack also remains: English recovered to 72.9% against a clean baseline near 85.9%, and oracle black reached only 74.6%, indicating that black fill and gating remove most clean cost without closing that residual. Dyslexify and SamplingTAR results are style ports rather than exact author checkpoints. These constraints bound the present claim; they do not reverse the main finding on printed dual-box typographic attacks.

---

## 5. Conclusion

Dual-box typographic overlays reduced four language-specific CLIP models to a mean accuracy of 7.1%. Intersecting last-layer English and partner attention, converting the agreed region to bounding boxes, and filling those boxes with solid black only when a heatmap-shape detector predicted an attack recovered bilingual MIXED2000 scores of 81.7%, 78.4%, and 82.5% for Chinese, Korean, and Japanese, with Clean Δ EN of −0.1, −0.6, and 0.0 percentage points. The method leaves CLIP frozen, requires four forward passes per pairing, transfers across the three partners, and depends on letterform co-attention rather than blank pads. Recovery is strong for printed text, partial for hybrid overlays, and weak for animal stickers. Beyond prediction-level disagreement, a practical multilingual defense can localize regions of cross-lingual attention agreement, occlude those regions, and reclassify.

---

## Notes before Word export

- JHSS order: Abstract → Introduction → Materials and Methods → Results → Discussion → Conclusion → References.
- Body prose for those sections is in this file. Title-page fields still need real author/school info.
- Rebuild the tables with Word's table tool from [`paper_tables_final.md`](paper_tables_final.md); pull figures into separate files; re-check that citation numbers run in order of first appearance.
- Paste the numbered list from [`references.md`](references.md) into the Word References section after renumbering to first-appearance order.
- Keep claims honest: occlusion alone does not beat Defense-Prefix on English MIXED2000; OCR wins the four-language mean MIXED2000; animal-sticker recovery is a limitation.
- Optional: Acknowledgements (mentor, compute, funding) before References.
