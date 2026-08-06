# References (JHSS starter set)

**Venue format:** Numbered in order of first appearance in the manuscript. Cite in text as `(1)`, `(2-5)`. Punctuation after the citation. Prefer APA-ish entries + **live DOI** (or arXiv / HF URL if no DOI).  
**Status:** Working bibliography for `paper_draft.md`. Re-number once Intro/Methods prose is locked so order matches first-use in the Word file.

**How to use while writing:** When you first mention a concept, drop `(n)` and keep this list in sync. Do not use EndNote “add-ins” in the JHSS `.docx`.

---

## Suggested citation map (by draft topic)

| Topic in draft | Cite |
|---|---|
| CLIP / zero-shot VL | (1) |
| Typographic / written-vs-visual vulnerability | (2), (3) |
| Defense-Prefix baseline | (4) |
| Dyslexify baseline | (5) |
| SamplingTAR baseline | (6) |
| Vision Transformer / CLS attention | (7) |
| CIFAR-10 dataset | (8) |
| OpenCLIP (EN load path) | (9) |
| Chinese-CLIP (ZH) | (10) |
| Korean CLIP checkpoint | (11) |
| Japanese CLIP (llm-jp) | (12) |
| EasyOCR (OCR+blur upper bound) | (13) |
| SVM / logistic detector (optional methods cite) | (14) |

Optional add-ons later: occlusion/saliency reviews, multilingual CLIP surveys, GradCAM (only if mentioned historically).

---

## Numbered list (paste into JHSS References section)

**(1)** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, 8748–8763. https://doi.org/10.48550/arXiv.2103.00020  
*Note:* Prefer the PMLR proceedings page for camera-ready; arXiv DOI above is acceptable if JHSS accepts it. Official: https://proceedings.mlr.press/v139/radford21a.html

**(2)** Materzyńska, J., Torralba, A., & Bau, D. (2022). Disentangling visual and written concepts in CLIP. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16410–16419. https://doi.org/10.1109/CVPR52688.2022.01592

**(3)** Goh, G., Cammarata, N., Voss, C., Carter, S., Petrov, M., Schubert, L., et al. (2021). Multimodal neurons in artificial neural networks. *Distill*. https://doi.org/10.23915/distill.00030  
*Typographic sticker examples popularized here; cite with Materzyńska for attack framing.*

**(4)** Azuma, H., & Matsui, Y. (2023). Defense-Prefix for preventing typographic attacks on CLIP. *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*, 3644–3653. https://doi.org/10.1109/ICCVW60793.2023.00392  
*Also:* https://doi.org/10.48550/arXiv.2304.04512

**(5)** Hufe, L., Venhoff, C., Purelku, E., Dreyer, M., Lapuschkin, S., & Samek, W. (2025). Dyslexify: A mechanistic defense against typographic attacks in CLIP. arXiv. https://doi.org/10.48550/arXiv.2508.20570

**(6)** Liu, B., Ye, W., Xiong, G., He, Z., Sinha, S., & Zhang, A. (2026). Towards robustness against typographic attack with training-free concept localization (SamplingTAR). *European Conference on Computer Vision (ECCV)*. https://doi.org/10.48550/arXiv.2607.02494

**(7)** Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*. https://doi.org/10.48550/arXiv.2010.11929

**(8)** Krizhevsky, A. (2009). Learning multiple layers of features from tiny images (Technical Report). University of Toronto. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf  
*CIFAR-10 source. No DOI — link is fine for JHSS if consistent.*

**(9)** Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., et al. (2021). OpenCLIP. Zenodo / GitHub. https://doi.org/10.5281/zenodo.5143773  
*Confirm Zenodo version DOI when citing software; repo: https://github.com/mlfoundations/open_clip*

**(10)** Yang, A., Pan, J., Lin, J., Men, R., Zhang, Y., Zhou, J., & Zhou, C. (2022). Chinese CLIP: Contrastive vision-language pretraining in Chinese. arXiv. https://doi.org/10.48550/arXiv.2211.01335

**(11)** Bingsu. (n.d.). clip-vit-base-patch32-ko [Computer software]. Hugging Face. https://huggingface.co/Bingsu/clip-vit-base-patch32-ko  
*No paper DOI — cite model card; update if you find a preferred KO CLIP paper.*

**(12)** llm-jp. (n.d.). llm-jp-clip-vit-base-patch16 [Computer software]. Hugging Face. https://huggingface.co/llm-jp/llm-jp-clip-vit-base-patch16  
*Prefer a paper/tech report if llm-jp published one; otherwise model card + access date.*

**(13)** Jaided AI. (n.d.). EasyOCR [Computer software]. GitHub. https://github.com/JaidedAI/EasyOCR  
*Used for OCR+blur spatial upper bound.*

**(14)** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://doi.org/10.48550/arXiv.1201.0490  
*Optional: cite if you name logistic/SVM implementation in Methods.*

---

## Example in-text usage (JHSS style)

Typographic overlays can flip CLIP predictions even when the depicted object is unchanged (1-3). Prior defenses include a learned text prefix (4) and training-free attention-head interventions (5, 6). The present study used CIFAR-10 (8) and four language-specific CLIP checkpoints (1, 9-12).

---

## Still to verify / expand

- [ ] Confirm OpenCLIP Zenodo DOI version used in your environment  
- [ ] Find paper citations for KO / JA checkpoints if available (upgrade from HF cards)  
- [ ] Add access dates for all `(n.d.)` software entries when writing the final Word file  
- [ ] Re-number after first full Intro+Methods prose pass so order = first appearance  
- [ ] Optional: Add more Related Work (Ilharco robustness fine-tuning, other OCR defenses) once §1.3 expands  

## JHSS reminder

- List ≤6 authors in full; if >6, list first 6 + et al.  
- One continuous paragraph per reference (no blank line inside an entry)  
- Live clickable DOI/URL after each entry  
- Consistent style throughout (APA preferred)
