Multilingual Consensus Purification as a Defense Against Adversarial Attacks on Zero-Shot Vision–Language Models
Abstract
Multilingual zero-shot models (e.g., CLIP) label an image by matching its embedding to text labels, with many languages sharing one image encoder. An attack targeting one language's labels misaligns the shared embedding for that language while often leaving the others correct, so the languages disagree. We test whether that disagreement defends against such alignment-side attacks, using two training-free baselines—a multilingual ensemble and a cross-lingual disagreement detector—and a self-supervised consensus-purification denoiser, evaluated against single-language and adaptive multi-language attacks, including an attack that backpropagates through the denoiser. Language-agnostic attacks, which preserve consensus, are out of scope.
 
Figure 1. An imperceptible perturbation flips the prediction (top). This work targets language-specific attacks (bottom), where languages disagree and a consensus-purification denoiser restores the correct class; consensus-preserving language-agnostic attacks are out of scope.
1. Background and problem
A multilingual zero-shot model classifies an image by its nearest text label, expressing labels in many languages through one shared image encoder. An attack on a specific language's labels misaligns the shared embedding for that language but may leave the others correct, creating cross-lingual disagreement—the signal this work exploits. Attacks that instead corrupt the shared embedding directly preserve agreement (on the wrong class) and bound the scope. We study whether this disagreement can defend zero-shot classification, both as a test-time check and as a learned purification signal, and measure its limits under adaptive attackers.
2. Research questions and hypotheses
1.	How far does a single-language attack transfer across languages, and how does it depend on ε and the language pair?
2.	Can a consensus-purification denoiser restore accuracy under language-specific attacks, and does it survive an attack through the denoiser?
•	H1.  Transfer is partial, rising with ε and with language closeness.
•	H2.  The attacker's budget grows as more languages are targeted at once.
•	H3.  The denoiser recovers accuracy non-adaptively, with reduced (possibly little) benefit under attacks that pass through it.
3. Methodology
Use a frozen multilingual model (M-CLIP with a ViT-B/32 image encoder) so all languages share one image encoder; only the small denoiser is trained. Datasets are chosen for clean, translatable class names: STL-10 (primary), CIFAR-10 (fallback), Imagenette (higher-resolution check). Languages: English (target and baseline), Korean (primary contrast), plus Spanish, French, and Japanese for the ensemble-size study—mixing close and distant languages. Pipeline: cache per-language label embeddings, encode each image once, and classify by cosine similarity; these per-language scores feed the attacks, baselines, and the denoiser.
Attacks are white-box, per-image, L∞-bounded (ε ∈ {2,4,8,16}/255), via FGSM and PGD:
•	Single-language:  maximize the loss on one language's labels (English); the other languages reveal transfer.
•	Adaptive multi-language:  maximize the summed loss over several languages, tracing how attacker cost scales with the number of languages.
•	Language-agnostic:  embedding-distance, no labels — used only as an out-of-scope negative control.
Defenses:
•	Multilingual ensemble:  softmax-average the per-language scores and take the argmax.
•	Disagreement detector:  flag an image when cross-language disagreement exceeds a threshold, swept for an ROC curve.
•	Consensus-purification denoiser (main):  a small residual CNN trained self-supervised on language-specific adversarial examples to restore cross-lingual agreement—matching the clean image's consensus prediction (a pseudo-label needing no human labels), with a fidelity term to prevent collapse. At test time, purify then classify.
Evaluation: robust accuracy per language and for the ensemble vs. ε (with and without the denoiser); transfer fraction (targeted vs. non-targeted languages); the attacker-cost curve (minimum ε to defeat the ensemble vs. number of languages); detector ROC-AUC; and—critically—robust accuracy when the attack backpropagates through the denoiser, compared to the non-adaptive case.
4. Expected results
A single-language attack should transfer only partially, so the ensemble and detector help, with attacker cost rising as more languages must be fooled at once. The denoiser should recover substantial accuracy non-adaptively; under the through-the-denoiser attack its benefit may shrink, possibly to little—an informative result consistent with the known fragility of purification defenses. The contribution is the measured transfer and attacker-cost curves and an honest adaptive evaluation of consensus purification.
6. References
•	Radford et al. (2021) — CLIP; Goodfellow et al. (2015) — FGSM; Madry et al. (2018) — PGD.
•	Athalye, Carlini & Wagner (2018) — obfuscated gradients (why purification needs adaptive evaluation); Nie et al. (2022) — diffusion-based adversarial purification.
•	Multilingual-CLIP (M-CLIP) / multilingual SigLIP — cite the specific model used.
