export const meta = {
  name: 'notebook-review',
  description: 'Adversarial multi-dimension review of the Colab notebook (compat, correctness, pedagogy, budget)',
  phases: [
    { title: 'Review', detail: 'one agent per review dimension' },
    { title: 'Verify', detail: 'skeptic verifies each finding against the code' },
  ],
}

const FILES = `Files (read what you need):
- /ssd4tb/etc/adversarial/multilingual_consensus_colab.ipynb  (THE NOTEBOOK under review)
- /ssd4tb/etc/adversarial/build_notebook.py  (its generator; edit findings should reference this since the .ipynb is generated from it)
- /ssd4tb/etc/adversarial/FINDINGS.md  (validated results the notebook must faithfully reproduce)
- /ssd4tb/etc/adversarial/mclip_lib.py, attacks.py, run_transfer.py, analyze.py, embed_sim.py, denoiser.py, mclip_fredde.py  (the validated reference implementation)
The notebook targets a FREE Google Colab T4 GPU (16GB), end-to-end ~10-15 min.`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] },
          location: { type: 'string', description: 'cell/function/line area, e.g. "denoiser train cell" or build_notebook.py region' },
          issue: { type: 'string', description: 'what is wrong and why it matters on Colab/for correctness' },
          fix: { type: 'string', description: 'concrete proposed fix' },
        },
        required: ['severity', 'location', 'issue', 'fix'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    real: { type: 'boolean', description: 'true if this is a genuine problem worth fixing' },
    reason: { type: 'string' },
    corrected_severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit', 'not-a-bug'] },
    corrected_fix: { type: 'string' },
  },
  required: ['real', 'reason', 'corrected_severity', 'corrected_fix'],
}

const DIMENSIONS = [
  { key: 'colab-compat', prompt: `You are reviewing for COLAB COMPATIBILITY on a fresh free T4 runtime.
${FILES}
Check rigorously: will the %pip install cell get everything needed (open_clip_torch pulls torch/transformers; does the XLM-R tokenizer need sentencepiece/protobuf? is anything missing or version-fragile on Colab's preinstalled stack)? GPU assertion correctness; dataset download sizes/time (CIFAR vs STL); use of magics; matplotlib display on Colab; any dependency on local repo files that won't exist on Colab; writes to cwd 'data'; Python-version assumptions; non-ASCII (Korean/Japanese) string handling. Report concrete problems only.` },
  { key: 'correctness', prompt: `You are reviewing for CODE CORRECTNESS versus the validated reference.
${FILES}
Verify the notebook faithfully reproduces the validated experiment: pixel-space L-inf attack (eps ball projection + [0,1] clamp + normalization folded into forward), classification via cosine*logit_scale, transfer-fraction formula, agreement computation, detector disagreement scores + ROC-AUC (tie handling, sign/direction so AUC<0.5 means 'worse than random'), the consensus denoiser training (pseudo-label source, clean-input preservation, fidelity), and the ADAPTIVE attack (gradients truly flow through the differentiable denoiser; eval in den.eval()). Flag any bug, off-by-one, wrong-variable, or logic that diverges from the reference and would change conclusions.` },
  { key: 'pedagogy', prompt: `You are reviewing for PEDAGOGICAL COMPLETENESS and CLAIM ACCURACY.
${FILES}
Check that every markdown claim is actually supported by what the adjacent code computes (no overclaiming, no result stated that the code doesn't produce); that the narrative is thorough and self-contained (covers H1/transfer, mechanism, ensemble, disagreement detector, H2, denoiser non-adaptive vs adaptive, ablation, verdict); that interpretations are correct; and that a reader new to the topic can follow it. Flag misleading, missing, or unsupported explanations.` },
  { key: 'budget', prompt: `You are reviewing for COMPUTE-BUDGET REALISM on a free Colab T4 (16GB, end-to-end ~10-15 min target).
${FILES}
Identify the most expensive cells (transfer sweep, denoiser training, adaptive eval). Estimate whether the default config knobs keep total runtime ~10-15 min on a T4 and peak GPU memory < 16GB (DnCNN at 224^2 batch 40 can be memory-heavy). Flag any cell that risks OOM or timeout, and propose knob changes (batch size, n, steps) if needed. Also check the config knobs are coherent and that default RUN_ABLATION=False is the cheap path.` },
]

phase('Review')
const reviews = await parallel(DIMENSIONS.map(d => () =>
  agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA })
    .then(r => ({ key: d.key, findings: (r && r.findings) || [] }))
))

const all = []
for (const r of reviews.filter(Boolean)) {
  for (const f of r.findings) all.push({ ...f, dimension: r.key })
}
log(`collected ${all.length} raw findings across ${reviews.filter(Boolean).length} dimensions`)

phase('Verify')
const verified = await parallel(all.map(f => () =>
  agent(`Skeptically verify this review finding against the actual code. Default to real=false if it is not a genuine problem or is already handled.
${FILES}

FINDING (dimension=${f.dimension}, claimed severity=${f.severity}):
location: ${f.location}
issue: ${f.issue}
proposed fix: ${f.fix}

Read the relevant code and decide whether this is a real problem worth fixing. Be precise.`,
    { label: `verify:${f.dimension}`, phase: 'Verify', schema: VERDICT_SCHEMA })
    .then(v => ({ finding: f, verdict: v }))
    .catch(() => null)
))

const confirmed = verified.filter(Boolean).filter(x => x.verdict && x.verdict.real)
const order = { blocker: 0, major: 1, minor: 2, nit: 3, 'not-a-bug': 4 }
confirmed.sort((a, b) => (order[a.verdict.corrected_severity] ?? 9) - (order[b.verdict.corrected_severity] ?? 9))

return {
  raw_count: all.length,
  confirmed_count: confirmed.length,
  confirmed: confirmed.map(x => ({
    severity: x.verdict.corrected_severity,
    dimension: x.finding.dimension,
    location: x.finding.location,
    issue: x.finding.issue,
    fix: x.verdict.corrected_fix,
    reason: x.verdict.reason,
  })),
}
