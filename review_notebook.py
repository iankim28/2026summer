"""
Code review for dual_encoder_divergence.ipynb.
Runs a headless kernel and executes the setup + architecture cells only
(no training, no dataset download) to catch import/shape/API errors.
"""
import json, sys, queue, time
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\ian\2026summer\notebooks\dual_encoder_divergence.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

from jupyter_client import KernelManager
km = KernelManager(kernel_name='2026summer')
km.start_kernel()
kc = km.client()
kc.start_channels()
print("Starting kernel...", flush=True)
kc.wait_for_ready(timeout=120)
print("Kernel ready.", flush=True)

def run(label, src, timeout=120):
    print(f"\n{'='*55}\n  {label}\n{'='*55}", flush=True)
    msg_id = kc.execute(src)
    t0 = time.time()
    idle = False
    ok = True
    while not idle:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            print(f"  [TIMEOUT]")
            ok = False; break
        mt = msg['msg_type']
        content = msg['content']
        if mt == 'stream':
            print(content.get('text', ''), end='', flush=True)
        elif mt == 'error':
            print(f"ERROR: {content['ename']}: {content['evalue']}", flush=True)
            for tb in content.get('traceback', []):
                # strip ANSI codes roughly
                import re
                print(re.sub(r'\x1b\[[0-9;]*m', '', tb), flush=True)
            ok = False
        elif mt == 'status' and content.get('execution_state') == 'idle':
            idle = True
    print(f"  [{'OK' if ok else 'FAIL'} in {time.time()-t0:.1f}s]", flush=True)
    return ok

all_ok = True

# Run install
all_ok &= run("Cell 1 - install", ''.join(cells[1]['source']))

# Run imports
all_ok &= run("Cell 2 - imports", ''.join(cells[2]['source']))

# Run constants
all_ok &= run("Cell 3 - constants", ''.join(cells[3]['source']))

# Run model
all_ok &= run("Cell 4 - model", ''.join(cells[4]['source']))

# Run translations
all_ok &= run("Cell 5 - translations", ''.join(cells[5]['source']))

# Run dataset (just definitions; skip loader calls that download data)
dataset_src = ''.join(cells[6]['source'])
# Remove the last two lines that actually load CIFAR and build TXT
dataset_defs_only = '\n'.join(
    line for line in dataset_src.splitlines()
    if not line.startswith('loader') and not line.startswith('TXT =')
)
all_ok &= run("Cell 6 - dataset defs (no download)", dataset_defs_only)

# Mock CLASSES so arch cell works without a real loader
mock_classes = "CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']"
run("mock CLASSES", mock_classes)

# Run architecture (will call get_loader(2) for shape check — fast, just definitions check)
# But we need txt_backbone defined for build_txt_lang — skip shape check at bottom,
# just verify the class/function definitions and the model API calls
arch_src = ''.join(cells[7]['source'])
# Keep everything except the shape check at the end (which calls get_loader)
arch_no_shapecheck = arch_src[:arch_src.index('# ── Shape check')]
arch_no_shapecheck += '\nprint("Architecture definitions OK (shape check skipped)")'
all_ok &= run("Cell 7 - architecture defs", arch_no_shapecheck)

# Minimal shape check using model directly
shape_check = """
import torch
_dummy = torch.zeros(2, 3, 224, 224).to(device)
with torch.no_grad():
    _f = encode_image_ml(_dummy, "en")
    print(f"encode_image_ml: {_f.shape}  (expected [2, 512])")
    assert _f.shape == (2, 512), f"WRONG: {_f.shape}"
del _dummy, _f

# Verify encode_text_lang with real tokenizer
_prompts = ["a photo of a cat.", "a photo of a dog."]
# txt_backbone not yet defined -- just test the XLM-R + MeanPooler part
_tokens = tokenizer(_prompts).to(device)
_attn = (_tokens != model.text.config.pad_token_id).long()
with torch.no_grad():
    _out = model.text.transformer(input_ids=_tokens, attention_mask=_attn)
    _masked = _out.last_hidden_state * _attn.unsqueeze(-1).float()
    _pooled = _masked.sum(dim=1) / _attn.sum(-1, keepdim=True).float()
    print(f"XLM-R MeanPooler output: {_pooled.shape}  (expected [2, 768])")
    assert _pooled.shape == (2, 768), f"WRONG: {_pooled.shape}"
    _proj = lang_text_proj["en"](_pooled)
    print(f"lang_text_proj['en'] output: {_proj.shape}  (expected [2, 512])")
    assert _proj.shape == (2, 512), f"WRONG: {_proj.shape}"
del _tokens, _attn, _out, _masked, _pooled, _proj
print("All shape checks passed.")
"""
all_ok &= run("Shape checks", shape_check)

# Verify gradient flows through adapters
grad_check = """
import torch
_x = torch.zeros(2, 3, 224, 224, device=device, requires_grad=False)
_y = torch.zeros(2, dtype=torch.long, device=device)

# Verify adapter grads
_f = encode_image_ml(_x, "en")
_txt = F.normalize(lang_text_proj["en"](torch.zeros(10, 768, device=device)), dim=-1)
_loss = F.cross_entropy(logits_for(_f, _txt), _y)
_loss.backward()

# Check adapter A has grad
_sample_key = "en_0"
_grad = ml_adapters[_sample_key].A.weight.grad
assert _grad is not None, "ml_adapters['en_0'].A.weight has no gradient!"
print(f"ml_adapters['en_0'].A grad norm: {_grad.norm().item():.4f}  (should be non-zero)")

_grad_txt = lang_text_proj["en"].weight.grad
assert _grad_txt is not None, "lang_text_proj['en'].weight has no gradient!"
print(f"lang_text_proj['en'] grad norm: {_grad_txt.norm().item():.4f}  (should be non-zero)")

# Verify backbone is frozen (no grads)
_vit_param = next(iter(model.visual.parameters()))
assert _vit_param.grad is None, "ViT backbone param has gradient -- should be frozen!"
print("ViT backbone correctly frozen (no gradient).")

# Verify PGD gradient flows to input pixels
_xadv = _x.clone().detach()
_xadv.requires_grad_(True)
_feat = encode_image_ml(_xadv, "en")
_l = F.cross_entropy(logits_for(_feat, _txt), _y)
_g = torch.autograd.grad(_l, _xadv)[0]
assert _g is not None and _g.abs().sum() > 0, "PGD gradient is zero!"
print(f"PGD pixel gradient norm: {_g.norm().item():.4f}  (should be non-zero)")
print("All gradient checks passed.")
"""
all_ok &= run("Gradient checks", grad_check)

kc.stop_channels()
km.shutdown_kernel()

print(f"\n{'='*55}")
print(f"  REVIEW RESULT: {'ALL CHECKS PASSED' if all_ok else 'FAILURES DETECTED'}")
print(f"{'='*55}")
