def _norm_cam(cam):
    cam = np.maximum(cam if isinstance(cam, np.ndarray) else cam.cpu().numpy(), 0)
    cam -= cam.min()
    mx = cam.max()
    return cam / mx if mx > 0 else cam

def align_cam(cam, size=DISPLAY_SIZE):
    return np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ) / 255.0

def _make_openclip_hook(collector):
    def hook(module, inputs, output):
        q_in = inputs[0]
        if getattr(module, 'batch_first', False):
            B, L, D = q_in.shape
        else:
            L, B, D = q_in.shape
            q_in = q_in.transpose(0, 1).contiguous()
        n_head = module.num_heads
        hd = D // n_head
        with torch.no_grad():
            qkv = F.linear(q_in, module.in_proj_weight, module.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)
            q = q.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            k = k.reshape(B, L, n_head, hd).permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
            attn = attn.softmax(dim=-1)
        collector.append(attn[0].detach().cpu())
    return hook

def _build_attn_cam(all_attns, variant='last'):
    a = all_attns[-1]
    cls_row = a.mean(0)[0, 1:]
    if variant == 'rollout':
        L = all_attns[0].shape[-1]
        rollout = torch.eye(L)
        for att in all_attns:
            a_r = 0.5 * att.mean(0) + 0.5 * torch.eye(L)
            a_r = a_r / a_r.sum(-1, keepdim=True)
            rollout = a_r @ rollout
        cls_row = rollout[0, 1:]
    n = int(round(cls_row.shape[0] ** 0.5))
    return _norm_cam(cls_row.reshape(n, n).numpy())

def classify_and_attn(lang, pil_img, variant='last'):
    wrapper = models[lang]
    if wrapper.backend == 'open_clip':
        x = wrapper.pp(pil_img).unsqueeze(0).to(DEVICE)
        collector = []
        handles = [
            rb.attn.register_forward_hook(_make_openclip_hook(collector))
            for rb in wrapper.m.visual.transformer.resblocks
        ]
        with torch.no_grad():
            feat = wrapper.m.visual(x)
            imf = F.normalize(feat, dim=-1)
            pred = int((imf @ TEXT_EMB[lang].T).squeeze().argmax().item())
        for h in handles:
            h.remove()
        return pred, _build_attn_cam(collector, variant)

    pv = wrapper.p(images=[pil_img], return_tensors='pt').pixel_values.to(DEVICE)
    with torch.no_grad():
        vis_out = wrapper.m.vision_model(pixel_values=pv, output_attentions=True)
        if hasattr(wrapper.m, 'visual_projection'):
            proj = wrapper.m.visual_projection(vis_out.pooler_output)
        else:
            proj = vis_out.pooler_output
        imf = F.normalize(proj, dim=-1)
        pred = int((imf @ TEXT_EMB[lang].T).squeeze().argmax().item())
    attns = [a[0].cpu() for a in vis_out.attentions]
    return pred, _build_attn_cam(attns, variant)

for lang in ALL_LANGS:
    p, cam = classify_and_attn(lang, clean_224[0], 'last')
    print(f'  {lang}: pred={p} cam={cam.shape}')
print('Saliency ready.')
