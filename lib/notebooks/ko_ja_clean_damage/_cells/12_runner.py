def precompute_cams(L, images, indices, label=''):
    """Return list of (cam_en, cam_l) aligned to indices order."""
    imgs_sub = [images[i] for i in indices]
    n = len(indices)
    print(f'  cams {label} L={L} n={n}...', end=' ', flush=True)
    t0 = time.time()
    cams = []
    for img in imgs_sub:
        _, cam_en = classify_and_attn('en', img, 'last')
        _, cam_l = classify_and_attn(L, img, 'last')
        cams.append((cam_en, cam_l))
    print(f'{time.time() - t0:.1f}s')
    return cams

def run_defense_cached(L, images, indices, cams, mask_cfg, label=''):
    """Apply EN∩L cc_bbox_blur using precomputed cams; return preds for EN and L."""
    imgs_sub = [images[i] for i in indices]
    true_sub = true[indices]
    n = len(indices)
    thr = mask_cfg.get('threshold', 0.95)
    print(f'  defense {label} L={L} n={n} thr={thr} '
          f'dilate={mask_cfg.get("dilate", 3)} '
          f'bbox={mask_cfg.get("bbox_snap", True)}...', end=' ', flush=True)
    t0 = time.time()
    masked_imgs, coverages = [], []
    for img, (cam_en, cam_l) in zip(imgs_sub, cams):
        mask = build_cc_bbox_blur_mask(
            cam_en, cam_l,
            threshold=mask_cfg.get('threshold', 0.95),
            dilate=mask_cfg.get('dilate', 3),
            top_k=mask_cfg.get('top_k', 2),
            bbox_snap=mask_cfg.get('bbox_snap', True),
            max_coverage=mask_cfg.get('max_coverage'),
        )
        coverages.append(float(mask.mean()))
        masked_imgs.append(apply_mask(img, mask, fill='blur'))
    preds = {
        'en': classify_batch(models['en'], masked_imgs, CLASSES['en']),
        L: classify_batch(models[L], masked_imgs, CLASSES[L]),
    }
    elapsed = time.time() - t0
    print(f'{elapsed:.1f}s  cov={100 * np.mean(coverages):.1f}%')
    return {
        'preds': preds,
        'true': true_sub,
        'coverage': float(np.mean(coverages)),
        'time_s': elapsed,
    }

def metrics_for_pair(preds, true_sub, target_sub, L, baseline_acc, baseline_asr):
    out = {}
    for ml in ['en', L]:
        out[ml] = {
            'acc': float((preds[ml] == true_sub).mean()),
            'asr': float((preds[ml] == target_sub).mean()),
            'baseline_acc': baseline_acc[ml],
            'baseline_asr': baseline_asr[ml],
        }
    out['mean_acc'] = 0.5 * (out['en']['acc'] + out[L]['acc'])
    return out

def clean_delta_from_preds(preds, true_sub, L):
    deg = {}
    for ml in ['en', L]:
        masked = float((preds[ml] == true_sub).mean())
        deg[ml] = {
            'baseline_acc': clean_acc[ml],
            'masked_acc': masked,
            'delta_acc': masked - clean_acc[ml],
        }
    mean_d = 0.5 * (deg['en']['delta_acc'] + deg[L]['delta_acc'])
    return deg, mean_d

print('Runner ready.')
