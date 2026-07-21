def run_defense(L, images, indices, threshold=0.95, label=''):
    # Apply EN∩L cc_bbox_blur; return preds for EN and L.
    imgs_sub = [images[i] for i in indices]
    true_sub = true[indices]
    n = len(indices)
    print(f'  defense {label} L={L} n={n} thr={threshold}...', end=' ', flush=True)
    t0 = time.time()
    masked_imgs, coverages = [], []
    for img in imgs_sub:
        _, cam_en = classify_and_attn('en', img, 'last')
        _, cam_l = classify_and_attn(L, img, 'last')
        mask = build_cc_bbox_blur_mask(cam_en, cam_l, threshold=threshold)
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

print('Runner ready.')
