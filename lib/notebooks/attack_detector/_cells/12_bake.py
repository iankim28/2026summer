def load_or_bake_pair(L, attacked_imgs, out_dir):
    cache_dir = Path(out_dir) / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'attn_en_{L}_clean_multi.npz'
    if cache_path.exists():
        print(f'Loading cache {cache_path}')
        z = np.load(cache_path, allow_pickle=False)
        # Support legacy zh key names from Step 1
        def _get(prefix, new, old):
            if new in z.files:
                return z[new]
            return z[old]
        return {
            'clean': {
                'cams_en': z['clean_cams_en'],
                'cams_l': _get('clean', 'clean_cams_l', 'clean_cams_zh'),
                'preds_en': z['clean_preds_en'],
                'preds_l': _get('clean', 'clean_preds_l', 'clean_preds_zh'),
            },
            'atk': {
                'cams_en': z['atk_cams_en'],
                'cams_l': _get('atk', 'atk_cams_l', 'atk_cams_zh'),
                'preds_en': z['atk_preds_en'],
                'preds_l': _get('atk', 'atk_preds_l', 'atk_preds_zh'),
            },
        }

    def _bake(images, tag):
        ce, cl, pe, pl = [], [], [], []
        t0 = time.time()
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f'  [{tag}] {i}/{len(images)}...', flush=True)
            p_en, c_en = classify_and_attn('en', img, 'last')
            p_l, c_l = classify_and_attn(L, img, 'last')
            pe.append(p_en); ce.append(c_en.astype(np.float32))
            pl.append(p_l); cl.append(c_l.astype(np.float32))
        print(f'  [{tag}] done in {time.time()-t0:.1f}s')
        return {
            'cams_en': np.stack(ce), 'cams_l': np.stack(cl),
            'preds_en': np.array(pe, dtype=np.int64),
            'preds_l': np.array(pl, dtype=np.int64),
        }

    print(f'Baking clean saliency EN&{L}...')
    clean = _bake(clean_224, f'clean/{L}')
    print(f'Baking attacked saliency EN&{L}...')
    atk = _bake(attacked_imgs, f'atk/{L}')
    np.savez_compressed(
        cache_path,
        clean_cams_en=clean['cams_en'], clean_cams_l=clean['cams_l'],
        clean_preds_en=clean['preds_en'], clean_preds_l=clean['preds_l'],
        atk_cams_en=atk['cams_en'], atk_cams_l=atk['cams_l'],
        atk_preds_en=atk['preds_en'], atk_preds_l=atk['preds_l'],
    )
    print(f'Saved {cache_path}')
    return {'clean': clean, 'atk': atk}

print('Bake helpers ready.')
