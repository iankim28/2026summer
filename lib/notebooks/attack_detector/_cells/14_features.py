def _entropy(p):
    p = p.ravel().astype(np.float64)
    p = p - p.min()
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum())

def _topk_mass(p, frac):
    flat = p.ravel().astype(np.float64)
    flat = flat - flat.min()
    s = flat.sum()
    if s <= 0:
        return 0.0
    k = max(1, int(round(len(flat) * frac)))
    top = np.partition(flat, -k)[-k:]
    return float(top.sum() / s)

def _gini(p):
    flat = np.sort(p.ravel().astype(np.float64))
    n = len(flat)
    if n == 0 or flat.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * flat).sum()) / (n * flat.sum()) - (n + 1) / n)

def _spatial_kurtosis(p):
    flat = p.ravel().astype(np.float64)
    mu = flat.mean()
    sig = flat.std()
    if sig < 1e-12:
        return 0.0
    z = (flat - mu) / sig
    return float((z ** 4).mean() - 3.0)

def _map_features(cam, prefix):
    a = align_cam(cam)
    mx = float(a.max())
    mn = float(a.mean()) + 1e-12
    cov95 = float((a >= np.percentile(a, 95)).mean())
    _, ncc = ndimage.label(a >= np.percentile(a, 95))
    return {
        f'{prefix}_entropy': _entropy(a),
        f'{prefix}_topk05': _topk_mass(a, 0.05),
        f'{prefix}_topk10': _topk_mass(a, 0.10),
        f'{prefix}_max_over_mean': mx / mn,
        f'{prefix}_gini': _gini(a),
        f'{prefix}_kurtosis': _spatial_kurtosis(a),
        f'{prefix}_cov95': cov95,
        f'{prefix}_ncc95': float(ncc),
    }

def extract_pair_features(cam_en, cam_l):
    feats = {}
    feats.update(_map_features(cam_en, 'en'))
    feats.update(_map_features(cam_l, 'l'))
    inter = n_cam_intersection(cam_en, cam_l)
    feats.update(_map_features(inter, 'inter'))
    ae, al = align_cam(cam_en), align_cam(cam_l)
    feats['en_l_corr'] = float(np.corrcoef(ae.ravel(), al.ravel())[0, 1]) if ae.std() > 0 and al.std() > 0 else 0.0
    hot_e = ae >= np.percentile(ae, 95)
    hot_l = al >= np.percentile(al, 95)
    union = (hot_e | hot_l).sum()
    feats['en_l_iou95'] = float((hot_e & hot_l).sum() / union) if union > 0 else 0.0
    return feats

def build_feature_matrix(clean_cams, atk_cams):
    rows_feat, y_labels, img_ids = [], [], []
    for i in range(n_images):
        rows_feat.append(extract_pair_features(clean_cams['cams_en'][i], clean_cams['cams_l'][i]))
        y_labels.append(0); img_ids.append(i)
    for i in range(n_images):
        rows_feat.append(extract_pair_features(atk_cams['cams_en'][i], atk_cams['cams_l'][i]))
        y_labels.append(1); img_ids.append(i)
    names = list(rows_feat[0].keys())
    X = np.array([[r[k] for k in names] for r in rows_feat], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_labels, dtype=np.int64)
    img_ids = np.array(img_ids, dtype=np.int64)
    return X, y, img_ids, names

print('Feature helpers ready.')
