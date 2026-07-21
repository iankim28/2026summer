def n_cam_intersection(*cams):
    return np.minimum.reduce([align_cam(c) for c in cams])

def dilate_mask(mask, iterations=3):
    m = mask.astype(bool)
    for _ in range(iterations):
        pad = np.pad(m, 1, mode='constant', constant_values=False)
        m = (pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:] |
             pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
             pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:])
    return m

def cam_to_mask(saliency, threshold=0.85, dilate=3):
    thr = np.percentile(saliency, threshold * 100)
    mask = saliency >= thr
    if dilate > 0:
        mask = dilate_mask(mask, iterations=dilate)
    return mask

def filter_mask_components(mask, top_k=2, bbox_snap=False):
    labeled, n = ndimage.label(mask.astype(bool))
    if n == 0:
        return mask.astype(bool)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    keep = set(np.argsort(sizes)[::-1][:top_k] + 1)
    out = np.zeros_like(mask, dtype=bool)
    for i in keep:
        comp = labeled == i
        if bbox_snap:
            ys, xs = np.where(comp)
            out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        else:
            out |= comp
    return out

def apply_mask(pil_img, mask, fill='blur'):
    arr = np.array(pil_img.convert('RGB'))
    m = mask.astype(bool)
    if mask.shape != arr.shape[:2]:
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
            arr.shape[1::-1], Image.NEAREST)) > 127
    out = arr.copy()
    if fill == 'blur':
        blurred = np.array(Image.fromarray(arr).filter(
            ImageFilter.GaussianBlur(radius=BLUR_RADIUS)))
        out[m] = blurred[m]
    else:
        mean = arr[~m].mean(0) if (~m).any() else arr.reshape(-1, 3).mean(0)
        out[m] = mean
    return Image.fromarray(out.astype(np.uint8))

def _shrink_to_max_coverage(saliency, mask, max_coverage, dilate, top_k, bbox_snap):
    """Raise percentile until mask coverage <= max_coverage (or thr hits 0.99)."""
    if max_coverage is None or float(mask.mean()) <= max_coverage:
        return mask
    for thr in np.linspace(0.90, 0.99, 10):
        m = cam_to_mask(saliency, thr, dilate=dilate)
        m = filter_mask_components(m, top_k=top_k, bbox_snap=bbox_snap)
        if float(m.mean()) <= max_coverage:
            return m
    return mask

def build_cc_bbox_blur_mask(
    cam_en, cam_l, threshold=0.95, dilate=3, top_k=2,
    bbox_snap=True, max_coverage=None,
):
    inter = n_cam_intersection(cam_en, cam_l)
    mask = cam_to_mask(inter, threshold, dilate=dilate)
    mask = filter_mask_components(mask, top_k=top_k, bbox_snap=bbox_snap)
    if max_coverage is not None and float(mask.mean()) > max_coverage:
        mask = _shrink_to_max_coverage(
            inter, mask, max_coverage, dilate, top_k, bbox_snap)
    return mask

print('Masking helpers ready.')
