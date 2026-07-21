hf = load_dataset('uoft-cs/cifar10', split='test')
label_key = 'label' if 'label' in hf.column_names else 'labels'
image_key = 'img'   if 'img'   in hf.column_names else 'image'

_sample_path = '../image_samples/CIFAR10_BALANCED_1000_SAMPLE.json'
with open(_sample_path, encoding='utf-8') as f:
    _saved = json.load(f)

idx  = _saved['idx']
rows = hf.select(idx)
true = np.array(rows[label_key])
assert len(idx) == 1000 and np.array_equal(true, np.array(_saved['true']))

rng    = random.Random(0)
target = np.array([rng.choice([c for c in range(10) if c != int(true[k])])
                   for k in range(len(idx))])

clean_224 = [im.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
             for im in rows[image_key]]

all_idx  = np.arange(len(clean_224))
tune_idx = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])
print(f'Loaded {len(clean_224)} images; tune subset = {len(tune_idx)}')

_FONT_CACHE = {}

def _font_paths():
    if platform.system() == 'Windows':
        wf = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
        cjk = os.path.join(wf, 'msyh.ttc')
        lat = os.path.join(wf, 'arial.ttf')
        ko  = os.path.join(wf, 'malgun.ttf')
        if not os.path.isfile(ko):
            ko = cjk
        return cjk, lat, ko
    cjk = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    lat = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    if not os.path.isfile(cjk):
        cjk = '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf'
    return cjk, lat, cjk

_CJK_FONT, _LAT_FONT, _KO_FONT = _font_paths()

def _font_for_lang(lang):
    if lang == 'en':
        return _LAT_FONT
    if lang == 'ko':
        return _KO_FONT
    return _CJK_FONT

def _get_font(fp, size=FONT_SIZE):
    key = (fp or '__default__', size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]

def _rects_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

def _random_nonoverlapping_rect(rng_, bw, bh, placed):
    x_hi = max(0, DISPLAY_SIZE - bw)
    y_hi = max(0, DISPLAY_SIZE - bh)
    rx = ry = 0
    for _ in range(64):
        rx = rng_.randint(0, x_hi) if x_hi > 0 else 0
        ry = rng_.randint(0, y_hi) if y_hi > 0 else 0
        rect = (rx, ry, rx + bw, ry + bh)
        if all(not _rects_overlap(rect, p) for p in placed):
            return rect
    return (rx, ry, rx + bw, ry + bh)

def draw_dual_box(img, word0, lang0, word1, lang1, img_idx, already_224=False):
    if not already_224:
        img = img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    else:
        img = img.copy()
    draw = ImageDraw.Draw(img)
    placed = []
    for box_i, (word, lang) in enumerate([(word0, lang0), (word1, lang1)]):
        font = _get_font(_font_for_lang(lang))
        bb   = draw.textbbox((0, 0), word, font=font)
        bw   = (bb[2] - bb[0]) + 2 * PAD
        bh   = (bb[3] - bb[1]) + PAD + 12
        rng_ = random.Random(int(img_idx) * NUM_BOXES + box_i)
        rect = _random_nonoverlapping_rect(rng_, bw, bh, placed)
        placed.append(rect)
        rx, ry, rx2, ry2 = rect
        draw.rectangle([rx, ry, rx2, ry2], fill='white')
        draw.text((rx + PAD - bb[0], ry + PAD - bb[1]), word, fill='black', font=font)
    return img

def build_attack(attack, L):
    out = []
    for i in range(len(clean_224)):
        t = int(target[i])
        en_w = CLASSES['en'][t]
        l_w  = CLASSES[L][t]
        if attack == 'uni_en':
            img = draw_dual_box(clean_224[i], en_w, 'en', en_w, 'en', i, True)
        elif attack == 'uni_l':
            img = draw_dual_box(clean_224[i], l_w, L, l_w, L, i, True)
        else:
            img = draw_dual_box(clean_224[i], en_w, 'en', l_w, L, i, True)
        out.append(img)
    return out

_strip = Image.new('RGB', (DISPLAY_SIZE * 4, 80), (240, 240, 240))
_sd = ImageDraw.Draw(_strip)
for j, (lang, word) in enumerate([
    ('en', 'airplane'), ('zh', '飞机'), ('ko', '비행기'), ('ja', '飛行機')
]):
    f = _get_font(_font_for_lang(lang), 28)
    _sd.text((j * DISPLAY_SIZE + 10, 20), f'{lang}: {word}', fill='black', font=f)
_strip.save('results/font_check.png')
print('Fonts: LAT=', _LAT_FONT, 'CJK=', _CJK_FONT, 'KO=', _KO_FONT)
print('Saved results/font_check.png')

clean_acc = {
    ml: float((classify_batch(models[ml], clean_224, CLASSES[ml]) == true).mean())
    for ml in ALL_LANGS
}
print('Clean acc:', {k: f'{100*v:.1f}%' for k, v in clean_acc.items()})
