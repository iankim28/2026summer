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

attack_pos = _saved['attack_pos']
assert len(attack_pos['en']) == len(idx) and len(attack_pos['l']) == len(idx)

rng    = random.Random(0)
target = np.array([rng.choice([c for c in range(10) if c != int(true[k])])
                   for k in range(len(idx))])

clean_224 = [im.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
             for im in rows[image_key]]

all_idx  = np.arange(len(clean_224))
tune_idx = np.concatenate([np.where(true == c)[0][:10] for c in range(10)])
print(f'Loaded {len(clean_224)} images; tune subset = {len(tune_idx)}')
print(f"Attack positions: frozen from sample JSON (ref {attack_pos['ref_bw']}x{attack_pos['ref_bh']})")

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

def _clamp_xy(xy, bw, bh):
    x, y = int(xy[0]), int(xy[1])
    x = max(0, min(x, max(0, DISPLAY_SIZE - bw)))
    y = max(0, min(y, max(0, DISPLAY_SIZE - bh)))
    return x, y

def draw_dual_box(img, word0, lang0, word1, lang1, img_idx, already_224=False):
    """Place boxes at frozen EN/L anchors from the sample JSON."""
    if not already_224:
        img = img.convert('RGB').resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BICUBIC)
    else:
        img = img.copy()
    draw = ImageDraw.Draw(img)
    xy0 = attack_pos['en'][int(img_idx)]
    xy1 = attack_pos['l'][int(img_idx)]
    for word, lang, xy in [(word0, lang0, xy0), (word1, lang1, xy1)]:
        font = _get_font(_font_for_lang(lang))
        bb   = draw.textbbox((0, 0), word, font=font)
        bw   = (bb[2] - bb[0]) + 2 * PAD
        bh   = (bb[3] - bb[1]) + PAD + 12
        rx, ry = _clamp_xy(xy, bw, bh)
        draw.rectangle([rx, ry, rx + bw, ry + bh], fill='white')
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
