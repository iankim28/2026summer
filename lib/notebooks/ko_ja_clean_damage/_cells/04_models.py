def _clip_feat(out):
    if torch.is_tensor(out):
        return out
    if getattr(out, 'pooler_output', None) is not None:
        return out.pooler_output
    raise TypeError(type(out))

class EnCLIP:
    lang = 'en'
    backend = 'open_clip'
    def __init__(self):
        self.m, _, self.pp = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer('ViT-B-32')
    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL['en'].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)

class KoCLIP:
    lang = 'ko'
    backend = 'hf_vision'
    def __init__(self):
        self.m = AutoModel.from_pretrained(
            'Bingsu/clip-vit-base-patch32-ko',
            attn_implementation='eager').to(DEVICE).eval()
        self.p = AutoProcessor.from_pretrained('Bingsu/clip-vit-base-patch32-ko')
    @torch.no_grad()
    def embed_images(self, imgs):
        pv = self.p(images=imgs, return_tensors='pt').pixel_values.to(DEVICE)
        return F.normalize(_clip_feat(self.m.get_image_features(pixel_values=pv)), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.p(text=[TMPL['ko'].format(w) for w in words], padding=True,
                   return_tensors='pt').to(DEVICE)
        out = self.m.get_text_features(input_ids=t['input_ids'],
                                       attention_mask=t['attention_mask'])
        return F.normalize(_clip_feat(out), dim=-1)

class JaCLIP:
    lang = 'ja'
    backend = 'open_clip'
    def __init__(self):
        mid = 'hf-hub:llm-jp/llm-jp-clip-vit-base-patch16'
        self.m, _, self.pp = open_clip.create_model_and_transforms(mid)
        self.m = self.m.to(DEVICE).eval()
        self.tok = open_clip.get_tokenizer(mid)
    @torch.no_grad()
    def embed_images(self, imgs):
        x = torch.stack([self.pp(im) for im in imgs]).to(DEVICE)
        return F.normalize(self.m.encode_image(x), dim=-1)
    @torch.no_grad()
    def embed_texts(self, words):
        t = self.tok([TMPL['ja'].format(w) for w in words]).to(DEVICE)
        return F.normalize(self.m.encode_text(t), dim=-1)

def classify_batch(model, imgs, words, batch_size=128):
    preds = []
    for i in range(0, len(imgs), batch_size):
        imf = model.embed_images(imgs[i:i+batch_size])
        tf  = model.embed_texts(words)
        preds.append((imf @ tf.t()).argmax(-1).cpu().numpy())
    return np.concatenate(preds)

MODEL_CLS = {'en': EnCLIP, 'ko': KoCLIP, 'ja': JaCLIP}
models = {}
for lang, cls in MODEL_CLS.items():
    t0 = time.time()
    print(f'Loading {lang}...', end=' ', flush=True)
    models[lang] = cls()
    print(f'{time.time()-t0:.1f}s')
TEXT_EMB = {lang: models[lang].embed_texts(CLASSES[lang]).detach() for lang in ALL_LANGS}
print('Models ready.')
