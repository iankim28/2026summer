import os, platform, random, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
from scipy import ndimage

os.makedirs('results', exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)

DISPLAY_SIZE = 224
NUM_BOXES    = 2
FONT_SIZE    = 24
PAD          = 8
BLUR_RADIUS  = 12
THRESHOLDS   = [0.75, 0.80, 0.85, 0.90, 0.95]
PARETO_THRESHOLDS = [0.85, 0.90, 0.95]
PARTNER_LANGS = ['ko', 'ja']
ATTACKS = ['uni_en', 'uni_l', 'multi']
VARIANTS = ['baseline', 'thr_floor_095', 'pareto_tune', 'tight_dilate', 'no_bbox']

CLASSES = {
    'en': ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
    'ko': ['비행기','자동차','새','고양이','사슴','개','개구리','말','배','트럭'],
    'ja': ['飛行機','自動車','鳥','猫','鹿','犬','カエル','馬','船','トラック'],
}
TMPL = {
    'en': 'a photo of a {}.',
    'ko': '{}의 사진.',
    'ja': '{}の写真。',
}
ALL_LANGS = ['en', 'ko', 'ja']
