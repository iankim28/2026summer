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
from transformers import (
    ChineseCLIPModel, ChineseCLIPProcessor, AutoModel, AutoProcessor,
)
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.calibration import CalibratedClassifierCV

os.makedirs('results', exist_ok=True)

assert torch.cuda.is_available(), 'CUDA required — install a CUDA build of PyTorch'
DEVICE = 'cuda'
print('Device:', DEVICE, torch.cuda.get_device_name(0))

DISPLAY_SIZE = 224
NUM_BOXES    = 2
FONT_SIZE    = 24
PAD          = 8
BLUR_RADIUS  = 12
PARTNER_LANGS = ['zh', 'ko', 'ja']
ATTACK       = 'multi'
DEFENSE_THR  = 0.95  # PROTOCOL floor
SPLIT_SEED   = 0
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
ATTACK_RECALL_TARGET = 0.99  # catch almost all attacks so gated atk drop stays <1pp
# If True, skip partners that already have gated_comparison.json.
SKIP_EXISTING = True  # caches + results from recall=0.99 retune are current

CLASSES = {
    'en': ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
    'zh': ['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车'],
    'ko': ['비행기','자동차','새','고양이','사슴','개','개구리','말','배','트럭'],
    'ja': ['飛行機','自動車','鳥','猫','鹿','犬','カエル','馬','船','トラック'],
}
TMPL = {
    'en': 'a photo of a {}.',
    'zh': '一张{}的照片。',
    'ko': '{}의 사진.',
    'ja': '{}の写真。',
}
ALL_LANGS = ['en', 'zh', 'ko', 'ja']
print(f'Scope: EN&L for L={PARTNER_LANGS} / {ATTACK} / thr={DEFENSE_THR}')
