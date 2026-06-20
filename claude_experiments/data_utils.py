"""Dataset loaders returning pixel-space [0,1] 224x224 tensors (no normalization)."""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
from mclip_lib import STL10_CLASSES, CIFAR10_CLASSES

DATA_ROOT = "/ssd4tb/etc/adversarial/data"

_pixel_tf = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),  # -> [0,1]
])


def get_dataset(name):
    if name == "cifar10":
        ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=_pixel_tf)
        classes = CIFAR10_CLASSES
    elif name == "stl10":
        ds = torchvision.datasets.STL10(DATA_ROOT, split="test", download=False, transform=_pixel_tf)
        classes = STL10_CLASSES
    else:
        raise ValueError(name)
    return ds, classes


def get_loader(name, n=None, batch_size=64, seed=0, shuffle_subset=True):
    """Return (loader, classes). If n given, take a class-balanced-ish random subset of size n."""
    ds, classes = get_dataset(name)
    if n is not None and n < len(ds):
        g = np.random.default_rng(seed)
        idx = g.permutation(len(ds))[:n] if shuffle_subset else np.arange(n)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return loader, classes
