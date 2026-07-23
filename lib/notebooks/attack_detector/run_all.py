"""Run all code cells as a single script (same order as the notebook)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS = ROOT / '_cells'
ORDER = [
    '01_pip.py',
    '02_imports.py',
    '04_models.py',
    '06_data.py',
    '08_saliency.py',
    '10_mask.py',
    '12_bake.py',
    '14_features.py',
    '16_viz.py',
    '18_detector.py',
]

def main():
    ns = {'__name__': '__main__'}
    for name in ORDER:
        path = CELLS / name
        print(f'\n======== {name} ========', flush=True)
        code = path.read_text(encoding='utf-8')
        exec(compile(code, str(path), 'exec'), ns, ns)
    print('\nAll cells finished.', flush=True)

if __name__ == '__main__':
    main()
