"""Assemble attack_detector.ipynb from _cells/ sources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS_DIR = ROOT / '_cells'
OUT = ROOT / 'attack_detector.ipynb'

ORDER = [
    '00_intro.md',
    '01_pip.py',
    '02_imports.py',
    '03_models_h.md',
    '04_models.py',
    '05_data_h.md',
    '06_data.py',
    '07_saliency_h.md',
    '08_saliency.py',
    '09_mask_h.md',
    '10_mask.py',
    '11_bake_h.md',
    '12_bake.py',
    '13_features_h.md',
    '14_features.py',
    '15_viz_h.md',
    '16_viz.py',
    '17_detector_h.md',
    '18_detector.py',
]


def make_cell(path: Path):
    raw = path.read_text(encoding='utf-8')
    lines = [ln + '\n' for ln in raw.splitlines()]
    if not lines:
        lines = ['\n']
    elif not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    if path.suffix == '.md':
        return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': lines,
    }


def main():
    cells = [make_cell(CELLS_DIR / name) for name in ORDER]
    nb = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
        },
        'cells': cells,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {OUT} ({len(cells)} cells)')


if __name__ == '__main__':
    main()
