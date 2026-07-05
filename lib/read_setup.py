import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
nb_path = Path(__file__).resolve().parents[1] / 'lib' / 'notebooks' / 'updated_multilingual_consensus_colab.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
for i in [2, 3, 5, 7, 9, 11]:
    print(f'=== CELL {i} ({cells[i]["cell_type"]}) ===')
    print(''.join(cells[i]['source']))
    print()
