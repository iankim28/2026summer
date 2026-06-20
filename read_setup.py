import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'd:\ian\2026summer\notebooks\updated_multilingual_consensus_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
for i in [2, 3, 5, 7, 9, 11]:
    print(f'=== CELL {i} ({cells[i]["cell_type"]}) ===')
    print(''.join(cells[i]['source']))
    print()
