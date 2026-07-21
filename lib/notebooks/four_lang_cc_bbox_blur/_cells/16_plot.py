keys = list(comparison.keys())
n = len(keys)
fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.45 * n + 1.5)), sharey=True)

for ax, which in zip(axes, ['en', 'partner']):
    labels, atk_vals, def_vals = [], [], []
    for k in keys:
        row = comparison[k]
        L = row['L']
        ml = 'en' if which == 'en' else L
        labels.append(k)
        atk_vals.append(100 * row['baseline_acc'][ml])
        def_vals.append(100 * row['defense'][ml]['acc'])
    y = np.arange(len(labels))
    h = 0.35
    ax.barh(y - h / 2, atk_vals, h, label='attacked', color='#c44e52')
    ax.barh(y + h / 2, def_vals, h, label='defended', color='#4c72b0')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    title = 'English CLIP' if which == 'en' else 'Partner CLIP (L)'
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=8)

fig.suptitle('4-lang cc_bbox_blur — attacked vs defended', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/final_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved results/final_comparison.png')

print('\n=== Summary ===')
hdr = f'{"cell":<16} {"thr":>5} {"atkEN":>7} {"defEN":>7} {"atkL":>7} {"defL":>7} {"mean":>7} {"cD":>7} {"full":>5}'
print(hdr)
for k, row in comparison.items():
    L = row['L']
    cdelta = row.get('mean_clean_delta')
    cdelta_s = f'{100*cdelta:+.1f}' if cdelta is not None else '  n/a'
    full_s = 'Y' if row.get('ran_full') else 'n'
    print(
        f'{k:<16} {row["threshold"]:5.2f} '
        f'{100*row["baseline_acc"]["en"]:6.1f}% '
        f'{100*row["defense"]["en"]["acc"]:6.1f}% '
        f'{100*row["baseline_acc"][L]:6.1f}% '
        f'{100*row["defense"][L]["acc"]:6.1f}% '
        f'{100*row["mean_acc"]:6.1f}% '
        f'{cdelta_s:>7} '
        f'{full_s:>5}'
    )
