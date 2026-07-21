variant_colors = {
    'baseline': '#4c72b0',
    'thr_floor_095': '#55a868',
    'pareto_tune': '#c44e52',
    'tight_dilate': '#8172b3',
    'no_bbox': '#ccb974',
}

cells = [(L, a) for L in PARTNER_LANGS for a in ATTACKS]
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

x = np.arange(len(cells))
width = 0.15
offsets = (np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2) * width

# Top: clean damage as positive pp drop (lower bar = less damage = better).
# Bottom: mean defended accuracy under attack (higher = better).
for ax, metric, ylabel, title in [
    (axes[0], 'clean_damage', 'Clean damage (pp)',
     'Clean-image damage (lower = better)'),
    (axes[1], 'mean_acc', 'Mean defended acc (%)',
     'Attacked mean defended accuracy (higher = better)'),
]:
    for vi, variant in enumerate(VARIANTS):
        vals = []
        for L, attack in cells:
            key = f'{L}/{attack}/{variant}'
            row = comparison.get(key)
            if row is None:
                vals.append(0.0)
            elif metric == 'clean_damage':
                vals.append(-100 * row['mean_clean_delta'])  # drop as positive pp
            else:
                vals.append(100 * row['mean_acc'])
        ax.bar(x + offsets[vi], vals, width, label=variant,
               color=variant_colors[variant])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=len(VARIANTS), loc='best')

axes[0].set_ylim(0, None)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{L}/{a}' for L, a in cells], rotation=30, ha='right')
fig.suptitle('KO/JA clean-damage ablation — cc_bbox_blur variants',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/final_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved results/final_comparison.png')

print('\n=== Winners ===')
for k, w in winners.items():
    print(
        f'{k:<12} {w["variant"]:<14} '
        f'mean={100*w["mean_acc"]:5.1f}%  '
        f'cD={100*w["mean_clean_delta"]:+5.1f}pp  '
        f'thr={w["threshold"]:.2f} dilate={w["dilate"]} bbox={w["bbox_snap"]}'
    )

print('\n=== Full table ===')
hdr = (f'{"cell":<28} {"thr":>5} {"dil":>3} {"bbox":>4} '
       f'{"mean":>7} {"cD":>7} {"cov":>6}')
print(hdr)
for k, row in comparison.items():
    print(
        f'{k:<28} {row["threshold"]:5.2f} {row["dilate"]:3d} '
        f'{str(row["bbox_snap"])!s:>4} '
        f'{100*row["mean_acc"]:6.1f}% '
        f'{100*row["mean_clean_delta"]:+6.1f} '
        f'{100*row["coverage"]:5.1f}%'
    )
