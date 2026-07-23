all_results = {}
for L in PARTNER_LANGS:
    all_results[L] = run_partner(L)

# Roll-up summary
rows = []
for L, g in all_results.items():
    s = g['summary']
    d = g['detector']
    suc = g['success']
    rows.append({
        'L': L,
        'attack': ATTACK,
        'detector_primary': d['primary'],
        'test_auc': d['test_auc'],
        'test_fire_clean': d['test_fire_rate_clean'],
        'test_fire_attacked': d['test_fire_rate_attacked'],
        'never_mean_atk_acc': s['never_mean_attacked_acc'],
        'always_mean_atk_acc': s['always_mean_attacked_acc'],
        'gated_mean_atk_acc': s['gated_mean_attacked_acc'],
        'always_mean_clean_delta': s['always_mean_clean_delta'],
        'gated_mean_clean_delta': s['gated_mean_clean_delta'],
        'atk_acc_drop_pp': suc['attacked_acc_drop_pp'],
        'clean_delta_improved_pp': suc['clean_delta_improved_pp'],
        'meets_success_bar': suc['meets_success_bar'],
    })

comparison_summary = {
    'attack': ATTACK,
    'defense_threshold': DEFENSE_THR,
    'partners': rows,
}
with open('results/comparison_summary.json', 'w', encoding='utf-8') as f:
    json.dump(comparison_summary, f, indent=2)
print('Saved results/comparison_summary.json')

print('\n=== ROLL-UP ===')
print(f'{"L":<4} {"AUC":>6} {"fireC":>6} {"fireA":>6} {"always":>7} {"gated":>7} '
      f'{"dAtk":>6} {"aΔ":>7} {"gΔ":>7} {"ok":>4}')
for r in rows:
    print(f'{r["L"]:<4} {r["test_auc"]:6.3f} {100*r["test_fire_clean"]:5.1f}% '
          f'{100*r["test_fire_attacked"]:5.1f}% '
          f'{100*r["always_mean_atk_acc"]:6.1f}% {100*r["gated_mean_atk_acc"]:6.1f}% '
          f'{r["atk_acc_drop_pp"]:5.2f} '
          f'{100*r["always_mean_clean_delta"]:6.2f} {100*r["gated_mean_clean_delta"]:6.2f} '
          f'{str(r["meets_success_bar"]):>4}')
