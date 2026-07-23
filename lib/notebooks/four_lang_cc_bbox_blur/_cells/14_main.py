tune_best_cfg = {}
comparison = {}

for L in PARTNER_LANGS:
    for attack in ATTACKS:
        cell_key = f'{L}/{attack}'
        print(f'\n========== {cell_key} ==========')
        out_dir = Path('results') / L / attack
        out_dir.mkdir(parents=True, exist_ok=True)

        attacked = build_attack(attack, L)
        score_langs = ['en', L]

        preds_atk = {ml: classify_batch(models[ml], attacked, CLASSES[ml]) for ml in score_langs}
        baseline_acc = {ml: float((preds_atk[ml] == true).mean()) for ml in score_langs}
        baseline_asr = {ml: float((preds_atk[ml] == target).mean()) for ml in score_langs}
        print('  attacked acc:', {k: f'{100*v:.1f}%' for k, v in baseline_acc.items()})
        print('  attacked ASR:', {k: f'{100*v:.1f}%' for k, v in baseline_asr.items()})

        # --- tune on n=100 ---
        best_acc, best_thr, best_cov = -1.0, 0.95, 0.0
        for thr in THRESHOLDS:
            res = run_defense(L, attacked, tune_idx, threshold=thr, label=f'tune thr={thr}')
            en_acc = float((res['preds']['en'] == res['true']).mean())
            l_acc = float((res['preds'][L] == res['true']).mean())
            print(f'    thr={thr} EN={100*en_acc:.1f}% {L.upper()}={100*l_acc:.1f}%')
            if en_acc > best_acc:
                best_acc, best_thr, best_cov = en_acc, thr, res['coverage']

        free_thr = best_thr
        best_thr = max(free_thr, 0.95)
        if best_thr != free_thr:
            # Recompute tune coverage at the floored thr for logging.
            floor_res = run_defense(
                L, attacked, tune_idx, threshold=best_thr, label=f'tune thr={best_thr} (floor)')
            best_cov = floor_res['coverage']
            best_acc = float((floor_res['preds']['en'] == floor_res['true']).mean())

        tune_best_cfg[cell_key] = {
            'threshold': best_thr,
            'threshold_free': free_thr,
            'tune_en_acc': best_acc,
            'tune_cov': best_cov,
            'L': L,
            'attack': attack,
        }
        print(
            f'  BEST thr={best_thr} (free={free_thr}) '
            f'tune EN={100*best_acc:.1f}% cov={100*best_cov:.1f}%'
        )

        # --- full n=1000 ---
        atk = run_defense(L, attacked, all_idx, threshold=best_thr, label='full-atk')
        cln = run_defense(L, clean_224, all_idx, threshold=best_thr, label='full-clean')
        defense = metrics_for_pair(atk['preds'], true, target, L, baseline_acc, baseline_asr)
        clean_deg = {
            ml: {
                'baseline_acc': clean_acc[ml],
                'masked_acc': float((cln['preds'][ml] == true).mean()),
                'delta_acc': float((cln['preds'][ml] == true).mean()) - clean_acc[ml],
            } for ml in score_langs
        }
        row = {
            'L': L,
            'attack': attack,
            'threshold': best_thr,
            'threshold_free': free_thr,
            'ran_full': True,
            'clean_acc': {ml: clean_acc[ml] for ml in score_langs},
            'baseline_acc': baseline_acc,
            'baseline_asr': baseline_asr,
            'defense': defense,
            'clean_deg': clean_deg,
            'coverage': atk['coverage'],
            'mean_acc': defense['mean_acc'],
            'mean_clean_delta': 0.5 * (
                clean_deg['en']['delta_acc'] + clean_deg[L]['delta_acc']),
            'cost': 4,
        }
        print(
            f'  FULL mean acc={100*defense["mean_acc"]:.1f}%  '
            f'clean d={100*row["mean_clean_delta"]:+.1f}pp  '
            f'cov={100*atk["coverage"]:.1f}%'
        )

        with open(out_dir / 'confusion_results.json', 'w', encoding='utf-8') as f:
            json.dump(row, f, indent=2)
        comparison[cell_key] = row

with open('results/tune_best_cfg.json', 'w', encoding='utf-8') as f:
    json.dump(tune_best_cfg, f, indent=2)
with open('results/comparison_summary.json', 'w', encoding='utf-8') as f:
    json.dump(comparison, f, indent=2)
print('\nSaved results/tune_best_cfg.json and results/comparison_summary.json')
