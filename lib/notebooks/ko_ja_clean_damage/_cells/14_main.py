def _eval_row(L, attack, variant, mask_cfg, baseline_acc, baseline_asr,
              atk_full, cln_full):
    defense = metrics_for_pair(
        atk_full['preds'], atk_full['true'], target[all_idx], L,
        baseline_acc, baseline_asr)
    clean_deg, mean_clean_delta = clean_delta_from_preds(
        cln_full['preds'], cln_full['true'], L)
    return {
        'L': L,
        'attack': attack,
        'variant': variant,
        'threshold': mask_cfg['threshold'],
        'dilate': mask_cfg.get('dilate', 3),
        'bbox_snap': mask_cfg.get('bbox_snap', True),
        'ran_full': True,
        'clean_acc': {ml: clean_acc[ml] for ml in ['en', L]},
        'baseline_acc': baseline_acc,
        'baseline_asr': baseline_asr,
        'defense': defense,
        'clean_deg': clean_deg,
        'coverage': atk_full['coverage'],
        'mean_acc': defense['mean_acc'],
        'mean_clean_delta': mean_clean_delta,
        'cost': 4,
        'mask_cfg': mask_cfg,
    }


def _pick_winner(variant_rows, baseline_mean_acc):
    """Prefer best (closest to 0) clean Δ among rows within 3pp of baseline mean_acc."""
    eligible = [
        r for r in variant_rows
        if r['mean_acc'] >= baseline_mean_acc - 0.03
    ]
    pool = eligible if eligible else variant_rows
    return max(pool, key=lambda r: r['mean_clean_delta'])


def _tune_baseline_thr(L, attacked, cams_atk_tune):
    best_acc, best_thr, best_cov = -1.0, 0.95, 0.0
    for thr in THRESHOLDS:
        cfg = {'threshold': thr, 'dilate': 3, 'bbox_snap': True}
        res = run_defense_cached(
            L, attacked, tune_idx, cams_atk_tune, cfg, label=f'base thr={thr}')
        en_acc = float((res['preds']['en'] == res['true']).mean())
        l_acc = float((res['preds'][L] == res['true']).mean())
        print(f'    thr={thr} EN={100*en_acc:.1f}% {L.upper()}={100*l_acc:.1f}%')
        if en_acc > best_acc:
            best_acc, best_thr, best_cov = en_acc, thr, res['coverage']
    return best_thr, best_acc, best_cov


def _tune_pareto_thr(L, attacked, cams_atk_tune, cams_cln_tune):
    best_score, best_thr = -1e9, 0.95
    for thr in PARETO_THRESHOLDS:
        cfg = {'threshold': thr, 'dilate': 3, 'bbox_snap': True}
        atk = run_defense_cached(
            L, attacked, tune_idx, cams_atk_tune, cfg, label=f'pareto-atk thr={thr}')
        cln = run_defense_cached(
            L, clean_224, tune_idx, cams_cln_tune, cfg, label=f'pareto-cln thr={thr}')
        en_atk = float((atk['preds']['en'] == atk['true']).mean())
        _, mean_cd = clean_delta_from_preds(cln['preds'], cln['true'], L)
        score = en_atk + 0.5 * mean_cd
        print(f'    pareto thr={thr} en_atk={100*en_atk:.1f}% '
              f'cD={100*mean_cd:+.1f}pp score={score:.3f}')
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr, best_score


tune_best_cfg = {}
comparison = {}
winners = {}

for L in PARTNER_LANGS:
    for attack in ATTACKS:
        cell_key = f'{L}/{attack}'
        print(f'\n========== {cell_key} ==========')
        out_dir = Path('results') / L / attack
        out_dir.mkdir(parents=True, exist_ok=True)

        attacked = build_attack(attack, L)
        score_langs = ['en', L]

        preds_atk = {
            ml: classify_batch(models[ml], attacked, CLASSES[ml])
            for ml in score_langs
        }
        baseline_acc = {ml: float((preds_atk[ml] == true).mean()) for ml in score_langs}
        baseline_asr = {ml: float((preds_atk[ml] == target).mean()) for ml in score_langs}
        print('  attacked acc:', {k: f'{100*v:.1f}%' for k, v in baseline_acc.items()})
        print('  attacked ASR:', {k: f'{100*v:.1f}%' for k, v in baseline_asr.items()})

        # --- cache CAMs once (tune + full, attacked + clean) ---
        cams_atk_tune = precompute_cams(L, attacked, tune_idx, label='atk-tune')
        cams_cln_tune = precompute_cams(L, clean_224, tune_idx, label='cln-tune')
        cams_atk_full = precompute_cams(L, attacked, all_idx, label='atk-full')
        cams_cln_full = precompute_cams(L, clean_224, all_idx, label='cln-full')

        base_thr, base_tune_en, base_tune_cov = _tune_baseline_thr(
            L, attacked, cams_atk_tune)
        pareto_thr, pareto_score = _tune_pareto_thr(
            L, attacked, cams_atk_tune, cams_cln_tune)

        variant_cfgs = {
            'baseline': {
                'threshold': base_thr, 'dilate': 3, 'bbox_snap': True,
            },
            'thr_floor_095': {
                'threshold': 0.95, 'dilate': 3, 'bbox_snap': True,
            },
            'pareto_tune': {
                'threshold': pareto_thr, 'dilate': 3, 'bbox_snap': True,
            },
            'tight_dilate': {
                'threshold': pareto_thr, 'dilate': 1, 'bbox_snap': True,
            },
            'no_bbox': {
                'threshold': pareto_thr, 'dilate': 3, 'bbox_snap': False,
            },
        }

        # If thr_floor still leaves cov > 12% on tune attacked, enable max_coverage shrink.
        floor_cfg = dict(variant_cfgs['thr_floor_095'])
        floor_tune = run_defense_cached(
            L, attacked, tune_idx, cams_atk_tune, floor_cfg, label='floor-cov-check')
        if floor_tune['coverage'] > 0.12:
            for name in ('thr_floor_095', 'pareto_tune', 'tight_dilate', 'no_bbox'):
                variant_cfgs[name] = dict(variant_cfgs[name])
                variant_cfgs[name]['max_coverage'] = 0.12
            print('  enabled max_coverage=0.12 (floor tune cov '
                  f'{100*floor_tune["coverage"]:.1f}%)')

        tune_best_cfg[cell_key] = {
            'baseline_threshold': base_thr,
            'baseline_tune_en_acc': base_tune_en,
            'baseline_tune_cov': base_tune_cov,
            'pareto_threshold': pareto_thr,
            'pareto_score': pareto_score,
            'L': L,
            'attack': attack,
            'variant_cfgs': variant_cfgs,
        }
        print(f'  baseline thr={base_thr}  pareto thr={pareto_thr}')

        cell_rows = []
        for variant in VARIANTS:
            cfg = variant_cfgs[variant]
            print(f'  --- full {variant} ---')
            atk = run_defense_cached(
                L, attacked, all_idx, cams_atk_full, cfg, label=f'full-atk {variant}')
            cln = run_defense_cached(
                L, clean_224, all_idx, cams_cln_full, cfg, label=f'full-cln {variant}')
            row = _eval_row(
                L, attack, variant, cfg, baseline_acc, baseline_asr, atk, cln)
            print(
                f'  {variant}: mean={100*row["mean_acc"]:.1f}%  '
                f'cD={100*row["mean_clean_delta"]:+.1f}pp  '
                f'cov={100*row["coverage"]:.1f}%'
            )
            with open(out_dir / f'{variant}.json', 'w', encoding='utf-8') as f:
                json.dump(row, f, indent=2)
            comparison[f'{cell_key}/{variant}'] = row
            cell_rows.append(row)

        baseline_row = next(r for r in cell_rows if r['variant'] == 'baseline')
        winner = _pick_winner(cell_rows, baseline_row['mean_acc'])
        winners[cell_key] = {
            'variant': winner['variant'],
            'mean_acc': winner['mean_acc'],
            'mean_clean_delta': winner['mean_clean_delta'],
            'threshold': winner['threshold'],
            'dilate': winner['dilate'],
            'bbox_snap': winner['bbox_snap'],
        }
        print(f'  WINNER {cell_key}: {winner["variant"]}  '
              f'mean={100*winner["mean_acc"]:.1f}%  '
              f'cD={100*winner["mean_clean_delta"]:+.1f}pp')

with open('results/tune_best_cfg.json', 'w', encoding='utf-8') as f:
    json.dump(tune_best_cfg, f, indent=2)
with open('results/comparison_summary.json', 'w', encoding='utf-8') as f:
    json.dump(comparison, f, indent=2)
with open('results/winners.json', 'w', encoding='utf-8') as f:
    json.dump(winners, f, indent=2)
print('\nSaved results/tune_best_cfg.json, comparison_summary.json, winners.json')
