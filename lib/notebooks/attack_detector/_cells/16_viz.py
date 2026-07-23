class _ScaledModel:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model
    def predict_proba(self, X_):
        return self.model.predict_proba(self.scaler.transform(X_))
    def predict(self, X_):
        return self.model.predict(self.scaler.transform(X_))

def defend_images(images, cams_en, cams_l, gate_flags, threshold=DEFENSE_THR, label=''):
    t0 = time.time()
    out, covs, n_defended = [], [], 0
    for i, img in enumerate(images):
        if gate_flags is not None and not gate_flags[i]:
            out.append(img); covs.append(0.0); continue
        mask = build_cc_bbox_blur_mask(cams_en[i], cams_l[i], threshold=threshold)
        covs.append(float(mask.mean()))
        out.append(apply_mask(img, mask, fill='blur'))
        n_defended += 1
    print(f'  [{label}] defended={n_defended}/{len(images)}  '
          f'cov={100*np.mean(covs):.1f}%  {time.time()-t0:.1f}s')
    return out, float(np.mean(covs)), n_defended / len(images)

def score_pair(imgs, L, label=''):
    preds = {
        'en': classify_batch(models['en'], imgs, CLASSES['en']),
        L: classify_batch(models[L], imgs, CLASSES[L]),
    }
    acc = {ml: float((preds[ml] == true).mean()) for ml in ['en', L]}
    asr = {ml: float((preds[ml] == target).mean()) for ml in ['en', L]}
    print(f'  [{label}] acc EN={100*acc["en"]:.1f}% {L.upper()}={100*acc[L]:.1f}%  '
          f'ASR EN={100*asr["en"]:.1f}% {L.upper()}={100*asr[L]:.1f}%')
    return preds, acc, asr

def run_partner(L):
    out_dir = Path('results') / L / ATTACK
    out_dir.mkdir(parents=True, exist_ok=True)
    gated_path = out_dir / 'gated_comparison.json'
    if SKIP_EXISTING and gated_path.exists():
        print(f'\n===== SKIP {L} (existing {gated_path}) =====')
        with open(gated_path, encoding='utf-8') as f:
            return json.load(f)

    print(f'\n===== RUN EN&{L} / {ATTACK} =====')
    attacked = attacked_by_L[L]
    packed = load_or_bake_pair(L, attacked, out_dir)
    clean_cams, atk_cams = packed['clean'], packed['atk']
    print(f'Clean pred acc EN/{L}:',
          float((clean_cams['preds_en'] == true).mean()),
          float((clean_cams['preds_l'] == true).mean()))
    print(f'Atk   pred acc EN/{L}:',
          float((atk_cams['preds_en'] == true).mean()),
          float((atk_cams['preds_l'] == true).mean()))

    X, y, img_ids, feature_names = build_feature_matrix(clean_cams, atk_cams)
    print(f'Features: {X.shape}')

    rng_split = np.random.RandomState(SPLIT_SEED)
    perm = rng_split.permutation(n_images)
    n_train = int(round(TRAIN_FRAC * n_images))
    n_val = int(round(VAL_FRAC * n_images))
    train_imgs = set(perm[:n_train].tolist())
    val_imgs = set(perm[n_train:n_train + n_val].tolist())
    test_imgs = set(perm[n_train + n_val:].tolist())
    train_m = np.array([i in train_imgs for i in img_ids])
    val_m = np.array([i in val_imgs for i in img_ids])
    test_m = np.array([i in test_imgs for i in img_ids])

    # --- Phase A ---
    scaler_viz = StandardScaler().fit(X[train_m])
    Xz = scaler_viz.transform(X)
    pca = PCA(n_components=2, random_state=SPLIT_SEED)
    Xp = pca.fit_transform(Xz)
    c0 = Xp[train_m & (y == 0)].mean(0)
    c1 = Xp[train_m & (y == 1)].mean(0)
    pca_pred = ((Xp - c0) ** 2).sum(1) > ((Xp - c1) ** 2).sum(1)
    pca_acc_test = float((pca_pred[test_m] == y[test_m]).mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, name, color in [(0, 'clean', '#4C78A8'), (1, 'attacked', '#E45756')]:
        m = y == lab
        ax.scatter(Xp[m, 0], Xp[m, 1], s=8, alpha=0.45, c=color, label=name, edgecolors='none')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'PCA EN&{L} (test NN-cent={100*pca_acc_test:.1f}%)')
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(out_dir / 'pca_features.png', dpi=150); plt.close(fig)

    print('Running t-SNE...')
    Xt = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca',
              random_state=SPLIT_SEED).fit_transform(Xz)
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, name, color in [(0, 'clean', '#4C78A8'), (1, 'attacked', '#E45756')]:
        m = y == lab
        ax.scatter(Xt[m, 0], Xt[m, 1], s=8, alpha=0.45, c=color, label=name, edgecolors='none')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE EN&{L} heatmap features')
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(out_dir / 'tsne_features.png', dpi=150); plt.close(fig)

    phase_a = {
        'pca_explained_variance_sum': float(pca.explained_variance_ratio_.sum()),
        'pca_nearest_centroid_test_acc': pca_acc_test,
        'n_features': len(feature_names),
    }
    with open(out_dir / 'phase_a_summary.json', 'w', encoding='utf-8') as f:
        json.dump(phase_a, f, indent=2)

    # --- Phase B ---
    X_tr, y_tr = X[train_m], y[train_m]
    X_va, y_va = X[val_m], y[val_m]
    X_te, y_te = X[test_m], y[test_m]

    logit = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SPLIT_SEED)),
    ])
    logit.fit(X_tr, y_tr)

    _svm_scaler = StandardScaler().fit(X_tr)
    svm_inner = CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=5000, random_state=SPLIT_SEED), cv=3)
    svm_inner.fit(_svm_scaler.transform(X_tr), y_tr)
    svm = _ScaledModel(_svm_scaler, svm_inner)

    def _eval(model, X_, y_, name):
        proba = model.predict_proba(X_)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_, proba))
        acc = float(accuracy_score(y_, pred))
        p, r, f1, _ = precision_recall_fscore_support(y_, pred, average='binary', zero_division=0)
        print(f'  {name}: acc={100*acc:.1f}% AUC={auc:.3f} P={p:.3f} R={r:.3f}')
        return {'acc': acc, 'auc': auc, 'precision': float(p), 'recall': float(r),
                'f1': float(f1), 'proba': proba}

    logit_va, logit_te = _eval(logit, X_va, y_va, 'logit/val'), _eval(logit, X_te, y_te, 'logit/test')
    svm_va, svm_te = _eval(svm, X_va, y_va, 'svm/val'), _eval(svm, X_te, y_te, 'svm/test')
    if logit_va['auc'] >= svm_va['auc']:
        primary_name, primary, primary_va, primary_te = 'logistic', logit, logit_va, logit_te
    else:
        primary_name, primary, primary_va, primary_te = 'linear_svm', svm, svm_va, svm_te
    print(f'  Primary: {primary_name}')

    proba_va = primary_va['proba']
    _, tpr, thr_curve = roc_curve(y_va, proba_va)
    candidates = [(float(th), float(rec)) for th, rec in zip(thr_curve, tpr) if not np.isnan(th)]
    ok = [c for c in candidates if c[1] >= ATTACK_RECALL_TARGET]
    if ok:
        best_thr = max(ok, key=lambda c: c[0])[0]
    else:
        best_f1, best_thr = -1.0, 0.5
        for th in np.linspace(0.05, 0.95, 37):
            pred = (proba_va >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_va, pred, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(th)
    print(f'  threshold={best_thr:.4f}')

    def metrics_at_thr(proba, y_, thr):
        pred = (proba >= thr).astype(int)
        auc = float(roc_auc_score(y_, proba))
        acc = float(accuracy_score(y_, pred))
        p, r, f1, _ = precision_recall_fscore_support(y_, pred, average='binary', zero_division=0)
        return {
            'acc': acc, 'auc': auc, 'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'confusion_matrix': confusion_matrix(y_, pred).tolist(),
            'fire_rate_clean': float(pred[y_ == 0].mean()) if (y_ == 0).any() else 0.0,
            'fire_rate_attacked': float(pred[y_ == 1].mean()) if (y_ == 1).any() else 0.0,
            'threshold': thr,
        }

    te_at_thr = metrics_at_thr(primary_te['proba'], y_te, best_thr)
    va_at_thr = metrics_at_thr(proba_va, y_va, best_thr)
    print('  Test@thr:', {k: te_at_thr[k] for k in
          ('acc', 'auc', 'recall', 'fire_rate_clean', 'fire_rate_attacked')})

    weights = logit.named_steps['clf'].coef_.ravel()
    order = np.argsort(np.abs(weights))[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    top_k = min(20, len(feature_names))
    ax.barh(range(top_k), weights[order[:top_k]][::-1], color='#72B7B2')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in order[:top_k]][::-1], fontsize=8)
    ax.set_xlabel('Logistic coefficient')
    ax.set_title(f'Feature importance EN&{L}')
    fig.tight_layout(); fig.savefig(out_dir / 'feature_importance.png', dpi=150); plt.close(fig)

    all_proba = primary.predict_proba(X)[:, 1]
    all_pred = (all_proba >= best_thr).astype(int)
    gate_clean = all_pred[:n_images].astype(bool)
    gate_attacked = all_pred[n_images:].astype(bool)
    print(f'  Gate fire clean={100*gate_clean.mean():.1f}% attacked={100*gate_attacked.mean():.1f}%')

    detector_metrics = {
        'L': L, 'attack': ATTACK, 'primary': primary_name, 'threshold': best_thr,
        'attack_recall_target': ATTACK_RECALL_TARGET,
        'feature_names': feature_names,
        'logistic_val_auc_thr05': logit_va['auc'],
        'logistic_test_auc_thr05': logit_te['auc'],
        'svm_val_auc_thr05': svm_va['auc'],
        'svm_test_auc_thr05': svm_te['auc'],
        'val_at_threshold': {k: v for k, v in va_at_thr.items()},
        'test_at_threshold': {k: v for k, v in te_at_thr.items()},
        'logistic_top_weights': [
            {'feature': feature_names[i], 'weight': float(weights[i])}
            for i in order[:15].tolist()
        ],
        'phase_a_pca_nn_cent_acc': pca_acc_test,
    }
    with open(out_dir / 'detector_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(detector_metrics, f, indent=2)

    # --- Phase C ---
    policies = {}
    print('=== never_defend / attacked ===')
    _, base_acc, base_asr = score_pair(attacked, L, 'atk/never')
    policies['never_defend'] = {
        'attacked_acc': base_acc, 'attacked_asr': base_asr,
        'clean_acc': {ml: clean_acc[ml] for ml in ['en', L]},
        'clean_delta': {ml: 0.0 for ml in ['en', L]},
        'defend_frac_attacked': 0.0, 'defend_frac_clean': 0.0,
    }

    print('=== always_defend ===')
    always_flags = np.ones(n_images, dtype=bool)
    atk_always, cov_aa, frac_aa = defend_images(
        attacked, atk_cams['cams_en'], atk_cams['cams_l'], always_flags, label='atk/always')
    _, acc_aa, asr_aa = score_pair(atk_always, L, 'atk/always')
    cln_always, cov_ac, frac_ac = defend_images(
        clean_224, clean_cams['cams_en'], clean_cams['cams_l'], always_flags, label='cln/always')
    _, acc_ac, _ = score_pair(cln_always, L, 'cln/always')
    policies['always_defend'] = {
        'attacked_acc': acc_aa, 'attacked_asr': asr_aa,
        'clean_acc_masked': acc_ac,
        'clean_acc': {ml: clean_acc[ml] for ml in ['en', L]},
        'clean_delta': {ml: acc_ac[ml] - clean_acc[ml] for ml in ['en', L]},
        'defend_frac_attacked': frac_aa, 'defend_frac_clean': frac_ac,
        'coverage_attacked': cov_aa, 'coverage_clean': cov_ac,
    }

    print('=== gated ===')
    atk_gated, cov_ga, frac_ga = defend_images(
        attacked, atk_cams['cams_en'], atk_cams['cams_l'], gate_attacked, label='atk/gated')
    _, acc_ga, asr_ga = score_pair(atk_gated, L, 'atk/gated')
    cln_gated, cov_gc, frac_gc = defend_images(
        clean_224, clean_cams['cams_en'], clean_cams['cams_l'], gate_clean, label='cln/gated')
    _, acc_gc, _ = score_pair(cln_gated, L, 'cln/gated')
    policies['gated'] = {
        'attacked_acc': acc_ga, 'attacked_asr': asr_ga,
        'clean_acc_masked': acc_gc,
        'clean_acc': {ml: clean_acc[ml] for ml in ['en', L]},
        'clean_delta': {ml: acc_gc[ml] - clean_acc[ml] for ml in ['en', L]},
        'defend_frac_attacked': frac_ga, 'defend_frac_clean': frac_gc,
        'coverage_attacked': cov_ga, 'coverage_clean': cov_gc,
    }

    def _mean_acc(d):
        return 0.5 * (d['en'] + d[L])
    def _mean_delta(d):
        return 0.5 * (d['en'] + d[L])

    always_mean_acc = _mean_acc(policies['always_defend']['attacked_acc'])
    gated_mean_acc = _mean_acc(policies['gated']['attacked_acc'])
    always_mean_delta = _mean_delta(policies['always_defend']['clean_delta'])
    gated_mean_delta = _mean_delta(policies['gated']['clean_delta'])
    success = {
        'clean_delta_improved_pp': 100 * (gated_mean_delta - always_mean_delta),
        'attacked_acc_drop_pp': 100 * (always_mean_acc - gated_mean_acc),
        'meets_success_bar': (
            (gated_mean_delta - always_mean_delta) >= 0.01
            and (always_mean_acc - gated_mean_acc) <= 0.01
        ),
    }
    print('Success check:', success)

    gated_comparison = {
        'L': L, 'attack': ATTACK, 'defense_threshold': DEFENSE_THR,
        'detector': {
            'primary': primary_name, 'threshold': best_thr,
            'test_auc': te_at_thr['auc'], 'test_recall': te_at_thr['recall'],
            'test_fire_rate_clean': te_at_thr['fire_rate_clean'],
            'test_fire_rate_attacked': te_at_thr['fire_rate_attacked'],
        },
        'policies': policies,
        'summary': {
            'always_mean_attacked_acc': always_mean_acc,
            'gated_mean_attacked_acc': gated_mean_acc,
            'always_mean_clean_delta': always_mean_delta,
            'gated_mean_clean_delta': gated_mean_delta,
            'never_mean_attacked_acc': _mean_acc(base_acc),
        },
        'success': success,
    }
    with open(gated_path, 'w', encoding='utf-8') as f:
        json.dump(gated_comparison, f, indent=2)

    # summary plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ['never', 'always', 'gated']
    atk_means = [
        100 * gated_comparison['summary']['never_mean_attacked_acc'],
        100 * always_mean_acc, 100 * gated_mean_acc,
    ]
    colors = ['#E45756', '#4C78A8', '#54A24B']
    axes[0].bar(names, atk_means, color=colors)
    axes[0].set_ylabel('Mean acc EN+L (%)'); axes[0].set_title('Attacked accuracy')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(atk_means):
        axes[0].text(i, v + 1.5, f'{v:.1f}', ha='center', fontsize=9)
    deltas = [0.0, 100 * always_mean_delta, 100 * gated_mean_delta]
    axes[1].bar(names, deltas, color=colors)
    axes[1].axhline(0, color='gray', lw=0.8)
    axes[1].set_ylabel('Mean Clean delta (pp)'); axes[1].set_title('Clean-image degradation')
    for i, v in enumerate(deltas):
        axes[1].text(i, v - 0.4 if v < 0 else v + 0.3, f'{v:.2f}', ha='center', fontsize=9)
    fig.suptitle(
        f'Gated cc_bbox_blur (EN&{L} / {ATTACK}) | {primary_name} AUC={te_at_thr["auc"]:.3f}',
        fontsize=11)
    fig.tight_layout(); fig.savefig(out_dir / 'gated_comparison.png', dpi=150); plt.close(fig)
    print(f'Saved results under {out_dir}')
    return gated_comparison

print('run_partner ready.')
