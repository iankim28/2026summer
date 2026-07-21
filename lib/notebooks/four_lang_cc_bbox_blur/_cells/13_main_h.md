## 6. Main loop — tune n=100 → full n=1000

For each `(L, attack)`: build attack → baselines → tune thr on EN masked acc
(n=100) → freeze best thr → defended eval on full n=1000 + clean degradation.
