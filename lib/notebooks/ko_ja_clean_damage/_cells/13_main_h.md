## 6. Main loop — variants on cached CAMs

For each `(L, attack)`: build attack → baselines → cache Attn-last cams on
tune + full (attacked and clean) → evaluate five mask variants → pick winner
(best Clean Δ among variants within 3pp mean-acc of baseline).
