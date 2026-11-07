# stacking — multi-event GW profile stack + Fisher forecast

Sums the per-event profile ΔlnL(γ) curves (common γ) from
`gw-pipeline/` runs. This is a **summed profile likelihood, NOT a
stacked posterior** — do not describe it as one.

## Files

- `stack_profile.py` — builds the stacked profile from the
  per-event `profile_gamma.json` files.
- `stack_data.py` — per-event profile data (GW250114, GW231123,
  GW150914, GW200129, GW190412).
- `fisher_stacking.py` — Fisher forecast for reaching γ ≈ 0.009.

## Results

| Stack | ΔlnL(0.4) | Wilks 95% γ < |
|---|---|---|
| 4 events (v5) | −57.8 | 0.144 |
| 5 events (v6, + GW190412) | −52.4 | 0.173 |
| 5 events, without GW231123 | −65.4 | 0.111 |

γ ≈ 0.009 is untouched by the stack (stack at γ = 0.1 is
+2.0 / +3.6). Corrected forecast: **~20 events at SNR ~80 for
1σ on γ ≈ 0.009, ~200 for 3σ** (four estimators from real
GW250114 data; the earlier "13–16" was the most optimistic
estimator at 1σ for γ = 0.01, not 0.00903).

Per-event pattern by mass is non-monotone (36.8 M☉: +5.4;
63.5: −15.7; 64.5: +0.1; 65.8: −55.2; 238: +13.0) — no γ(M)
survives all five events; it reads as five template misfits
until a spin-marginalised PE says otherwise.
