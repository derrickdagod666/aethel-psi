# act-dr6 — real ACT DR6.02 CMB ripple test

Runs the paper's Prediction 7 (log-oscillation ripple at ℓ₀ ≈ 2200)
against **real** ACT DR6.02 TT+TE+EE bandpowers with the full
covariance — no synthetic data.

## Install (heavy; separate venv recommended)

```bash
python -m venv .venv-act && source .venv-act/bin/activate
pip install cobaya==3.6.2 camb==2.0.5 sacc scipy numpy
cobaya-install act_dr6_mflike   # downloads dr6_data.fits (GBs); first attempt
                                 # in our run failed on an IncompleteRead and
                                 # succeeded after manually placing the file
```

Data file used: `dr6_data_cmbonly.fits` (foreground-marginalized
CMB-only, 135 pts) from LAMBDA. **Not included in the repo.**

## Run

```bash
python a5_real.py    # TT-only: 18 bins, 1500 <= l <= 3000
python a5_joint.py   # joint TT+TE+EE, full 54x54 covariance, 20,000 null sims
python camb_joint_refit.py   # joint LCDM refit with CAMB
```

## Results (real data)

- TT-only: T_max = 5.88 at ω̂ = 9.75, Â = 0.00703, global p = 0.241,
  A95 = 1.60 × 10⁻².
- Joint TT+TE+EE: T_max = 7.95 at ω̂ = 11.00, Â = 0.00779 ± 0.0025,
  global p = 0.105, A95 = 1.33 × 10⁻². **No ripple.** A ≈ 0.1
  excluded; A ≈ 0.005 not excluded. CAMB joint refit Δχ² = 2.84,
  p ≈ 0.24.
- **Trap — do not quote the diagonal-only analysis**: on the same
  real data it gives p = 0.004, a false positive that vanishes
  completely with the real covariance (real TT-block lag-1
  correlation 0.46). It is documented here so nobody repeats it.

## Limitations

CosmoRec is not installed, so no MCMC was attempted. Per-bin real
errors are ~1–4%, not the 0.25% the synthetic v3 test assumed —
that is why the real limit is ~12× weaker than the synthetic one.
