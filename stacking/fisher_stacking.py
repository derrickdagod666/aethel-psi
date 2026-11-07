"""6a Fisher stacking forecast — RECOMPUTED from the real GW250114 data,
not copied from GAMMA_OF_MASS.md.

Four independent estimators of sigma_gamma for ONE GW250114-like event
(SNR ~ 80), all derived from the real-data products in
workspace/gate3-gw250114/:
  E1 mismatch scaling: mm(gamma) = 0.0094*(gamma/0.4)^2 (GATE3_REPORT §4.3,
     PSD-weighted mismatch measured on real data). 1-sigma SNR needed
     = 1/sqrt(2*mm). Stacking N identical events: SNR_eff = SNR*sqrt(N).
  E2 profile curvature: refit Delta lnL = -k*gamma^2 to the real profile
     curve in profile_gamma.json (FULL_PE_REPORT §4 quoted k~345).
     sigma = 1/sqrt(2*k).
  E3 Fisher projection from V3_FIXES.md §1.5: projected sigma_gamma=0.0533
     (toy waveform, cross-checked vs real 95% limit within 50%).
  E4 marginalised PE posterior (full-pe/posterior_uaf.csv, 3697 samples):
     sigma from the 90%/95% upper limits assuming an approximate
     half-normal shape, plus the raw posterior std for reference.

Stacking arithmetic (common gamma — justified ONLY by the flat gamma(M)
= 0.00903 derivation; if gamma varied with mass, stacking a common gamma
would be wrong and the per-event limits are the result instead):
  N_1sigma = (sigma_single / gamma_target)^2
  N_3sigma = 9 * N_1sigma   (3-sigma measurement, not just 1-sigma reach)
Target: gamma = 0.00903 (Eq. 11R derived value); also tabulated at 0.01
(the value the earlier "13-16 events" claim used) so the discrepancy
with the earlier claim is explicit.
"""
import json
import numpy as np
import pandas as pd

GAMMA_TARGET = 0.00903
MISMATCH_04 = 0.0094  # GW250114 real-data PSD-weighted mismatch at gamma=0.4
SNR_250114 = 78.2     # phase-maximised matched-filter SNR, real data

out = {}

# --- E1: mismatch scaling ---
# mm(gamma) = mm_ref*(gamma/0.4)^2; SNR_needed(1sig) = 1/sqrt(2 mm)
for g in (GAMMA_TARGET, 0.01):
    mm = MISMATCH_04 * (g / 0.4) ** 2
    snr_needed = 1.0 / np.sqrt(2 * mm)
    n80 = (snr_needed / SNR_250114) ** 2
    out[f"E1_mismatch_gamma_{g}"] = {
        "mismatch": float(mm), "SNR_needed_1sigma": float(snr_needed),
        "N_events_at_SNR78.2_1sigma": float(n80),
        "N_events_at_SNR78.2_3sigma": float(9 * n80),
        "total_SNR2_needed_1sigma": float(snr_needed ** 2),
        "total_SNR2_needed_3sigma": float(9 * snr_needed ** 2)}
    print(f"E1 gamma={g}: mismatch={mm:.2e}, SNR_needed(1sig)={snr_needed:.0f}, "
          f"N@SNR78.2: 1sig={n80:.1f}, 3sig={9*n80:.0f}")

# sigma_gamma implied by E1 at SNR 78.2: gamma where SNR_needed == 78.2
sigma_e1 = 0.4 * np.sqrt((1.0 / (2 * SNR_250114 ** 2)) / MISMATCH_04)
print(f"E1 implied sigma_gamma at SNR 78.2: {sigma_e1:.4f}")

# --- E2: refit profile curvature from profile_gamma.json ---
prof = json.load(open("/home/hatch/workspace/gate3-gw250114/profile_gamma.json"))
curve = prof["gamma_curve"]
gs = np.array([float(k) for k in curve.keys()])
dl = np.array([curve[k]["lnL_ratio_max_phase"] for k in curve.keys()])
dl = dl - dl[0]  # Delta vs gamma=0
# Fit Delta = -k * gamma^2 through the small-gamma points (gamma <= 0.4),
# least squares on -dl/gamma^2
mask = (gs > 0) & (gs <= 0.4)
k_fit = float(np.sum(gs[mask] ** 2 * (-dl[mask])) / np.sum(gs[mask] ** 4))
sigma_e2 = 1.0 / np.sqrt(2 * k_fit)
print(f"E2 profile curvature: k={k_fit:.1f} (FULL_PE quoted ~345), "
      f"sigma_gamma={sigma_e2:.4f}")
print(f"   profile Delta lnL: " +
      ", ".join(f"{g:.1f}:{d:+.1f}" for g, d in zip(gs, dl)))
out["E2_profile_curvature"] = {"k": k_fit, "sigma_gamma": float(sigma_e2)}

# --- E3: Fisher projection (V3) ---
sigma_e3 = 0.0533
print(f"E3 Fisher projected sigma_gamma=0.0533 (V3_FIXES §1.5)")

# --- E4: PE posterior ---
post = pd.read_csv("/home/hatch/workspace/gate3-gw250114/full-pe/posterior_uaf.csv")
g = post["gamma"].values
q = np.quantile(g, [0.5, 0.84, 0.90, 0.95, 0.99])
sigma_e4_95 = q[3] / 1.96   # half-normal: 95th pct = 1.96 sigma
sigma_e4_90 = q[2] / 1.6449
sigma_e4_raw = float(np.std(g))
print(f"E4 PE posterior: median={q[0]:.4f}, 84%={q[1]:.4f}, 90%={q[2]:.4f}, "
      f"95%={q[3]:.4f}, 99%={q[4]:.4f}, raw std={sigma_e4_raw:.4f}, "
      f"sigma from 95%={sigma_e4_95:.4f}, from 90%={sigma_e4_90:.4f}")
out["E4_PE_posterior"] = {"median": float(q[0]), "q84": float(q[1]),
                          "q90": float(q[2]), "q95": float(q[3]),
                          "q99": float(q[4]), "raw_std": sigma_e4_raw,
                          "sigma_from_95pct": float(sigma_e4_95),
                          "sigma_from_90pct": float(sigma_e4_90)}

# --- Stacking table: N events needed per estimator ---
sigmas = {"E1_mismatch": sigma_e1, "E2_profile_curvature": sigma_e2,
          "E3_Fisher_projected": sigma_e3,
          "E4_PE_posterior_95pct": sigma_e4_95}
print("\nN events at SNR~78 (GW250114-like) to reach gamma=0.00903:")
table = {}
for name, s in sigmas.items():
    n1 = (s / GAMMA_TARGET) ** 2
    n1_001 = (s / 0.01) ** 2
    table[name] = {"sigma_single": float(s),
                   "N_1sigma_gamma0.00903": float(n1),
                   "N_3sigma_gamma0.00903": float(9 * n1),
                   "N_1sigma_gamma0.01": float(n1_001),
                   "N_3sigma_gamma0.01": float(9 * n1_001)}
    print(f"  {name}: sigma={s:.4f} -> N(1sig, 0.00903)={n1:.1f}, "
          f"N(3sig)={9*n1:.0f}; N(1sig, 0.01)={n1_001:.1f}")
out["stacking_table_SNR78"] = table

# --- Same at other single-event SNRs (sigma scales as 1/SNR) ---
print("\nN events to reach gamma=0.00903 at 1 sigma, by per-event SNR "
      "(using E4 sigma=0.040 at SNR 78.2 as the central estimator):")
for snr in (80, 50, 30, 22.6, 15):
    s = sigma_e4_95 * SNR_250114 / snr
    n1 = (s / GAMMA_TARGET) ** 2
    print(f"  SNR {snr}: sigma_single={s:.4f}, N(1sig)={n1:.0f}, N(3sig)={9*n1:.0f}")

# --- SNR^2 budget vs what exists ---
print("\nSNR^2 budget: 1-sigma reach of gamma=0.00903 needs total "
      f"SNR^2 = {out['E1_mismatch_gamma_0.00903']['total_SNR2_needed_1sigma']:.0f} "
      f"(3-sigma: {out['E1_mismatch_gamma_0.00903']['total_SNR2_needed_3sigma']:.0f}).")
print(f"GW250114 alone contributes SNR^2 = {SNR_250114**2:.0f}. "
      f"GW231123 contributes 22.6^2 = {22.6**2:.0f}.")
json.dump(out, open("/home/hatch/workspace/gate3-gw250114/stacking/fisher_stacking.json", "w"), indent=2)
print("Wrote fisher_stacking.json")
