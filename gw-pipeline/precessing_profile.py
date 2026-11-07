"""Gate 3 — GW231123 profile likelihood with a PRECESSING approximant.

Pre-registered design (written BEFORE any XPHM likelihood was evaluated):
  - Same data / PSD / band / segment as the Pv2 profile
    (gate3_231123_profile.py): real GWOSC O4a strain, H1+L1, 8 s,
    30-256 Hz, Welch-median PSD, Tukey window. NOTHING about the data
    preparation changes; only the waveform approximant and the spin
    configuration change.
  - Approximant: IMRPhenomXPHM (precessing, higher modes; one of the
    five models LVK combined for GW231123). SEOBNRv5PHM is NOT available
    in this environment: LALSimulation 8.0.0 has the enum but no
    generator ("no generator defined for approximant 36"). NRSur7dq4
    fails at this band (fRef=20 < fMin=30, "fRef cannot be lesser than
    fMin") and was not used. No approximant was installed; this is what
    the venv provides.
  - Config A (isolates the approximant): XPHM, aligned spins a1=a2=0.31
    (chi_eff=0.31), tilts 0 — same fixed point as the Pv2 profile.
  - Config B (precessing): XPHM, published spin magnitudes a1=0.90,
    a2=0.80, tilt_1=tilt_2=1.0 rad, phi_12=0, phi_jl=0.
    chi_p at this point: max(a1*sin t1, a2*sin t2 * q-factor) ~ 0.76,
    matching published chi_p=0.77. Tilt 1.0 rad was chosen BEFORE
    seeing results as the representative precessing point, not fitted.
  - Config C (tilt sensitivity, gamma=0.4 only): XPHM, a1=0.90/a2=0.80,
    tilts 0.5 and 1.5 rad, each compared with GR at the SAME tilt.
  - dt grid: -0.20..+0.10 s in 0.005 s steps (the wide grid from the
    Pv2 dt diagnostic; best Pv2 dt was -0.055 s, interior). theta_jn
    maximised over {0.5, 1.0, 1.4, 2.0} at the GR point per config,
    then held for the gamma curve. Phase maximised analytically.
  - f_merger=35 Hz, width_frac=0.15 central; f_merger 25/45 Hz checks
    under config B only.
  - UAF waveform: same fixed code as gate3_231123_pe.py
    (width = width_frac * f_merger; f_merger explicit; floor 0.5),
    with waveform_approximant swapped Pv2 -> XPHM.

Interpretation: PROFILE likelihood at fixed nuisance parameters, NOT
a marginalised posterior and NOT a Bayes factor. The full PE
(precessing_pe.py, same directory) marginalises; this profile is the
fast check of whether the 35 Hz residual survives precession.
"""
import json
import os
import sys
import numpy as np
import bilby

# build_ifos() in gate3_231123_common.py loads data_H1/L1.npz by relative
# path, so run from the GW231123 directory; outputs go to precessing-pe/.
os.chdir("/home/hatch/workspace/gate3-gw231123")
OUTDIR = "/home/hatch/workspace/gate3-gw231123/precessing-pe"

sys.path.insert(0, "/home/hatch/workspace/gate3-gw231123")
from gate3_231123_common import build_ifos, TRIGGER, F_MIN, F_MAX, FS, DUR
from gate3_231123_pe import uaf_blunted_waveform

BASE = dict(mass_1=190.4, mass_2=143.2, luminosity_distance=2200.0,
            phase=0.0, psi=0.0, ra=3.02, dec=-0.02)
F_MERGER = 35.0
WIDTH_FRAC = 0.15
DTS = tuple(np.round(np.arange(-0.20, 0.1001, 0.005), 3))
GAMMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def cinner(a, b, fr, psd, mask):
    m = mask & (fr >= F_MIN) & (fr <= F_MAX)
    return 4.0 * np.sum(np.conj(a[m]) * b[m] / psd[m]) * (fr[1] - fr[0])


def max_phase_lnl_ratio(ifos, wg, params):
    pol = wg.frequency_domain_strain(params)
    z_tot = 0j
    hh_tot = 0.0
    for ifo in ifos:
        h = ifo.get_detector_response(pol, params)
        d = ifo.frequency_domain_strain
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        z_tot += cinner(d, h, fr, psd, mask)
        hh_tot += cinner(h, h, fr, psd, mask).real
    return abs(z_tot) - 0.5 * hh_tot, abs(z_tot) / np.sqrt(hh_tot), np.sqrt(hh_tot)


def run_config(ifos, wg, label, spin_kwargs):
    print(f"\n=== {label}: spins {spin_kwargs} ===", flush=True)
    base = dict(BASE, **spin_kwargs)
    # Step 1: best GR over theta_jn x dt
    best_gr = None
    for th in (0.5, 1.0, 1.4, 2.0):
        for dt in DTS:
            p = dict(base, gamma=0.0, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                     theta_jn=th, geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best_gr is None or val > best_gr[0]:
                best_gr = (val, th, dt, mf, opt)
    print(f"  Best GR (XPHM): lnL={best_gr[0]:.1f}, theta_jn={best_gr[1]}, "
          f"dt={best_gr[2]:+.3f}s, |MF|={best_gr[3]:.1f}, opt SNR={best_gr[4]:.1f}", flush=True)
    th_best, dt_best = best_gr[1], best_gr[2]
    # Step 2: gamma curve
    curve = {}
    for g in GAMMAS:
        best = None
        for dt in DTS:
            p = dict(base, gamma=g, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                     theta_jn=th_best, geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt, mf, opt)
        curve[g] = {"lnL_ratio_max_phase": float(best[0]),
                    "best_dt_s": float(best[1]),
                    "matched_filter_snr": float(best[2]),
                    "optimal_snr": float(best[3])}
        print(f"  gamma={g:.1f}: lnL={best[0]:.1f} "
              f"(Delta={best[0]-best_gr[0]:+.1f}, dt={best[1]:+.3f}s, "
              f"|MF|={best[2]:.1f}, opt SNR={best[3]:.1f})", flush=True)
    return {"label": label, "spin_kwargs": spin_kwargs,
            "lnL_ratio_GR_best": float(best_gr[0]),
            "theta_jn_best": th_best, "dt_best_s": float(dt_best),
            "gamma_curve": curve,
            "delta_lnL_vs_GR": {str(g): float(curve[g]["lnL_ratio_max_phase"] - best_gr[0])
                                for g in GAMMAS}}


def gamma_at(ifos, wg, base, th, g, fm=F_MERGER, wf=WIDTH_FRAC):
    """Best max-phase lnL over dt for one (gamma, f_merger, width) point."""
    best = None
    for dt in DTS:
        p = dict(base, gamma=g, f_merger=fm, width_frac=wf,
                 theta_jn=th, geocent_time=TRIGGER + dt)
        val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
        if best is None or val > best[0]:
            best = (val, dt)
    return best


def main():
    ifos = build_ifos()
    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                                reference_frequency=20.0,
                                minimum_frequency=F_MIN,
                                maximum_frequency=F_MAX, floor=0.5))
    out = {"approximant": "IMRPhenomXPHM",
           "method": "profile likelihood, fixed nuisance parameters, phase "
                     "maximised analytically, dt scanned -0.20..+0.10 s, "
                     "theta_jn maximised over {0.5,1.0,1.4,2.0} at GR point",
           "configs": {}}
    # Config A: aligned, isolates approximant change
    out["configs"]["A_aligned_XPHM"] = run_config(
        ifos, wg, "A: XPHM aligned a=0.31",
        dict(a_1=0.31, a_2=0.31, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0))
    # Config B: precessing, published spin magnitudes
    cfg_b = run_config(
        ifos, wg, "B: XPHM precessing a=0.90/0.80 tilt=1.0",
        dict(a_1=0.90, a_2=0.80, tilt_1=1.0, tilt_2=1.0, phi_12=0.0, phi_jl=0.0))
    out["configs"]["B_precessing_XPHM"] = cfg_b
    # Sensitivity under B: f_merger placement + tilt, gamma=0.4 vs GR same config
    base_b = dict(BASE, a_1=0.90, a_2=0.80, tilt_1=1.0, tilt_2=1.0,
                  phi_12=0.0, phi_jl=0.0)
    th_b = cfg_b["theta_jn_best"]
    l0_b = cfg_b["lnL_ratio_GR_best"]
    sens = {}
    for fm in (25.0, 45.0):
        v, dt = gamma_at(ifos, wg, base_b, th_b, 0.4, fm=fm)
        sens[f"f_merger={fm}"] = {"lnL_ratio_gamma0.4": float(v),
                                  "delta_vs_GR_same_config": float(v - l0_b),
                                  "best_dt_s": float(dt)}
        print(f"sensitivity B f_merger={fm}: Delta lnL vs GR = {v - l0_b:+.1f}", flush=True)
    for wf in (0.10, 0.25):
        v, dt = gamma_at(ifos, wg, base_b, th_b, 0.4, wf=wf)
        sens[f"width_frac={wf}"] = {"lnL_ratio_gamma0.4": float(v),
                                    "delta_vs_GR_same_config": float(v - l0_b),
                                    "best_dt_s": float(dt)}
        print(f"sensitivity B width_frac={wf}: Delta lnL vs GR = {v - l0_b:+.1f}", flush=True)
    # Config C: tilt sensitivity at gamma=0.4 (GR at same tilt for fair delta)
    for tilt in (0.5, 1.5):
        base_c = dict(BASE, a_1=0.90, a_2=0.80, tilt_1=tilt, tilt_2=tilt,
                      phi_12=0.0, phi_jl=0.0)
        # best GR over theta_jn x dt at this tilt
        best_gr_c = None
        for th in (0.5, 1.0, 1.4, 2.0):
            for dt in DTS:
                p = dict(base_c, gamma=0.0, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                         theta_jn=th, geocent_time=TRIGGER + dt)
                val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
                if best_gr_c is None or val > best_gr_c[0]:
                    best_gr_c = (val, th, dt, mf, opt)
        v, dt = gamma_at(ifos, wg, base_c, best_gr_c[1], 0.4)
        sens[f"tilt={tilt}"] = {"lnL_ratio_GR_same_tilt": float(best_gr_c[0]),
                                "theta_jn_best_GR": best_gr_c[1],
                                "lnL_ratio_gamma0.4": float(v),
                                "delta_vs_GR_same_tilt": float(v - best_gr_c[0]),
                                "best_dt_s": float(dt)}
        print(f"sensitivity C tilt={tilt}: GR lnL={best_gr_c[0]:.1f} "
              f"(theta_jn={best_gr_c[1]}), gamma=0.4 lnL={v:.1f}, "
              f"Delta={v - best_gr_c[0]:+.1f}", flush=True)
    out["sensitivity_B_and_C"] = sens
    json.dump(out, open(os.path.join(OUTDIR, "precessing_profile.json"), "w"), indent=2)
    print("Wrote precessing_profile.json", flush=True)


if __name__ == "__main__":
    main()
