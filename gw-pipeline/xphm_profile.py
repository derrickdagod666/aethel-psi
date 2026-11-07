"""Gate 3 — GW200129 profile likelihood with a PRECESSING approximant.

Pre-registered design (written in XPHM_PROFILE_REPORT.md §1 BEFORE any
XPHM likelihood was evaluated):
  - Same data / PSD / band / segment as the Pv2 stacking run
    (stack_profile.py): real GWOSC O3b strain, H1+L1, 8 s,
    20-400 Hz, Welch-median PSD, Tukey window. NOTHING about the data
    preparation changes; only the waveform approximant and the spin
    configuration change. Prepared segment/PSD reused from
    ~/workspace/gate3-gw250114/stacking/GW200129/data_H1.npz,
    data_L1.npz (md5-verified strain files, see report §1.3).
  - Approximant: IMRPhenomXPHM (precessing, higher modes; the model
    GWTC-3 used for this event alongside SEOBNRv4PHM).
  - f_merger = 143 Hz pre-registered (150 * 71.5 / 74.9 = 143.2 Hz,
    stacking value). width_frac = 0.15. Floor 0.5.
  - Sky fixed at the stacking run's coarse-grid GR maximum:
    ra=1.57 rad, dec=+0.80 rad (NOT re-maximised under XPHM).
  - Config A (control, isolates the approximant): XPHM, aligned spins
    a1=a2=0.11 (chi_eff=0.11), tilts 0 — same fixed point as Pv2.
  - Config B (precessing): XPHM, a1=0.90, a2=0.50, tilt_1=tilt_2=1.4 rad,
    phi_12=phi_jl=0 (chi_p ~ 0.89, chi_eff ~ 0.12; Hannam et al. 2022
    a1=0.9, tilt~1.4 rad; GWTC-3 chi_p=0.891, chi_eff=0.079).
  - dt grid: -0.20..+0.10 s in 0.005 s steps (wide grid, as GW231123
    precessing profile). theta_jn maximised over {0.5,1.0,1.4,2.0} at
    the GR point per config, then held for the gamma curve. Phase
    maximised analytically.
  - AMENDED (grid resolution only, before the headline run; see
    diag_dt.py / diag_dt.log): dt step 0.005 s -> 0.0005 s over the
    same -0.20..+0.10 s range. Reason, demonstrated not assumed: a
    first run on the 5 ms grid gave XPHM-A GR lnL=134.0, while a 1 ms
    diagnostic scan gives 294.4 at dt=+0.013 s (Pv2: 292.5 at
    dt=+0.020 s, reproduced exactly). The XPHM lnL peak is ~2-3 ms
    wide at f_merger=143 Hz (lnL 296.8 at +0.0125 s, 123.0 at +0.0150
    s), so a 5 ms step aliases it; the max-over-dt Pv2-XPHM template
    overlap is 0.9963, i.e. the approximants differ only in t_c
    convention, not shape. The 5 ms run is preserved as
    xphm_profile_coarse5ms.json (its numbers are NOT the headline).
    The amendment changes no physics parameter (f_merger, width,
    spins, sky, priors all unchanged) and is applied identically to
    GR and every gamma, both configs; it is a numerical-resolution
    fix, not a refit.
  - UAF waveform: same fixed code as gate3_231123_pe.uaf_blunted_waveform
    (width = width_frac * f_merger; f_merger explicit; floor 0.5),
    with waveform_approximant swapped Pv2 -> XPHM.
  - Sensitivities under B at gamma=0.4: f_merger 133/153 Hz,
    width_frac 0.10/0.25, f_min=30 Hz.

Interpretation: PROFILE likelihood at fixed nuisance parameters, NOT
a marginalised posterior and NOT a Bayes factor.
"""
import json
import os
import sys
import numpy as np
import bilby
from bilby.gw.detector import get_empty_interferometer, PowerSpectralDensity

sys.path.insert(0, "/home/hatch/workspace/gate3-gw231123")
from gate3_231123_pe import uaf_blunted_waveform  # fixed UAF waveform

STACKDIR = "/home/hatch/workspace/gate3-gw250114/stacking"
OUTDIR = "/home/hatch/workspace/gate3-gw200129/xphm"
TRIGGER = 1264316116.43
F_MIN, F_MAX = 20.0, 400.0
FS, DUR = 4096.0, 8.0
F_MERGER = 143.0
WIDTH_FRAC = 0.15
DTS = tuple(np.round(np.arange(-0.20, 0.1001, 0.0005), 4))  # 0.5 ms steps; see header note
GAMMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RA, DEC = 1.5707963267948966, 0.8  # stacking run's GR sky maximum (fixed)

BASE = dict(mass_1=40.7, mass_2=34.2, luminosity_distance=890.0,
            phase=0.0, psi=0.0, ra=RA, dec=DEC)


def build_ifos(f_min=F_MIN):
    ifos = []
    for det in ("H1", "L1"):
        d = np.load(os.path.join(STACKDIR, "GW200129", f"data_{det}.npz"))
        ifo = get_empty_interferometer(det)
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=d["psd_freqs"], psd_array=d["psd"])
        ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain=d["fd_strain"], sampling_frequency=FS,
            duration=DUR, start_time=float(d["seg_start"]))
        ifo.minimum_frequency = f_min
        ifo.maximum_frequency = F_MAX
        ifos.append(ifo)
    return bilby.gw.detector.InterferometerList(ifos)


def cinner(a, b, fr, psd, mask, f_min=F_MIN):
    m = mask & (fr >= f_min) & (fr <= F_MAX)
    return 4.0 * np.sum(np.conj(a[m]) * b[m] / psd[m]) * (fr[1] - fr[0])


def max_phase_lnl_ratio(ifos, wg, params, f_min=F_MIN):
    pol = wg.frequency_domain_strain(params)
    z_tot = 0j
    hh_tot = 0.0
    per_det = {}
    for ifo in ifos:
        h = ifo.get_detector_response(pol, params)
        d = ifo.frequency_domain_strain
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        z = cinner(d, h, fr, psd, mask, f_min)
        hh = cinner(h, h, fr, psd, mask, f_min).real
        z_tot += z
        hh_tot += hh
        per_det[ifo.name] = {"matched_filter_snr": float(abs(z) / np.sqrt(hh)) if hh > 0 else 0.0,
                             "optimal_snr": float(np.sqrt(hh))}
    return (abs(z_tot) - 0.5 * hh_tot, abs(z_tot) / np.sqrt(hh_tot),
            np.sqrt(hh_tot), per_det)


def run_config(ifos, wg, label, spin_kwargs):
    print(f"\n=== {label}: spins {spin_kwargs} ===", flush=True)
    base = dict(BASE, **spin_kwargs)
    # Step 1: best GR over theta_jn x dt
    best_gr = None
    for th in (0.5, 1.0, 1.4, 2.0):
        for dt in DTS:
            p = dict(base, gamma=0.0, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                     theta_jn=th, geocent_time=TRIGGER + dt)
            val, mf, opt, _ = max_phase_lnl_ratio(ifos, wg, p)
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
            val, mf, opt, per = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt, mf, opt, per)
        curve[g] = {"lnL_ratio_max_phase": float(best[0]),
                    "best_dt_s": float(best[1]),
                    "matched_filter_snr": float(best[2]),
                    "optimal_snr": float(best[3]),
                    "per_detector": best[4]}
        print(f"  gamma={g:.1f}: lnL={best[0]:.1f} "
              f"(Delta={best[0]-best_gr[0]:+.1f}, dt={best[1]:+.3f}s, "
              f"|MF|={best[2]:.1f}, opt SNR={best[3]:.1f})", flush=True)
    return {"label": label, "spin_kwargs": spin_kwargs,
            "lnL_ratio_GR_best": float(best_gr[0]),
            "theta_jn_best": th_best, "dt_best_s": float(dt_best),
            "gamma_curve": curve,
            "delta_lnL_vs_GR": {str(g): float(curve[g]["lnL_ratio_max_phase"] - best_gr[0])
                                for g in GAMMAS}}


def gamma_at(ifos, wg, base, th, g, fm=F_MERGER, wf=WIDTH_FRAC, f_min=F_MIN):
    """Best max-phase lnL over dt for one (gamma, f_merger, width) point."""
    best = None
    for dt in DTS:
        p = dict(base, gamma=g, f_merger=fm, width_frac=wf,
                 theta_jn=th, geocent_time=TRIGGER + dt)
        val, mf, opt, _ = max_phase_lnl_ratio(ifos, wg, p, f_min=f_min)
        if best is None or val > best[0]:
            best = (val, dt)
    return best


def main():
    import time
    t0 = time.time()
    ifos = build_ifos()
    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                                reference_frequency=20.0,
                                minimum_frequency=F_MIN,
                                maximum_frequency=F_MAX, floor=0.5))
    # Timing probe: one XPHM waveform evaluation
    p_probe = dict(BASE, a_1=0.11, a_2=0.11, tilt_1=0.0, tilt_2=0.0,
                   phi_12=0.0, phi_jl=0.0, gamma=0.0, f_merger=F_MERGER,
                   width_frac=WIDTH_FRAC, theta_jn=1.0, geocent_time=TRIGGER)
    tp = time.time()
    max_phase_lnl_ratio(ifos, wg, p_probe)
    print(f"Timing probe: one XPHM likelihood eval = {time.time()-tp:.2f} s", flush=True)
    out = {"event": "GW200129_065458",
           "approximant": "IMRPhenomXPHM",
           "method": "profile likelihood, fixed nuisance parameters, phase "
                     "maximised analytically, dt scanned -0.20..+0.10 s, "
                     "theta_jn maximised over {0.5,1.0,1.4,2.0} at GR point, "
                     "sky fixed at stacking GR maximum ra=1.57 dec=0.80",
           "band_Hz": [20.0, 400.0], "notches": [],
           "f_merger_Hz_preregistered": F_MERGER, "width_frac": WIDTH_FRAC,
           "configs": {}}
    # Config A: aligned, isolates approximant change
    out["configs"]["A_aligned_XPHM"] = run_config(
        ifos, wg, "A: XPHM aligned a=0.11",
        dict(a_1=0.11, a_2=0.11, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0))
    # Config B: precessing, published spin magnitudes/tilt
    cfg_b = run_config(
        ifos, wg, "B: XPHM precessing a=0.90/0.50 tilt=1.4",
        dict(a_1=0.90, a_2=0.50, tilt_1=1.4, tilt_2=1.4, phi_12=0.0, phi_jl=0.0))
    out["configs"]["B_precessing_XPHM"] = cfg_b
    # Sensitivity under B: f_merger placement, width, f_min=30
    base_b = dict(BASE, a_1=0.90, a_2=0.50, tilt_1=1.4, tilt_2=1.4,
                  phi_12=0.0, phi_jl=0.0)
    th_b = cfg_b["theta_jn_best"]
    l0_b = cfg_b["lnL_ratio_GR_best"]
    sens = {}
    for fm in (133.0, 153.0):
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
    # f_min=30 Hz check: rebuild ifos + waveform generator at f_min=30
    ifos30 = build_ifos(f_min=30.0)
    wg30 = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                                reference_frequency=20.0,
                                minimum_frequency=30.0,
                                maximum_frequency=F_MAX, floor=0.5))
    v_gr30, _ = gamma_at(ifos30, wg30, base_b, th_b, 0.0, f_min=30.0)
    v_g30, dt30 = gamma_at(ifos30, wg30, base_b, th_b, 0.4, f_min=30.0)
    sens["f_min=30Hz"] = {"lnL_GR_fmin30": float(v_gr30),
                          "lnL_gamma0.4_fmin30": float(v_g30),
                          "delta_vs_GR_fmin30": float(v_g30 - v_gr30),
                          "best_dt_s": float(dt30)}
    print(f"sensitivity B f_min=30Hz: GR lnL={v_gr30:.1f}, "
          f"gamma=0.4 Delta={v_g30-v_gr30:+.1f}", flush=True)
    out["sensitivity_B"] = sens
    out["wall_time_s"] = float(time.time() - t0)
    json.dump(out, open(os.path.join(OUTDIR, "xphm_profile.json"), "w"), indent=2)
    print(f"Wrote xphm_profile.json (wall {time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
