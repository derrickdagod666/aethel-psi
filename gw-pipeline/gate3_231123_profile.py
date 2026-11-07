"""Gate 3 — GW231123 profile-likelihood result (REDUCED analysis, NOT a full PE).

Same method as gate3_profile.py (GW250114), adapted:
  - Fixed UAF waveform, both bugs fixed (width = 0.15 x f_merger;
    f_merger passed explicitly, never argmax).
  - Pre-registered central f_merger = 35 Hz (scaled from GW250114
    150 Hz x 71.5/333.6 = 32 Hz, ringdown scale ~35-40 Hz),
    width_frac = 0.15. Sensitivity scans (pre-registered, NOT refits):
    f_merger 25/45 Hz, width_frac 0.10/0.25, f_min 20 Hz, spin a=0/0.9,
    alt sky.
  - Phase maximised analytically per template; dt scanned on a grid
    -0.05..+0.05 s in 0.005 s steps (the published peak-time spread
    across models is ~0.1 s, wider than GW250114's 7 ms offset).
  - theta_jn is maximised over {0.5, 1.0, 1.4, 2.0} at the GR point
    first (nuisance parameter, as GW250114 fixed theta_jn=0.7 from
    its validation scan), then held at that best value for the
    gamma curve. DL=2200 Mpc, masses det 190.4/143.2, a1=a2=0.31,
    RA=3.02, dec=-0.02 fixed.
  - Band 30-256 Hz (see gate3_231123_common.py).

Interpretation (labelled precisely): PROFILE likelihood at fixed
nuisance parameters, NOT a marginalised posterior and NOT a Bayes
factor. Wilks limits for 1 parameter (Delta lnL = -1.35 for 90%,
-1.92 for 95%). A full PE would marginalise and could shift the
limit; it was not run.
"""
import json
import numpy as np
import bilby
from gate3_231123_common import build_ifos, TRIGGER, F_MIN, F_MAX, FS, DUR
from gate3_231123_pe import uaf_blunted_waveform

BASE = dict(mass_1=190.4, mass_2=143.2, luminosity_distance=2200.0,
            a_1=0.31, a_2=0.31, tilt_1=0.0, tilt_2=0.0, phi_12=0.0,
            phi_jl=0.0, phase=0.0, psi=0.0, theta_jn=1.4,
            ra=3.02, dec=-0.02)

F_MERGER = 35.0
WIDTH_FRAC = 0.15


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


def main():
    ifos = build_ifos()
    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=F_MIN,
                                maximum_frequency=F_MAX, floor=0.5))
    dts = tuple(np.round(np.arange(-0.05, 0.0501, 0.005), 3))
    # Step 1: best GR point over theta_jn x dt (phase maximised)
    print("Step 1: GR best point over theta_jn x dt")
    best_gr = None
    for th in (0.5, 1.0, 1.4, 2.0):
        for dt in dts:
            p = dict(BASE, gamma=0.0, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                     theta_jn=th, geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best_gr is None or val > best_gr[0]:
                best_gr = (val, th, dt, mf, opt)
    print(f"  Best GR: lnL ratio={best_gr[0]:.1f}, theta_jn={best_gr[1]}, "
          f"dt={best_gr[2]:+.3f}s, |MF|={best_gr[3]:.1f}, opt SNR={best_gr[4]:.1f}")
    th_best, dt_best = best_gr[1], best_gr[2]
    # Step 2: gamma curve at best theta_jn, dt scanned
    gammas = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    curve = {}
    for g in gammas:
        best = None
        for dt in dts:
            p = dict(BASE, gamma=g, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                     theta_jn=th_best, geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt, mf, opt)
        curve[g] = {"lnL_ratio_max_phase": float(best[0]),
                    "best_dt_s": float(best[1]),
                    "matched_filter_snr": float(best[2]),
                    "optimal_snr": float(best[3])}
        print(f"gamma={g:.1f}: max-phase lnL ratio={best[0]:.1f} "
              f"(dt={best[1]:+.3f}s, |MF|={best[2]:.1f}, opt SNR={best[3]:.1f})")
    l0 = curve[0.0]["lnL_ratio_max_phase"]
    for g in gammas:
        print(f"  gamma={g:.1f}: Delta lnL vs GR = "
              f"{curve[g]['lnL_ratio_max_phase'] - l0:+.1f}")
    # Step 3: sensitivity checks at gamma=0.4
    sens = {}
    for fm, wf, label in [(25.0, 0.15, "f_merger=25"), (45.0, 0.15, "f_merger=45"),
                          (35.0, 0.10, "width_frac=0.10"), (35.0, 0.25, "width_frac=0.25")]:
        best = None
        for dt in dts:
            p = dict(BASE, gamma=0.4, f_merger=fm, width_frac=wf,
                     theta_jn=th_best, geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt)
        sens[label] = {"lnL_ratio_max_phase": float(best[0]),
                       "delta_vs_GR": float(best[0] - l0), "best_dt_s": float(best[1])}
        print(f"sensitivity gamma=0.4, {label}: Delta lnL vs GR = {best[0] - l0:+.1f}")
    # spin sensitivity at gamma=0.4 and GR
    for a in (0.0, 0.9):
        best = None
        for dt in dts:
            p = dict(BASE, a_1=a, a_2=a, gamma=0.4, f_merger=F_MERGER,
                     width_frac=WIDTH_FRAC, theta_jn=th_best,
                     geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt, mf)
        # GR with same spin for a fair delta
        best_gr_a = None
        for dt in dts:
            p = dict(BASE, a_1=a, a_2=a, gamma=0.0, f_merger=F_MERGER,
                     width_frac=WIDTH_FRAC, theta_jn=th_best,
                     geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best_gr_a is None or val > best_gr_a[0]:
                best_gr_a = (val, dt, mf)
        sens[f"spin_a={a}"] = {"lnL_ratio_gamma0.4": float(best[0]),
                               "lnL_ratio_GR_same_spin": float(best_gr_a[0]),
                               "delta_vs_GR_same_spin": float(best[0] - best_gr_a[0]),
                               "best_dt_s": float(best[1]),
                               "matched_filter_snr_gamma0.4": float(best[2])}
        print(f"sensitivity spin a={a}: GR lnL={best_gr_a[0]:.1f} (|MF|={best_gr_a[2]:.1f}), "
              f"gamma=0.4 lnL={best[0]:.1f} (|MF|={best[2]:.1f}), "
              f"Delta={best[0]-best_gr_a[0]:+.1f}")
    # alt-sky sensitivity at gamma=0.4
    best = None
    best_gr_sky = None
    for dt in dts:
        p = dict(BASE, gamma=0.4, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                 theta_jn=th_best, ra=0.0, dec=0.0, geocent_time=TRIGGER + dt)
        val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
        if best is None or val > best[0]:
            best = (val, dt, mf)
        p0 = dict(BASE, gamma=0.0, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                  theta_jn=th_best, ra=0.0, dec=0.0, geocent_time=TRIGGER + dt)
        val0, mf0, opt0 = max_phase_lnl_ratio(ifos, wg, p0)
        if best_gr_sky is None or val0 > best_gr_sky[0]:
            best_gr_sky = (val0, dt, mf0)
    sens["sky_alt_ra0_dec0"] = {"lnL_ratio_gamma0.4": float(best[0]),
                                "lnL_ratio_GR_alt_sky": float(best_gr_sky[0]),
                                "delta_vs_GR_alt_sky": float(best[0] - best_gr_sky[0])}
    print(f"sensitivity alt sky (ra=0,dec=0): GR lnL={best_gr_sky[0]:.1f}, "
          f"gamma=0.4 lnL={best[0]:.1f}, Delta={best[0]-best_gr_sky[0]:+.1f}")
    # PSD-weighted mismatch GR vs UAF(gamma=0.4) at best point
    pol_gr = wg.frequency_domain_strain(
        dict(BASE, gamma=0.0, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
             theta_jn=th_best, geocent_time=TRIGGER + dt_best))
    pol_uaf = wg.frequency_domain_strain(
        dict(BASE, gamma=0.4, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
             theta_jn=th_best, geocent_time=TRIGGER + dt_best))
    mism = {}
    for ifo in ifos:
        h0 = ifo.get_detector_response(pol_gr, dict(BASE, theta_jn=th_best, geocent_time=TRIGGER + dt_best))
        h1 = ifo.get_detector_response(pol_uaf, dict(BASE, theta_jn=th_best, geocent_time=TRIGGER + dt_best))
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        hh0 = cinner(h0, h0, fr, psd, mask).real
        hh1 = cinner(h1, h1, fr, psd, mask).real
        h01 = cinner(h0, h1, fr, psd, mask)
        overlap = abs(h01) / np.sqrt(hh0 * hh1)
        mism[ifo.name] = {"overlap": float(overlap), "mismatch": float(1 - overlap)}
        print(f"{ifo.name}: PSD-weighted overlap GR vs UAF(gamma=0.4) = "
              f"{overlap:.4f}, mismatch = {1 - overlap:.4f}")
    out = {"method": "profile likelihood, fixed nuisance parameters, "
                     "phase maximised analytically, dt scanned -0.05..+0.05s, "
                     "theta_jn maximised over {0.5,1.0,1.4,2.0} at GR point",
           "fixed_point": {**BASE, "theta_jn_best": th_best, "dt_best_s": dt_best,
                           "f_merger_Hz": F_MERGER, "width_frac": WIDTH_FRAC},
           "gamma_curve": curve,
           "delta_lnL_vs_GR": {str(g): float(curve[g]["lnL_ratio_max_phase"] - l0)
                               for g in gammas},
           "sensitivity_gamma0.4": sens,
           "psd_weighted_mismatch_gamma0.4": mism,
           "lnL_ratio_GR_best": float(l0)}
    # Wilks limits
    gs = np.array(gammas)
    dl = np.array([curve[g]["lnL_ratio_max_phase"] - l0 for g in gammas])
    for thresh, label in [(-1.35, "90%"), (-1.92, "95%")]:
        below = np.where(dl < thresh)[0]
        if len(below) == 0:
            print(f"{label} upper limit: gamma < 0.9 (not reached in grid)")
            out[f"wilks_upper_limit_{label}"] = ">0.9 (not reached)"
        else:
            i = below[0]
            lim = float(gs[0]) if i == 0 else float(gs[i - 1] + (thresh - dl[i - 1]) * (gs[i] - gs[i - 1]) / (dl[i] - dl[i - 1]))
            print(f"{label} upper limit (Wilks, profile): gamma < {lim:.2f}")
            out[f"wilks_upper_limit_{label}"] = lim
    json.dump(out, open("profile_gamma.json", "w"), indent=2)
    print("Wrote profile_gamma.json")


if __name__ == "__main__":
    main()
