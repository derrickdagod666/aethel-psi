"""GW231123 profile with IMRPhenomTPHM — second precessing family.

Cross-check on the XPHM result (PRECESSING_PE_REPORT §2): does the
35 Hz residual also survive the time-domain precessing Phenom model?
Same data/PSD/band/dt grid as precessing_profile.py; only the
approximant changes (XPHM -> TPHM) and only config B (precessing
spins a1=0.90/a2=0.80, tilt=1.0) plus the f_merger 25/45 Hz checks
are run. TPHM was verified available by direct waveform call before
this script (finite, no error); SEOBNRv5PHM and NRSur7dq4 remain
unavailable as documented in the report.
"""
import json
import os
import sys
import numpy as np
import bilby

sys.path.insert(0, "/home/hatch/workspace/gate3-gw231123/precessing-pe")
sys.path.insert(0, "/home/hatch/workspace/gate3-gw231123")
from precessing_profile import (build_ifos, max_phase_lnl_ratio, BASE,
                                F_MERGER, WIDTH_FRAC, DTS, GAMMAS, TRIGGER)
from gate3_231123_pe import uaf_blunted_waveform

OUTDIR = "/home/hatch/workspace/gate3-gw231123/precessing-pe"


def main():
    ifos = build_ifos()
    wg = bilby.gw.WaveformGenerator(
        duration=8.0, sampling_frequency=4096.0,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomTPHM",
                                reference_frequency=20.0,
                                minimum_frequency=30.0,
                                maximum_frequency=256.0, floor=0.5))
    base = dict(BASE, a_1=0.90, a_2=0.80, tilt_1=1.0, tilt_2=1.0,
                phi_12=0.0, phi_jl=0.0)
    print("=== TPHM precessing a=0.90/0.80 tilt=1.0 ===", flush=True)
    best_gr = None
    for th in (0.5, 1.0, 1.4, 2.0):
        for dt in DTS:
            p = dict(base, gamma=0.0, f_merger=F_MERGER, width_frac=WIDTH_FRAC,
                     theta_jn=th, geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best_gr is None or val > best_gr[0]:
                best_gr = (val, th, dt, mf, opt)
    print(f"  Best GR (TPHM): lnL={best_gr[0]:.1f}, theta_jn={best_gr[1]}, "
          f"dt={best_gr[2]:+.3f}s, |MF|={best_gr[3]:.1f}, opt SNR={best_gr[4]:.1f}",
          flush=True)
    th_best = best_gr[1]
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
    sens = {}
    for fm in (25.0, 45.0):
        best = None
        for dt in DTS:
            p = dict(base, gamma=0.4, f_merger=fm, width_frac=WIDTH_FRAC,
                     theta_jn=th_best, geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt)
        sens[f"f_merger={fm}"] = {"delta_vs_GR": float(best[0] - best_gr[0])}
        print(f"  sensitivity f_merger={fm}: Delta={best[0]-best_gr[0]:+.1f}", flush=True)
    out = {"approximant": "IMRPhenomTPHM",
           "lnL_ratio_GR_best": float(best_gr[0]),
           "theta_jn_best": th_best, "dt_best_s": float(best_gr[2]),
           "gamma_curve": curve,
           "delta_lnL_vs_GR": {str(g): float(curve[g]["lnL_ratio_max_phase"] - best_gr[0])
                               for g in GAMMAS},
           "sensitivity_gamma0.4": sens}
    json.dump(out, open(os.path.join(OUTDIR, "tphm_profile.json"), "w"), indent=2)
    print("Wrote tphm_profile.json", flush=True)


if __name__ == "__main__":
    main()
