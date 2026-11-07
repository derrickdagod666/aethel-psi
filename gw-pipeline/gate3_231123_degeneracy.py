"""Degeneracy checks for the gamma preference (NOT gamma refits):
 1. GR amplitude degeneracy: scan DL and theta_jn at gamma=0. If a GR
    point with different (DL, theta_jn) reaches the gamma=0.4 lnL
    (~198), the dip is mimicking a lower merger amplitude, not a
    distinctive shape.
 2. f_min=20 Hz: rebuild ifos with F_MIN=20 (includes the L1 20-22 Hz
    excess and H1 glitch band) and re-run the gamma curve at the wide
    best point. If the preference vanishes, it was a band artefact.
Both use phase maximisation and the wide dt grid.
"""
import json
import numpy as np
import bilby
import gate3_231123_common as C
from gate3_231123_common import build_ifos, TRIGGER, F_MAX, FS, DUR
from gate3_231123_pe import uaf_blunted_waveform
import gate3_231123_profile as P

def run_curve(ifos, wg, base, dts, gammas=(0.0, 0.4, 0.9)):
    out = {}
    for g in gammas:
        best = None
        for dt in dts:
            p = dict(base, gamma=g, f_merger=35.0, width_frac=0.15,
                     geocent_time=TRIGGER + dt)
            val, mf, opt = P.max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt, mf, opt)
        out[g] = {"lnL": float(best[0]), "dt": float(best[1]), "mf": float(best[2])}
    return out

def main():
    dts = tuple(np.round(np.arange(-0.20, 0.1001, 0.005), 3))
    # --- Check 1: GR DL/theta scan ---
    ifos = build_ifos()
    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=C.F_MIN,
                                maximum_frequency=F_MAX, floor=0.5))
    print("Check 1: GR (gamma=0) over DL x theta_jn")
    best_gr = None
    for dl in (1500, 2200, 3000, 4000, 5000):
        for th in (0.5, 1.0, 1.4, 2.0, 2.5):
            for dt in dts:
                p = dict(P.BASE, gamma=0.0, f_merger=35.0, width_frac=0.15,
                         luminosity_distance=float(dl), theta_jn=th,
                         geocent_time=TRIGGER + dt)
                val, mf, opt = P.max_phase_lnl_ratio(ifos, wg, p)
                if best_gr is None or val > best_gr[0]:
                    best_gr = (val, dl, th, dt, mf, opt)
    print(f"  Best GR over DL/theta/dt: lnL={best_gr[0]:.1f}, DL={best_gr[1]}, "
          f"theta_jn={best_gr[2]}, dt={best_gr[3]:+.3f}, |MF|={best_gr[4]:.1f}, opt={best_gr[5]:.1f}")
    # gamma=0.4 at that same (DL, theta) for comparison
    p = dict(P.BASE, gamma=0.4, f_merger=35.0, width_frac=0.15,
             luminosity_distance=float(best_gr[1]), theta_jn=best_gr[2],
             geocent_time=TRIGGER + best_gr[3])
    val, mf, opt = P.max_phase_lnl_ratio(ifos, wg, p)
    print(f"  gamma=0.4 at best-GR (DL,theta,dt): lnL={val:.1f} Delta={val-best_gr[0]:+.1f}")
    res = {"best_gr_DL_theta": {"lnL": float(best_gr[0]), "DL": best_gr[1],
                                 "theta_jn": best_gr[2], "dt": float(best_gr[3]),
                                 "mf": float(best_gr[4]), "opt": float(best_gr[5])},
           "gamma0.4_at_best_gr_point": float(val)}
    # --- Check 2: f_min=20 ---
    C.F_MIN = 20.0
    P.F_MIN = 20.0
    ifos20 = build_ifos()
    wg20 = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=20.0,
                                maximum_frequency=F_MAX, floor=0.5))
    base20 = dict(P.BASE, theta_jn=2.0)
    curve20 = run_curve(ifos20, wg20, base20, dts)
    print("Check 2: f_min=20 Hz, theta_jn=2.0, wide dt:")
    for g, v in curve20.items():
        print(f"  gamma={g}: lnL={v['lnL']:.1f} dt={v['dt']:+.3f} |MF|={v['mf']:.1f} "
              f"Delta vs GR(fmin20)={v['lnL']-curve20[0.0]['lnL']:+.1f}")
    res["fmin20_curve_theta2.0"] = curve20
    json.dump(res, open("degeneracy_checks.json", "w"), indent=2)
    print("Wrote degeneracy_checks.json")

if __name__ == "__main__":
    main()
