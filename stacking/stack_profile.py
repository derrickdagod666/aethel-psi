"""6a stacking — profile likelihood for GW150914 and GW200129 (REAL data).

Method mirrors gate3_231123_profile.py / gate3_profile.py exactly:
  - Fixed UAF waveform, both bugs fixed (width = width_frac*f_merger;
    f_merger explicit; floor 0.5), approximant IMRPhenomPv2.
  - Phase maximised analytically per template; dt scanned; theta_jn
    maximised at the GR point first, then held for the gamma curve.
  - Band 20-400 Hz, NO notches. Justification (off-source, before any
    gamma likelihood): persistent instrumental lines at ~500-520 Hz in
    both events (GW150914 L1 forest 497-520 Hz, per-bin <d|d> up to
    12647; GW200129 H1 lines 509.9/514.4 Hz in every off-source
    segment) dominate <d|d> above 400 Hz and sit far from the dip
    (~150 Hz). Template power above 400 Hz for a ~70 Msun binary is
    negligible (checked in-run: optimal SNR fraction above 400 Hz).
    Using f_max=400 Hz instead of notching is the cleaner cut; the
    GW250114/GW231123 profiles keep their own bands — stacking sums
    Delta lnL per event, which does not require identical bands.
  - Sky (ra, dec): not published as medians (GWOSC parameters API
    gives only sky_area). Determined by a coarse GR matched-filter
    maximisation over ra in {0, pi/2, pi, 3pi/2} x dec in {-0.8, 0, 0.8}
    at gamma=0 — a nuisance maximisation analogous to the theta_jn
    scan in the earlier profiles, done BEFORE any gamma curve, and
    reported (chosen sky + SNR achieved vs published).
  - dt grids: GW150914 -0.05..+0.05 s (timing like GW250114, ms-level
    Pv2 offset expected); GW200129 -0.10..+0.10 s (precessing event,
    wider model timing spread, as GW231123). If the best dt sits on a
    grid edge the grid is widened and labelled a diagnostic, as in
    the GW231123 dt_wide run.
  - f_merger pre-registered from total detector mass (stack_data.py):
    GW150914 151 Hz, GW200129 143 Hz. Sensitivity: f_merger +/- ~10 Hz,
    width_frac 0.10/0.25, and for GW200129 an f_min=30 Hz check
    (L1 DQmask CAT2 failures around the event + elevated 20-50 Hz
    <d|d> in the event segment; see stack_data_200129.log).

Stacking: Delta lnL(gamma) summed over events (common gamma — valid
only under the flat gamma(M)=0.00903 derivation). Reported with and
without GW231123 (leave-one-out), since its profile prefers gamma
and is placement-sensitive.
"""
import json
import os
import sys
import numpy as np
import bilby
from bilby.gw.detector import get_empty_interferometer, PowerSpectralDensity

sys.path.insert(0, "/home/hatch/workspace/gate3-gw250114")
sys.path.insert(0, "/home/hatch/workspace/gate3-gw231123")
from gate3_pe import uaf_blunted_waveform  # fixed UAF waveform (Pv2)

STACKDIR = "/home/hatch/workspace/gate3-gw250114/stacking"
F_MIN, F_MAX = 20.0, 400.0
FS, DUR = 4096.0, 8.0
GAMMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

EVENTS = {
    "GW150914": dict(trigger=1126259462.42, f_merger=151.0,
                     mass_1=39.5, mass_2=31.7, dl=420.0, a_1=0.0, a_2=0.0,
                     dts=tuple(np.round(np.arange(-0.05, 0.0501, 0.005), 3))),
    "GW200129": dict(trigger=1264316116.43, f_merger=143.0,
                     mass_1=40.7, mass_2=34.2, dl=890.0, a_1=0.11, a_2=0.11,
                     dts=tuple(np.round(np.arange(-0.10, 0.1001, 0.005), 3))),
}
SKY_GRID = [(ra, dec) for ra in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
            for dec in (-0.8, 0.0, 0.8)]
THETA_GRID = (0.5, 1.0, 1.4, 2.0)


def build_ifos(ev_name):
    ifos = []
    for det in ("H1", "L1"):
        d = np.load(os.path.join(STACKDIR, ev_name, f"data_{det}.npz"))
        ifo = get_empty_interferometer(det)
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=d["psd_freqs"], psd_array=d["psd"])
        ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain=d["fd_strain"], sampling_frequency=FS,
            duration=DUR, start_time=float(d["seg_start"]))
        ifo.minimum_frequency = F_MIN
        ifo.maximum_frequency = F_MAX
        ifos.append(ifo)
    return bilby.gw.detector.InterferometerList(ifos)


def cinner(a, b, fr, psd, mask):
    m = mask & (fr >= F_MIN) & (fr <= F_MAX)
    return 4.0 * np.sum(np.conj(a[m]) * b[m] / psd[m]) * (fr[1] - fr[0])


def eval_point(ifos, wg, params):
    """Max-phase lnL ratio (network), |MF|, optimal SNR; per-detector too."""
    pol = wg.frequency_domain_strain(params)
    z_tot, hh_tot = 0j, 0.0
    per_det = {}
    for ifo in ifos:
        h = ifo.get_detector_response(pol, params)
        d = ifo.frequency_domain_strain
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        z = cinner(d, h, fr, psd, mask)
        hh = cinner(h, h, fr, psd, mask).real
        z_tot += z
        hh_tot += hh
        per_det[ifo.name] = {"abs_z": float(abs(z)), "hh": float(hh),
                             "mf": float(abs(z) / np.sqrt(hh)) if hh > 0 else 0.0,
                             "opt_snr": float(np.sqrt(hh))}
    return (abs(z_tot) - 0.5 * hh_tot, abs(z_tot) / np.sqrt(hh_tot),
            np.sqrt(hh_tot), per_det)


def run_event(ev_name):
    global F_MIN
    cfg = EVENTS[ev_name]
    trig = cfg["trigger"]
    print(f"\n=== {ev_name}: f_merger={cfg['f_merger']} Hz, "
          f"m_det={cfg['mass_1']}/{cfg['mass_2']}, DL={cfg['dl']} ===", flush=True)
    ifos = build_ifos(ev_name)
    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=F_MIN,
                                maximum_frequency=F_MAX, floor=0.5))
    base = dict(mass_1=cfg["mass_1"], mass_2=cfg["mass_2"],
                luminosity_distance=cfg["dl"], a_1=cfg["a_1"], a_2=cfg["a_2"],
                tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0, phase=0.0,
                psi=0.0)
    # Step 1: GR best over sky x theta_jn x dt (nuisance maximisation)
    best_gr = None
    for ra, dec in SKY_GRID:
        for th in THETA_GRID:
            for dt in cfg["dts"]:
                p = dict(base, gamma=0.0, f_merger=cfg["f_merger"],
                         width_frac=0.15, theta_jn=th, ra=ra, dec=dec,
                         geocent_time=trig + dt)
                val, mf, opt, _ = eval_point(ifos, wg, p)
                if best_gr is None or val > best_gr[0]:
                    best_gr = (val, th, dt, mf, opt, ra, dec)
    print(f"  Best GR: lnL={best_gr[0]:.1f}, theta_jn={best_gr[1]}, "
          f"dt={best_gr[2]:+.3f}s, |MF|={best_gr[3]:.1f}, opt SNR={best_gr[4]:.1f}, "
          f"sky ra={best_gr[5]:.2f} dec={best_gr[6]:.2f}", flush=True)
    _, th_best, dt_best, _, _, ra_best, dec_best = best_gr
    # Template power above 400 Hz (fraction of <h|h> in 20-896 Hz band)
    wg_full = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf_blunted_waveform,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=20.0,
                                maximum_frequency=896.0, floor=0.5))
    p_full = dict(base, gamma=0.0, f_merger=cfg["f_merger"], width_frac=0.15,
                  theta_jn=th_best, ra=ra_best, dec=dec_best,
                  geocent_time=trig + dt_best)
    pol = wg_full.frequency_domain_strain(p_full)
    hh_tot = {}
    for ifo in ifos:
        h = ifo.get_detector_response(pol, p_full)
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        for lo, hi in [(20, 400), (400, 896)]:
            m = mask & (fr >= lo) & (fr < hi)
            hh_tot[f"{ifo.name}_{lo}_{hi}"] = float(
                (4.0 * np.sum((np.abs(h[m]) ** 2 / psd[m])) * (fr[1] - fr[0])).real)
    frac_above_400 = {ifo.name: hh_tot[f"{ifo.name}_400_896"] /
                      (hh_tot[f"{ifo.name}_20_400"] + hh_tot[f"{ifo.name}_400_896"])
                      for ifo in ifos}
    print(f"  Template <h|h> fraction above 400 Hz: {frac_above_400}", flush=True)
    # Step 2: gamma curve at best sky/theta, dt scanned
    curve = {}
    for g in GAMMAS:
        best = None
        for dt in cfg["dts"]:
            p = dict(base, gamma=g, f_merger=cfg["f_merger"], width_frac=0.15,
                     theta_jn=th_best, ra=ra_best, dec=dec_best,
                     geocent_time=trig + dt)
            val, mf, opt, per = eval_point(ifos, wg, p)
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
    # Step 3: sensitivities at gamma=0.4
    sens = {}
    fm0 = cfg["f_merger"]
    for fm in (fm0 - 10, fm0 + 10):
        best = None
        for dt in cfg["dts"]:
            p = dict(base, gamma=0.4, f_merger=fm, width_frac=0.15,
                     theta_jn=th_best, ra=ra_best, dec=dec_best,
                     geocent_time=trig + dt)
            val, mf, opt, _ = eval_point(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt)
        sens[f"f_merger={fm}"] = {"delta_vs_GR": float(best[0] - best_gr[0]),
                                  "best_dt_s": float(best[1])}
        print(f"  sensitivity f_merger={fm}: Delta={best[0]-best_gr[0]:+.1f}", flush=True)
    for wf in (0.10, 0.25):
        best = None
        for dt in cfg["dts"]:
            p = dict(base, gamma=0.4, f_merger=fm0, width_frac=wf,
                     theta_jn=th_best, ra=ra_best, dec=dec_best,
                     geocent_time=trig + dt)
            val, mf, opt, _ = eval_point(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt)
        sens[f"width_frac={wf}"] = {"delta_vs_GR": float(best[0] - best_gr[0]),
                                    "best_dt_s": float(best[1])}
        print(f"  sensitivity width_frac={wf}: Delta={best[0]-best_gr[0]:+.1f}", flush=True)
    if ev_name == "GW200129":
        # f_min=30 Hz check: rebuild ifos with minimum_frequency 30
        F_MIN = 30.0
        for ifo in ifos:
            ifo.minimum_frequency = 30.0
        best = None
        best_gr30 = None
        for dt in cfg["dts"]:
            p0 = dict(base, gamma=0.0, f_merger=fm0, width_frac=0.15,
                      theta_jn=th_best, ra=ra_best, dec=dec_best,
                      geocent_time=trig + dt)
            v0, _, _, _ = eval_point(ifos, wg, p0)
            if best_gr30 is None or v0 > best_gr30[0]:
                best_gr30 = (v0, dt)
            p = dict(base, gamma=0.4, f_merger=fm0, width_frac=0.15,
                     theta_jn=th_best, ra=ra_best, dec=dec_best,
                     geocent_time=trig + dt)
            v, _, _, _ = eval_point(ifos, wg, p)
            if best is None or v > best[0]:
                best = (v, dt)
        sens["f_min=30Hz"] = {"lnL_GR_fmin30": float(best_gr30[0]),
                              "lnL_gamma0.4_fmin30": float(best[0]),
                              "delta_vs_GR_fmin30": float(best[0] - best_gr30[0])}
        print(f"  sensitivity f_min=30Hz: GR lnL={best_gr30[0]:.1f}, "
              f"gamma=0.4 Delta={best[0]-best_gr30[0]:+.1f}", flush=True)
        F_MIN = 20.0
    out = {"event": ev_name, "method": "profile likelihood, fixed nuisance "
           "parameters, phase maximised analytically, dt scanned, theta_jn "
           "and sky maximised over grids at GR point",
           "band_Hz": [20.0, 400.0], "notches": [],
           "fixed_point": {**base, "f_merger_Hz": fm0, "width_frac": 0.15,
                           "theta_jn_best": th_best, "dt_best_s": float(dt_best),
                           "ra_best": float(ra_best), "dec_best": float(dec_best)},
           "lnL_ratio_GR_best": float(best_gr[0]),
           "gamma_curve": curve,
           "delta_lnL_vs_GR": {str(g): float(curve[g]["lnL_ratio_max_phase"] - best_gr[0])
                               for g in GAMMAS},
           "template_hh_fraction_above_400Hz": frac_above_400,
           "sensitivity_gamma0.4": sens}
    json.dump(out, open(os.path.join(STACKDIR, ev_name, "profile_gamma.json"), "w"), indent=2)
    print(f"  Wrote {ev_name}/profile_gamma.json", flush=True)
    return out


def main():
    results = {}
    for ev in ("GW150914", "GW200129"):
        results[ev] = run_event(ev)
    # Stacked Delta lnL: this work's two events + the two existing profiles
    p250114 = json.load(open("/home/hatch/workspace/gate3-gw250114/profile_gamma.json"))
    p231123 = json.load(open("/home/hatch/workspace/gate3-gw231123/profile_gamma.json"))
    stack = {"events_included": {}, "stacked_delta_lnL": {}, "stacked_delta_lnL_no231123": {}}
    curves = {}
    curves["GW250114"] = {float(k): float(v) for k, v in p250114["delta_lnL_vs_GR"].items()}
    curves["GW231123_primary"] = {float(k): float(v)
                                  for k, v in p231123["delta_lnL_vs_GR"].items()}
    for ev in ("GW150914", "GW200129"):
        curves[ev] = {float(k): v for k, v in results[ev]["delta_lnL_vs_GR"].items()}
    for ev, c in curves.items():
        stack["events_included"][ev] = c
    for g in GAMMAS:
        stack["stacked_delta_lnL"][str(g)] = float(sum(c[g] for c in curves.values()))
        stack["stacked_delta_lnL_no231123"][str(g)] = float(
            sum(c[g] for ev, c in curves.items() if ev != "GW231123_primary"))
    print("\nSTACKED Delta lnL vs GR (4 events):")
    for g in GAMMAS:
        print(f"  gamma={g:.1f}: stacked={stack['stacked_delta_lnL'][str(g)]:+.1f}, "
              f"no-231123={stack['stacked_delta_lnL_no231123'][str(g)]:+.1f}")
    json.dump(stack, open(os.path.join(STACKDIR, "stacked_profile.json"), "w"), indent=2)
    print("Wrote stacked_profile.json")


if __name__ == "__main__":
    main()
