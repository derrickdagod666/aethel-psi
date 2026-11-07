"""Add-on tilt sensitivity for GW200129 XPHM (config C analogue of the
GW231123 precessing profile): at tilts 0.5 and 1.5 rad (a1=0.90, a2=0.50),
re-maximise GR over theta_jn x dt on the 0.5 ms grid, then gamma=0.4
at the same tilt. Tests whether the config-B softening of the
exclusion (Delta lnL -7.2 at tilt 1.4) is generic across precessing
points or specific to tilt=1.4. Same data/band/f_merger=143 Hz as the
main profile."""
import json, sys, numpy as np, bilby
sys.path.insert(0, '/home/hatch/workspace/gate3-gw200129/xphm')
import importlib.util
spec = importlib.util.spec_from_file_location('xp', 'xphm_profile.py')
xp = importlib.util.module_from_spec(spec); spec.loader.exec_module(xp)

ifos = xp.build_ifos()
wg = bilby.gw.WaveformGenerator(
    duration=xp.DUR, sampling_frequency=xp.FS,
    frequency_domain_source_model=xp.uaf_blunted_waveform,
    waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM",
                            reference_frequency=20.0,
                            minimum_frequency=xp.F_MIN,
                            maximum_frequency=xp.F_MAX, floor=0.5))
out = {}
for tilt in (0.5, 1.5):
    base = dict(xp.BASE, a_1=0.90, a_2=0.50, tilt_1=tilt, tilt_2=tilt,
                phi_12=0.0, phi_jl=0.0)
    best_gr = None
    for th in (0.5, 1.0, 1.4, 2.0):
        for dt in xp.DTS:
            p = dict(base, gamma=0.0, f_merger=xp.F_MERGER,
                     width_frac=xp.WIDTH_FRAC, theta_jn=th,
                     geocent_time=xp.TRIGGER + dt)
            val, mf, opt, _ = xp.max_phase_lnl_ratio(ifos, wg, p)
            if best_gr is None or val > best_gr[0]:
                best_gr = (val, th, dt, mf, opt)
    v, dt = xp.gamma_at(ifos, wg, base, best_gr[1], 0.4)
    out[f"tilt={tilt}"] = {
        "lnL_ratio_GR_same_tilt": float(best_gr[0]),
        "theta_jn_best_GR": best_gr[1],
        "dt_best_GR_s": float(best_gr[2]),
        "matched_filter_snr_GR": float(best_gr[3]),
        "optimal_snr_GR": float(best_gr[4]),
        "lnL_ratio_gamma0.4": float(v),
        "delta_vs_GR_same_tilt": float(v - best_gr[0]),
        "best_dt_gamma0.4_s": float(dt)}
    print(f"tilt={tilt}: GR lnL={best_gr[0]:.1f} (theta_jn={best_gr[1]}, "
          f"dt={best_gr[2]:+.4f}s, |MF|={best_gr[3]:.1f}, opt={best_gr[4]:.1f}), "
          f"gamma=0.4 lnL={v:.1f}, Delta={v-best_gr[0]:+.1f}", flush=True)
json.dump(out, open('tilt_check.json', 'w'), indent=2)
print('Wrote tilt_check.json', flush=True)
