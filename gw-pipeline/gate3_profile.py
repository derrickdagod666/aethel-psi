"""Gate 3 — profile-likelihood result (REDUCED analysis, NOT a full PE).

No dynesty/PE run was launched. What this does, in minutes, with only
fast likelihood evaluations (0.01-0.04 s each):

  At the measured best-fit GR point (theta_jn=0.7, DL=403 Mpc,
  geocent_time = TRIGGER+0.007 s, phase maximised analytically), compute
  the PSD-weighted log-likelihood ratio for the FIXED UAF waveform as a
  function of gamma, with f_merger=150 Hz and width_frac=0.15 held at the
  paper values, plus sensitivity checks at f_merger=130/170 Hz and
  width_frac=0.10/0.25.

  Phase is maximised analytically per template: for template h,
  max_phase lnL_ratio = |<d|h>| - 0.5<h|h>|, where <.|.> is the
  noise-weighted inner product summed over H1+L1 (20-896 Hz, notches
  504-508 / 510-514 Hz). Time shift dt is scanned on a small grid
  {0.005, 0.007, 0.009} s and the best is taken per gamma (dt is a
  nuisance parameter here, not a UAF parameter).

  Interpretation (labelled precisely): this is a PROFILE likelihood at
  fixed nuisance parameters (masses, spins, sky, distance, inclination,
  f_merger, width_frac), NOT a marginalised posterior and NOT a Bayes
  factor. Upper limits use the Wilks approximation for 1 parameter
  (Delta lnL = -1.35 for 90%, -1.92 for 95%, two-sided). A full PE
  (gate3_pe.py, pre-registered priors) would marginalise over all of
  these and could shift the limit by O(0.05-0.1 in gamma); it was not run.
"""
import json
import numpy as np
import bilby
from gate3_common import build_ifos, TRIGGER, F_MIN, F_MAX, FS, DUR
from gate3_pe import uaf_blunted_waveform

BASE = dict(mass_1=36.6, mass_2=35.1, luminosity_distance=403.0,
            a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0,
            phi_jl=0.0, phase=0.0, psi=0.0, theta_jn=0.7,
            ra=2.333, dec=0.190)


def cinner(a, b, fr, psd, mask):
    m = mask & (fr >= F_MIN) & (fr <= F_MAX)
    return 4.0 * np.sum(np.conj(a[m]) * b[m] / psd[m]) * (fr[1] - fr[0])


def max_phase_lnl_ratio(ifos, wg, params):
    """max over phase of lnL_ratio = |<d|h>| - 0.5<h|h>|, network sum."""
    pol = wg.frequency_domain_strain(params)
    # Phase is a single global parameter: z_total = sum_I <d_I|h_I> (complex),
    # hh_total = sum_I <h_I|h_I>. max_phi lnL = |z_total| - 0.5*hh_total.
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
    dts = (0.005, 0.007, 0.009)
    gammas = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    curve = {}
    for g in gammas:
        best = None
        for dt in dts:
            p = dict(BASE, gamma=g, f_merger=150.0, width_frac=0.15,
                     geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt, mf, opt)
        curve[g] = {"lnL_ratio_max_phase": float(best[0]),
                    "best_dt_s": float(best[1]),
                    "matched_filter_snr": float(best[2]),
                    "optimal_snr": float(best[3])}
        print(f"gamma={g:.1f}: max-phase lnL ratio={best[0]:.1f} "
              f"(dt={best[1]:.3f}s, |MF|={best[2]:.1f}, opt SNR={best[3]:.1f})")
    l0 = curve[0.0]["lnL_ratio_max_phase"]
    for g in gammas:
        print(f"  gamma={g:.1f}: Delta lnL vs GR = "
              f"{curve[g]['lnL_ratio_max_phase'] - l0:+.1f}")
    # sensitivity: f_merger / width_frac at gamma=0.4 (and gamma=0 baseline)
    sens = {}
    for fm, wf in [(130.0, 0.15), (170.0, 0.15), (150.0, 0.10), (150.0, 0.25)]:
        best = None
        for dt in dts:
            p = dict(BASE, gamma=0.4, f_merger=fm, width_frac=wf,
                     geocent_time=TRIGGER + dt)
            val, mf, opt = max_phase_lnl_ratio(ifos, wg, p)
            if best is None or val > best[0]:
                best = (val, dt)
        sens[f"f_merger={fm}_width_frac={wf}"] = {
            "lnL_ratio_max_phase": float(best[0]),
            "delta_vs_GR": float(best[0] - l0), "best_dt_s": float(best[1])}
        print(f"sensitivity gamma=0.4, f_merger={fm}, width_frac={wf}: "
              f"Delta lnL vs GR = {best[0] - l0:+.1f}")
    # PSD-weighted mismatch between GR and UAF templates at gamma=0.4
    pol_gr = wg.frequency_domain_strain(
        dict(BASE, gamma=0.0, f_merger=150.0, width_frac=0.15,
             geocent_time=TRIGGER + 0.007))
    pol_uaf = wg.frequency_domain_strain(
        dict(BASE, gamma=0.4, f_merger=150.0, width_frac=0.15,
             geocent_time=TRIGGER + 0.007))
    mism = {}
    for ifo in ifos:
        h0 = ifo.get_detector_response(pol_gr, dict(BASE, geocent_time=TRIGGER + 0.007))
        h1 = ifo.get_detector_response(pol_uaf, dict(BASE, geocent_time=TRIGGER + 0.007))
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        hh0 = cinner(h0, h0, fr, psd, mask).real
        hh1 = cinner(h1, h1, fr, psd, mask).real
        h01 = cinner(h0, h1, fr, psd, mask)
        overlap = abs(h01) / np.sqrt(hh0 * hh1)
        mism[ifo.name] = {"overlap": float(overlap),
                          "mismatch": float(1 - overlap)}
        print(f"{ifo.name}: PSD-weighted overlap GR vs UAF(gamma=0.4) = "
              f"{overlap:.4f}, mismatch = {1 - overlap:.4f}")
    out = {"method": "profile likelihood, fixed nuisance parameters, "
                     "phase maximised analytically, dt scanned {0.005,0.007,0.009}s",
           "fixed_point": {**BASE, "geocent_time_note": "TRIGGER+dt, dt scanned"},
           "gamma_curve": curve,
           "delta_lnL_vs_GR": {str(g): float(curve[g]["lnL_ratio_max_phase"] - l0)
                               for g in gammas},
           "sensitivity_gamma0.4": sens,
           "psd_weighted_mismatch_gamma0.4": mism,
           "lnL_ratio_GR_best": float(l0)}
    json.dump(out, open("profile_gamma.json", "w"), indent=2)
    # Wilks limits by linear interpolation on the Delta lnL curve
    gs = np.array(gammas)
    dl = np.array([curve[g]["lnL_ratio_max_phase"] - l0 for g in gammas])
    for thresh, label in [(-1.35, "90%"), (-1.92, "95%")]:
        below = np.where(dl < thresh)[0]
        if len(below) == 0:
            print(f"{label} upper limit: gamma < 0.9 (not reached in grid)")
            out[f"wilks_upper_limit_{label}"] = ">0.9 (not reached)"
        else:
            i = below[0]
            if i == 0:
                lim = float(gs[0])
            else:
                lim = float(gs[i - 1] + (thresh - dl[i - 1]) * (gs[i] - gs[i - 1]) / (dl[i] - dl[i - 1]))
            print(f"{label} upper limit (Wilks, profile): gamma < {lim:.2f}")
            out[f"wilks_upper_limit_{label}"] = lim
    json.dump(out, open("profile_gamma.json", "w"), indent=2)
    print("Wrote profile_gamma.json")


if __name__ == "__main__":
    main()
