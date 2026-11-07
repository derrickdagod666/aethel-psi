"""Gate 3 — validation on REAL GW250114 strain (before any PE).

Checks, with fixed published parameters (no fitting):
  1. Notch mask active at 506/512 Hz; <d|d> after notching.
  2. Matched-filter SNR of the GR IMRPhenomPv2 template at the published
     NRSur7dq4 medians. Expected: network ~80 (LVK: 80). Per-detector MF
     SNR must be positive and <= optimal SNR (within ~noise fluctuations).
  3. log-likelihood of GR template vs no-signal: Delta lnL ~ 0.5*SNR^2.
  4. Diagnostic gamma scan at fixed other parameters (f_merger=150 Hz,
     width_frac=0.15): lnL(gamma) for gamma in 0..0.9. PREVIEW ONLY — the
     PE posterior in gate3_pe.py is the result, not this scan.

Fixed parameters: detector-frame m1=36.6, m2=35.1 Msun (source 33.6/32.2,
z=0.09), DL=403 Mpc, RA=2.333 rad, dec=0.190 rad, a1=a2=0, tilts 0,
geocent_time=1420878141.235932. theta_jn is NOT published in the material
I have, so it is scanned over {0.5, 1.0, 1.5, 2.0} rad for the SNR check
and fixed at 1.0 rad for the gamma preview (PE samples it).
"""
import json
import numpy as np
import bilby
from gate3_common import build_ifos, inner, TRIGGER, F_MIN, F_MAX, FS, DUR
from gate3_pe import uaf_blunted_waveform as uaf


def main():
    ifos = build_ifos()
    # check notch mask + <d|d>
    for ifo in ifos:
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        print(f"{ifo.name}: mask at 506 Hz = {mask[np.argmin(np.abs(fr - 506))]}, "
              f"at 512 Hz = {mask[np.argmin(np.abs(fr - 512))]}, "
              f"at 150 Hz = {mask[np.argmin(np.abs(fr - 150))]}")
        dd = inner(ifo.frequency_domain_strain, ifo.frequency_domain_strain,
                   fr, ifo.power_spectral_density_array, mask)
        print(f"  <d|d> after notching = {dd:.0f} (noise-only expectation ~14000, "
              f"+ signal ~6400)")
    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=F_MIN,
                                maximum_frequency=F_MAX))
    base = dict(mass_1=36.6, mass_2=35.1, luminosity_distance=403.0,
                a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0,
                phi_jl=0.0, phase=0.0, psi=0.0, ra=2.333, dec=0.190,
                geocent_time=TRIGGER)
    out = {}
    for th in (0.5, 1.0, 1.5, 2.0):
        p = dict(base, theta_jn=th)
        pol = wg.frequency_domain_strain(p)
        snrs, opts = [], []
        for ifo in ifos:
            h = ifo.get_detector_response(pol, p)
            d = ifo.frequency_domain_strain
            psd = ifo.power_spectral_density_array
            fr = ifo.frequency_array
            mask = ifo.frequency_mask
            hh = inner(h, h, fr, psd, mask)
            dh = inner(d, h, fr, psd, mask)
            opts.append(np.sqrt(hh))
            snrs.append(dh / np.sqrt(hh))
        net_mf = float(np.sqrt(sum(s ** 2 for s in snrs)))
        net_opt = float(np.sqrt(sum(o ** 2 for o in opts)))
        print(f"theta_jn={th}: per-det MF SNR H1={snrs[0]:.1f} L1={snrs[1]:.1f} "
              f"network MF={net_mf:.1f} network optimal={net_opt:.1f}")
        out[f"theta_jn_{th}"] = {"mf_snr_H1": float(snrs[0]),
                                 "mf_snr_L1": float(snrs[1]),
                                 "network_mf_snr": net_mf,
                                 "network_optimal_snr": net_opt}
    # log-likelihood check (parameters passed explicitly: all-Delta priors
    # leave like.parameters empty in Bilby 2.8, so no-params calls fail).
    from bilby.gw import GravitationalWaveTransient
    like = GravitationalWaveTransient(
        ifos, wg, priors=bilby.core.prior.PriorDict())
    p_gr = dict(base, theta_jn=1.0)
    ll_gr = like.log_likelihood(parameters=p_gr)
    ll_0 = like.log_likelihood(parameters=dict(p_gr, luminosity_distance=1e9))
    print(f"\nlnL(GR template, theta_jn=1.0) = {ll_gr:.1f}")
    print(f"lnL(no signal) = {ll_0:.1f}  Delta lnL = {ll_gr - ll_0:.1f} "
          f"(expect ~0.5*SNR^2 = {0.5 * out['theta_jn_1.0']['network_mf_snr'] ** 2:.0f})")
    out["lnL_GR"] = float(ll_gr)
    out["lnL_nosignal"] = float(ll_0)

    # --- fixed UAF waveform, gamma preview (diagnostic only) ---
    # NOTE: superseded by gate3_profile.py. This preview is kept as a
    # cross-check at the MEASURED best point (theta_jn=0.7, phase=0.52,
    # geocent_time=TRIGGER+0.007 s, from the matched-filter scan). An
    # earlier version of this preview used phase=0 / dt=0, which is ~90
    # degrees and ~7 ms off the signal, and its numbers are meaningless.
    wg_uaf = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=uaf,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=F_MIN,
                                maximum_frequency=F_MAX,
                                floor=0.5))
    like_uaf = GravitationalWaveTransient(
        ifos, wg_uaf, priors=bilby.core.prior.PriorDict())
    like_gr2 = GravitationalWaveTransient(
        ifos, wg, priors=bilby.core.prior.PriorDict())
    p_best = dict(base, theta_jn=0.7, phase=0.52, geocent_time=TRIGGER + 0.007)
    ll_gr_best = like_gr2.log_likelihood(parameters=p_best)
    print(f"\nGamma preview at measured best point (theta_jn=0.7, phase=0.52, "
          f"dt=+0.007s); lnL(GR)={ll_gr_best:.1f}:")
    scan = {}
    for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9):
        ll = like_uaf.log_likelihood(parameters=dict(
            p_best, gamma=g, f_merger=150.0, width_frac=0.15))
        scan[g] = float(ll)
        print(f"  gamma={g:.1f}: lnL={ll:.1f}  Delta vs GR={ll - ll_gr_best:+.1f}")
    out["gamma_preview_best_point"] = scan
    out["lnL_GR_best_point"] = float(ll_gr_best)
    json.dump(out, open("validation.json", "w"), indent=2)
    print("\nWrote validation.json")


if __name__ == "__main__":
    main()
