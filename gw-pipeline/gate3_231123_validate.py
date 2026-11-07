"""Gate 3 — GW231123 validation on REAL strain (before any gamma evaluation).

Checks, with fixed published parameters (no fitting to gamma):
  1. <d|d> after band selection (30-256 Hz).
  2. Matched-filter SNR of the GR IMRPhenomPv2 template at the published
     combined medians, aligned spins a1=a2=0.31 (matching chi_eff=0.31;
     published spins 0.90/0.80 are tilted, chi_p=0.77 is UNMODELLED).
     Expected: network ~22.6 (LVK combined). Scan theta_jn and dt.
  3. log-likelihood of GR template vs no-signal.
  4. Template power fraction below 30 Hz (justifies F_MIN=30).
  5. Sensitivity of the recovered SNR to the spin choice (a=0, a=0.31,
     a=0.9 aligned) and to the sky (RA=3.02/dec=-0.02 vs RA=0/dec=0).

Fixed parameters: detector-frame m1=190.4, m2=143.2 Msun (source
137/103, z=0.39), DL=2200 Mpc, RA=3.02 rad, dec=-0.02 rad (TEOB
max-likelihood; NOT the LVK combined sky — see data script),
geocent_time=1384782888.7. theta_jn scanned; dt scanned on a grid.
"""
import json
import numpy as np
import bilby
from gate3_231123_common import build_ifos, inner, TRIGGER, F_MIN, F_MAX, FS, DUR

BASE = dict(mass_1=190.4, mass_2=143.2, luminosity_distance=2200.0,
            a_1=0.31, a_2=0.31, tilt_1=0.0, tilt_2=0.0, phi_12=0.0,
            phi_jl=0.0, phase=0.0, psi=0.0, ra=3.02, dec=-0.02)


def mf_snrs(ifos, wg, params):
    pol = wg.frequency_domain_strain(params)
    snrs, opts = [], []
    for ifo in ifos:
        h = ifo.get_detector_response(pol, params)
        d = ifo.frequency_domain_strain
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        hh = inner(h, h, fr, psd, mask)
        dh = inner(d, h, fr, psd, mask)
        opts.append(np.sqrt(hh))
        snrs.append(dh / np.sqrt(hh))
    return snrs, opts


def main():
    ifos = build_ifos()
    for ifo in ifos:
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        dd = inner(ifo.frequency_domain_strain, ifo.frequency_domain_strain,
                   fr, ifo.power_spectral_density_array, mask)
        print(f"{ifo.name}: <d|d> 30-256 Hz = {dd:.0f} "
              f"(noise-only expectation ~2 per bin x {(256-30)/0.125:.0f} bins ~ 3600)")
    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=F_MIN,
                                maximum_frequency=F_MAX))
    out = {}
    # template power below 30 Hz (using a wider-band generator for this check)
    wg_wide = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(waveform_approximant="IMRPhenomPv2",
                                reference_frequency=20.0,
                                minimum_frequency=20.0,
                                maximum_frequency=256.0))
    p0 = dict(BASE, theta_jn=1.4, geocent_time=TRIGGER)
    pol = wg_wide.frequency_domain_strain(p0)
    tot, below = 0.0, 0.0
    for ifo in ifos:
        h = ifo.get_detector_response(pol, p0)
        psd = ifo.power_spectral_density_array
        fr = ifo.frequency_array
        mask = ifo.frequency_mask
        # inner() defaults to F_MIN=30; recompute manually for 20-256
        m_all = mask & (fr >= 20) & (fr <= 256)
        m_lo = mask & (fr >= 20) & (fr < 30)
        df = fr[1] - fr[0]
        tot += 4.0 * np.sum((np.abs(h[m_all]) ** 2 / psd[m_all]).real) * df
        below += 4.0 * np.sum((np.abs(h[m_lo]) ** 2 / psd[m_lo]).real) * df
    print(f"Template <h|h> fraction below 30 Hz: {below/tot:.4f} "
          f"(total <h|h>={tot:.1f})")
    out["template_power_fraction_below_30Hz"] = float(below / tot)

    # theta_jn scan at dt=0
    for th in (0.5, 1.0, 1.4, 2.0, 2.5):
        p = dict(BASE, theta_jn=th, geocent_time=TRIGGER)
        snrs, opts = mf_snrs(ifos, wg, p)
        net_mf = float(np.sqrt(sum(s ** 2 for s in snrs)))
        net_opt = float(np.sqrt(sum(o ** 2 for o in opts)))
        print(f"theta_jn={th}: MF SNR H1={snrs[0]:.1f} L1={snrs[1]:.1f} "
              f"network MF={net_mf:.1f} network optimal={net_opt:.1f}")
        out[f"theta_jn_{th}"] = {"mf_snr_H1": float(snrs[0]),
                                 "mf_snr_L1": float(snrs[1]),
                                 "network_mf_snr": net_mf,
                                 "network_optimal_snr": net_opt}
    # dt scan at theta_jn=1.4 (phase is fixed 0 here; profile maximises phase)
    print("dt scan (theta_jn=1.4, phase=0):")
    best = None
    for dt in (-0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05):
        p = dict(BASE, theta_jn=1.4, geocent_time=TRIGGER + dt)
        snrs, opts = mf_snrs(ifos, wg, p)
        net_mf = float(np.sqrt(sum(s ** 2 for s in snrs)))
        print(f"  dt={dt:+.3f}s: network MF={net_mf:.1f} "
              f"(H1={snrs[0]:.1f}, L1={snrs[1]:.1f})")
        if best is None or net_mf > best[1]:
            best = (dt, net_mf)
    out["best_dt_s"] = float(best[0])
    out["best_dt_network_mf"] = float(best[1])

    # spin sensitivity at theta_jn=1.4, dt=best
    print("spin sensitivity (theta_jn=1.4, dt=best):")
    for a in (0.0, 0.31, 0.9):
        p = dict(BASE, a_1=a, a_2=a, theta_jn=1.4,
                 geocent_time=TRIGGER + best[0])
        snrs, opts = mf_snrs(ifos, wg, p)
        net_mf = float(np.sqrt(sum(s ** 2 for s in snrs)))
        net_opt = float(np.sqrt(sum(o ** 2 for o in opts)))
        print(f"  a1=a2={a}: network MF={net_mf:.1f} optimal={net_opt:.1f}")
        out[f"spin_a{a}"] = {"network_mf_snr": net_mf,
                             "network_optimal_snr": net_opt}
    # sky sensitivity
    print("sky sensitivity (theta_jn=1.4, dt=best, a=0.31):")
    for ra, dec, label in [(3.02, -0.02, "TEOB"), (0.0, 0.0, "alt")]:
        p = dict(BASE, ra=ra, dec=dec, theta_jn=1.4,
                 geocent_time=TRIGGER + best[0])
        snrs, opts = mf_snrs(ifos, wg, p)
        net_mf = float(np.sqrt(sum(s ** 2 for s in snrs)))
        print(f"  sky {label} (ra={ra}, dec={dec}): network MF={net_mf:.1f}")
        out[f"sky_{label}"] = {"network_mf_snr": net_mf}
    # lnL check at best point
    from bilby.gw import GravitationalWaveTransient
    like = GravitationalWaveTransient(
        ifos, wg, priors=bilby.core.prior.PriorDict())
    p_best = dict(BASE, theta_jn=1.4, geocent_time=TRIGGER + best[0])
    ll_gr = like.log_likelihood(parameters=p_best)
    ll_0 = like.log_likelihood(parameters=dict(p_best, luminosity_distance=1e9))
    print(f"\nlnL(GR template, best point) = {ll_gr:.1f}")
    print(f"lnL(no signal) = {ll_0:.1f}  Delta lnL = {ll_gr - ll_0:.1f}")
    out["lnL_GR"] = float(ll_gr)
    out["lnL_nosignal"] = float(ll_0)
    json.dump(out, open("validation.json", "w"), indent=2)
    print("\nWrote validation.json")


if __name__ == "__main__":
    main()
