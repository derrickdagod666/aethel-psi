"""Gate 3 — Bilby PE for GW250114: FIXED UAF waveform vs GR (IMRPhenomPv2).

Pre-registered design (written before any PE likelihood was evaluated):
  Data: real GWOSC O4b 4 kHz strain, H1+L1, 8 s segment starting
  TRIGGER-6 s, Tukey roll_off=0.2 s, 20-896 Hz, Welch-median PSD from
  256 s off-source, notches 504-508 and 510-514 Hz (see gate3_common.py).
  The +7 ms offset between the published NRSur7dq4 t_peak and the
  IMRPhenomPv2 t_c that best matches the data was MEASURED in
  gate3_validate.py (|MF| network 78.2 at dt=+0.007 s vs 17.8 at dt=0);
  the geocent_time prior below covers it with wide margin. It is a
  model-definition difference, not a fit to gamma.

  Fixed (NRSur7dq4 medians, LVK discovery paper):
    detector-frame m1=36.6, m2=35.1 Msun (source 33.6/32.2, z=0.09),
    a1=a2=0, tilt1=tilt2=0, phi12=phi_jl=0, ra=2.333 rad, dec=0.190 rad,
    psi=0. Fixed masses/spins is the REDUCED design the task allows;
    published 90% intervals are ~+-1 Msun, so the bias is small and the
    f_merger parameter is free anyway.

  Sampled, UAF model (7 params):
    gamma         Uniform(0, 0.9)          # paper value 0.4; 0 = GR
    f_merger      Uniform(100, 200) Hz     # paper value ~150 Hz
    width_frac    Uniform(0.05, 0.30)      # paper value 0.15
    luminosity_distance Uniform(200, 700) Mpc   # published 403+74/-?
    theta_jn      Uniform(0, pi)
    phase         Uniform(0, 2*pi)
    geocent_time  Uniform(TRIGGER-0.05, TRIGGER+0.05)
  Sampled, GR model (4 params): the same except gamma/f_merger/width_frac
  absent (gamma = 0 identically).

  Sampler: dynesty, nlive=1000, dlogz=0.1, sample='rwalk', walks=25,
  npool=4. Two runs on identical data/settings; Bayes factor =
  lnZ_UAF - lnZ_GR.

  Stopping rule (revised edition Sec. 6): if gamma = 0.4 is excluded at
  95% credibility on GW250114, Prediction 1 is revised downward or
  dropped. It is NOT rescued by refitting the width after seeing the
  data. No parameter, prior, or notch is changed after seeing results.

  UAF waveform (corrected listing, revised edition Sec. 6):
    width = width_frac * f_merger (FIX 1: not 0.15 Hz absolute)
    f_merger passed explicitly (FIX 2: never argmax(|h|), which gives 40 Hz)
    modifier = max(1 - gamma*exp(-(f-f_merger)^2/(2 width^2)), floor=0.5)
  Implementation note: floor is passed via waveform_arguments and read
  from **kwargs, NOT as a named parameter of the source function — Bilby's
  _format_parameters pops named source parameters that are absent from the
  sampled parameter dict, which crashed an earlier diagnostic (KeyError).
  gamma / f_merger / width_frac ARE named parameters and ARE sampled, so
  they survive formatting; they are deliberately NOT in waveform_arguments
  (which would overwrite the sampled values).
"""
import json
import numpy as np
import bilby
from bilby.gw import GravitationalWaveTransient
from gate3_common import build_ifos, TRIGGER, F_MIN, F_MAX, FS, DUR


def uaf_blunted_waveform(frequency_array, mass_1, mass_2, luminosity_distance,
                         theta_jn, phase, a_1=0.0, a_2=0.0, tilt_1=0.0,
                         tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
                         gamma=0.4, width_frac=0.15, f_merger=None, **kwargs):
    # FIX 2: f_merger supplied explicitly. Never argmax(|h|).
    if f_merger is None:
        raise ValueError("pass f_merger")
    floor = kwargs.get("floor", 0.5)
    pol = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn,
        phase=phase, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, phi_jl=phi_jl, **kwargs)
    # FIX 1: width is a FRACTION of f_merger (22.5 Hz at 150 Hz, FWHM 53 Hz),
    # not 0.15 Hz absolute (FWHM 0.35 Hz).
    width = width_frac * f_merger
    modifier = 1.0 - gamma * np.exp(
        -(frequency_array - f_merger) ** 2 / (2.0 * width ** 2))
    modifier = np.maximum(modifier, floor)
    return {k: v * modifier for k, v in pol.items()}


FIXED = dict(mass_1=36.6, mass_2=35.1, a_1=0.0, a_2=0.0, tilt_1=0.0,
             tilt_2=0.0, phi_12=0.0, phi_jl=0.0, ra=2.333, dec=0.190, psi=0.0)

COMMON_PRIORS = dict(
    luminosity_distance=bilby.core.prior.Uniform(200, 700, "luminosity_distance"),
    theta_jn=bilby.core.prior.Uniform(0, np.pi, "theta_jn"),
    phase=bilby.core.prior.Uniform(0, 2 * np.pi, "phase"),
    geocent_time=bilby.core.prior.Uniform(TRIGGER - 0.05, TRIGGER + 0.05,
                                          "geocent_time"),
)
UAF_PRIORS = dict(
    gamma=bilby.core.prior.Uniform(0, 0.9, "gamma"),
    f_merger=bilby.core.prior.Uniform(100, 200, "f_merger"),
    width_frac=bilby.core.prior.Uniform(0.05, 0.30, "width_frac"),
    **COMMON_PRIORS,
)

WAVEFORM_ARGS = dict(waveform_approximant="IMRPhenomPv2",
                     reference_frequency=20.0, minimum_frequency=F_MIN,
                     maximum_frequency=F_MAX)


def make_likelihood(ifos, model):
    if model == "uaf":
        wg = bilby.gw.WaveformGenerator(
            duration=DUR, sampling_frequency=FS,
            frequency_domain_source_model=uaf_blunted_waveform,
            waveform_arguments=dict(WAVEFORM_ARGS, floor=0.5))
        priors = bilby.core.prior.PriorDict(UAF_PRIORS)
    else:
        wg = bilby.gw.WaveformGenerator(
            duration=DUR, sampling_frequency=FS,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(WAVEFORM_ARGS))
        priors = bilby.core.prior.PriorDict(COMMON_PRIORS)
    for k, v in FIXED.items():
        priors[k] = bilby.core.prior.DeltaFunction(v)
    return GravitationalWaveTransient(ifos, wg, priors=priors), priors


def main():
    ifos = build_ifos()
    for model in ("uaf", "gr"):
        like, priors = make_likelihood(ifos, model)
        result = bilby.run_sampler(
            likelihood=like, priors=priors, sampler="dynesty",
            nlive=1000, dlogz=0.1, sample="rwalk", walks=25, npool=4,
            outdir=f"gate3_{model}", label="GW250114", clean=True,
            check_point_plot=True, check_point_delta_t=600)
        print(f"[{model}] lnZ = {result.log_evidence:.2f} "
              f"+/- {result.log_evidence_err:.2f}")
        print(f"[{model}] lnL max = {result.log_likelihood_evaluations.max() if hasattr(result, 'log_likelihood_evaluations') else 'n/a'}")
        result.save_to_file()
    print("Both runs complete.")


if __name__ == "__main__":
    main()
