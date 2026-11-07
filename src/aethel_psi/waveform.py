"""UAF blunted waveform — importable form of ``gw-pipeline/uaf_bilby_plugin.py``.

Extracted verbatim in logic from the Gate 3 plugin, decoupled from
``gate3_common.py`` (which builds interferometers from downloaded GWOSC
strain and is not part of this package). The Gate 3 PE scripts in
``gw-pipeline/`` remain the canonical run scripts; this module is the
same waveform, importable as ``from aethel_psi import uaf_blunted_waveform``.

Verified values (2026-11-07, Bilby 2.8.2, live): modifier at
f_merger = 150 Hz is 1.000 / 0.600 / 0.500 for gamma = 0 / 0.4 / 0.9
(the 0.9 case hits the 0.5 floor). See ``tests/test_waveform_modifier.py``.

FIX 1: width = width_frac * f_merger (22.5 Hz at 150 Hz, FWHM 53 Hz),
not 0.15 Hz absolute (FWHM 0.35 Hz).
FIX 2: f_merger passed explicitly. Never argmax(|h|), which gives 40 Hz.
Floor is passed via waveform_arguments and read from **kwargs, NOT as a
named parameter of the source function — Bilby's _format_parameters pops
named source parameters absent from the sampled parameter dict.
"""

import numpy as np

# Frequency band used by the Gate 3 runs (gate3_common.py: F_MIN, F_MAX).
F_MIN, F_MAX = 20.0, 896.0

WAVEFORM_ARGS = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=20.0,
    minimum_frequency=F_MIN,
    maximum_frequency=F_MAX,
)


def dip_modifier(frequency_array, gamma, f_merger, width_frac=0.15, floor=0.5):
    """Pure-numpy UAF dip modifier (no Bilby/LAL needed).

    modifier = max(1 - gamma * exp(-(f - f_merger)^2 / (2 * width^2)), floor)
    with width = width_frac * f_merger.

    At f = f_merger this is max(1 - gamma, floor): 1.000 / 0.600 / 0.500
    for gamma = 0 / 0.4 / 0.9.
    """
    frequency_array = np.asarray(frequency_array, dtype=float)
    width = width_frac * f_merger
    modifier = 1.0 - gamma * np.exp(
        -(frequency_array - f_merger) ** 2 / (2.0 * width ** 2)
    )
    return np.maximum(modifier, floor)


def uaf_blunted_waveform(frequency_array, mass_1, mass_2, luminosity_distance,
                         theta_jn, phase, a_1=0.0, a_2=0.0, tilt_1=0.0,
                         tilt_2=0.0, phi_12=0.0, phi_jl=0.0,
                         gamma=0.4, width_frac=0.15, f_merger=None, **kwargs):
    """UAF waveform: IMRPhenomPv2 baseline times the Gaussian dip modifier.

    ``f_merger`` MUST be passed explicitly (FIX 2). ``floor`` is read from
    ``**kwargs`` (default 0.5); see the module docstring for why.
    """
    import bilby

    if f_merger is None:
        raise ValueError("pass f_merger")
    floor = kwargs.get("floor", 0.5)
    pol = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn,
        phase=phase, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, phi_jl=phi_jl, **kwargs)
    modifier = dip_modifier(frequency_array, gamma, f_merger,
                            width_frac=width_frac, floor=floor)
    return {k: v * modifier for k, v in pol.items()}
