"""Waveform modifier check — the CI gate for the pip package.

Verified values (2026-11-07, Bilby 2.8.2, live): modifier at
f_merger = 150 Hz is 1.000 / 0.600 / 0.500 for gamma = 0 / 0.4 / 0.9
(the 0.9 case hits the 0.5 floor). The pure-numpy check needs no
Bilby/LAL; the waveform-ratio check needs bilby + lalsuite and is
skipped if they are not installed (CI installs them, so it runs there).
"""
import numpy as np
import pytest

from aethel_psi import dip_modifier

FREQ = np.arange(20, 512, 0.25)
F_MERGER = 150.0


@pytest.mark.parametrize(
    "gamma,expected",
    [(0.0, 1.000), (0.4, 0.600), (0.9, 0.500)],
)
def test_modifier_at_merger(gamma, expected):
    mod = dip_modifier(FREQ, gamma, F_MERGER)
    idx = int(np.argmin(np.abs(FREQ - F_MERGER)))
    assert mod[idx] == pytest.approx(expected, abs=1e-12)


def test_modifier_bounds_and_shape():
    mod = dip_modifier(FREQ, 0.4, F_MERGER)
    assert mod.min() >= 0.5  # floor
    assert mod.max() <= 1.0
    assert mod[int(np.argmin(np.abs(FREQ - F_MERGER)))] == mod.min()
    # Far from the dip the modifier returns to ~1.
    assert mod[0] == pytest.approx(1.0, abs=1e-6)


def test_modifier_width_is_fractional():
    # FIX 1: width = 0.15 * f_merger = 22.5 Hz, so at f_merger +/- 22.5 Hz
    # the dip is 1 - 0.4*exp(-0.5) = 0.7574, not ~1 (as a 0.15 Hz width would give).
    mod = dip_modifier(FREQ, 0.4, F_MERGER)
    idx = int(np.argmin(np.abs(FREQ - (F_MERGER + 22.5))))
    assert mod[idx] == pytest.approx(1 - 0.4 * np.exp(-0.5), abs=1e-3)


def test_waveform_ratio_matches_modifier():
    bilby = pytest.importorskip("bilby", reason="bilby not installed")
    pytest.importorskip("lal", reason="lalsuite not installed")
    from aethel_psi import uaf_blunted_waveform

    args = dict(mass_1=36.6, mass_2=35.1, luminosity_distance=403.0,
                theta_jn=0.4, phase=0.0, a_1=0.0, a_2=0.0,
                tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
    base = bilby.gw.source.lal_binary_black_hole(FREQ, **args)
    idx = int(np.argmin(np.abs(FREQ - F_MERGER)))
    for gamma, expected in [(0.0, 1.000), (0.4, 0.600), (0.9, 0.500)]:
        out = uaf_blunted_waveform(FREQ, gamma=gamma, f_merger=F_MERGER, **args)
        ratio = out["plus"][idx] / base["plus"][idx]
        assert abs(ratio) == pytest.approx(expected, abs=1e-6)
    # f_merger is required (FIX 2).
    with pytest.raises(ValueError, match="pass f_merger"):
        uaf_blunted_waveform(FREQ, gamma=0.4, **args)
