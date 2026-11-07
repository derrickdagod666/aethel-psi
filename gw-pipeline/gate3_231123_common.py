"""Gate 3 — GW231123 shared setup: build Bilby interferometers from the
real GWOSC strain prepared by gate3_231123_data.py.

Band decision (made from OFF-SOURCE data, before any gamma likelihood):
  Primary band 30-256 Hz. Reasons:
  - L1 has a persistent, huge excess at 20-22 Hz in EVERY off-source
    8 s segment checked (per-bin <d|d> contributions 140-1170 at
    20.0-21.2 Hz; band <d|d> 20-30 Hz = 5k-30k off-source, vs ~300-700
    for H1). It is noise, not signal — it is present far from the event.
  - The H1 glitch documented in the LVK paper (Sec. III) sits at
    15-30 Hz, 1.7-1.1 s before the event, inside the segment.
  - The published signal band is 30-80 Hz (~5 cycles); template power
    below 30 Hz for a 334 Msun detector-frame binary is negligible
    (checked in validation: <h|h> fraction below 30 Hz).
  A f_min=20 Hz run is kept as a sensitivity check in the profile script.
  No notches are applied in 30-256 Hz: off-source top bins there are
  unremarkable (max per-bin contribution ~30, vs 5000-9000 for the
  506/512 Hz lines in GW250114, which are out of band here anyway).
"""
import numpy as np
import bilby
from bilby.gw.detector import get_empty_interferometer, PowerSpectralDensity

TRIGGER = 1384782888.7
F_MIN, F_MAX = 30.0, 256.0
FS, DUR = 4096.0, 8.0
NOTCHES = []


def build_ifos():
    ifos = []
    for det in ("H1", "L1"):
        d = np.load(f"data_{det}.npz")
        ifo = get_empty_interferometer(det)
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=d["psd_freqs"], psd_array=d["psd"])
        ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain=d["fd_strain"], sampling_frequency=FS,
            duration=DUR, start_time=float(d["seg_start"]))
        ifo.strain_data.notch_list = list(NOTCHES)
        ifo.minimum_frequency = F_MIN
        ifo.maximum_frequency = F_MAX
        ifos.append(ifo)
    return bilby.gw.detector.InterferometerList(ifos)


def inner(a, b, freqs, psd, mask, fmin=F_MIN, fmax=F_MAX):
    m = mask & (freqs >= fmin) & (freqs <= fmax)
    df = freqs[1] - freqs[0]
    return 4.0 * np.sum((np.conj(a[m]) * b[m] / psd[m]).real) * df
