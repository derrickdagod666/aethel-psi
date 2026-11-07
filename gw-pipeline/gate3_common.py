"""Gate 3 — shared setup: build Bilby interferometers from the real
GW250114 GWOSC strain prepared by gate3_data.py.

Data-quality notches (decided from OFF-SOURCE data, not the event):
  Persistent instrumental lines at ~506 Hz and ~512 Hz appear in every
  8 s off-source segment (|d|^2/S ~ 5000-9000 per bin) and dominate <d|d>
  (300-896 Hz contributes 44732 of 58153 in the event segment). They are
  far from the UAF dip (f_merger ~150 Hz) and carry negligible template
  power. Notches 504-508 Hz and 510-514 Hz are therefore applied to both
  detectors. Off-source <d|d> is 51k-99k without notches, i.e. the excess
  is a property of the noise/lines, not of the event.
"""
import numpy as np
import bilby
from bilby.gw.detector import get_empty_interferometer, PowerSpectralDensity

TRIGGER = 1420878141.235932
F_MIN, F_MAX = 20.0, 896.0
FS, DUR = 4096.0, 8.0
NOTCHES = [(504.0, 508.0), (510.0, 514.0)]


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
