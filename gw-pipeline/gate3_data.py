"""Gate 3 — GW250114 data preparation (REAL GWOSC strain).

Event: GW250114_082203, 14 Jan 2025, 08:22:03 UTC.
GPS peak (geocenter): 1420878141.235932 s (LVK discovery paper, supplement Table II).
GWOSC O4b strain files (public release 10 Sep 2025, O4 Discovery Paper Data Release):
  H-H1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5  md5 f0f6c16c419282a26fa7a57810f715ac
  L-L1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5  md5 bea4dcf16e5dd9d64e425d66f97d5cc6
  GPS start 1420877824, duration 4096 s, 4 kHz, H1 + L1 only (Virgo not observing).
  Retrieved via https://gwosc.org/api/v2/events/GW250114_082203/strain-files
  (see gwosc_strain_files.json).

Analysis choices (pre-registered here, before any likelihood is evaluated):
  - 8 s segment, start GPS = trigger - 6 s, so the peak sits 6 s in, 2 s of
    post-peak data. Frequency range 20-896 Hz (as in the LVK paper).
  - Tukey window, roll_off=0.2 s (alpha=0.05 for 8 s), applied to the
    time-domain segment before the FFT — exactly Bilby's standard
    InterferometerStrainData default (roll_off=0.2 s). This is REQUIRED:
    the raw GWOSC strain carries enormous <20 Hz power and a DC offset,
    and a rectangular window leaks it across the whole band (measured:
    unwindowed <d|d> ~2.5e9 vs ~1.4e4 expected; periodogram/PSD ~1e5
    below 300 Hz). With the Tukey window the leakage is suppressed and
    the data are consistent with the off-source PSD. The segment is
    FFT'd as rfft(seg*window)/fs, Bilby's nfft convention, and the
    frequency-domain strain is set with window_factor=1, as Bilby does
    by default (BILBY_INCORRECT_PSD_NORMALIZATION unset).
  - PSD: Welch median over 256 s of off-source data (file seconds 40-296,
    i.e. GPS 1420877864-1420878120, ending 21 s before the peak), 4 s Hann
    segments, 50% overlap. No on-source data enter the PSD.
  - Fixed source parameters (NRSur7dq4 medians, LVK discovery paper):
    source-frame m1=33.6, m2=32.2 Msun, z=0.09 -> detector-frame
    m1=36.6, m2=35.1 Msun; spins fixed 0 (published chi <= 0.26);
    RA=2.333 rad, dec=0.190 rad (supplement Table II).
    Sampled in PE: gamma, f_merger, width_frac, luminosity_distance,
    theta_jn, phase, geocent_time (see gate3_pe.py).
"""
import json
import h5py
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import tukey

TRIGGER = 1420878141.235932  # GPS, geocenter peak
GPS_START = 1420877824
DURATION_FILE = 4096
FS = 4096.0
SEG_DUR = 8.0
SEG_START = TRIGGER - 6.0
F_MIN, F_MAX = 20.0, 896.0

FILES = {
    "H1": "H-H1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5",
    "L1": "L-L1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5",
}


def load_strain(det):
    with h5py.File(FILES[det], "r") as h:
        strain = h["strain/Strain"][:]
        meta = {k: h["meta"][k][()] for k in h["meta"]}
        dq = h["quality/simple/DQmask"][:]
        dq_desc = [d.decode() if isinstance(d, bytes) else str(d)
                   for d in h["quality/simple/DQDescriptions"][:]]
    return strain, meta, dq, dq_desc


def welch_median_psd(strain, fs, seg_sec=4.0):
    """Median Welch PSD over the given strain chunk."""
    nperseg = int(seg_sec * fs)
    freqs, psd = welch(strain, fs=fs, window="hann", nperseg=nperseg,
                       noverlap=nperseg // 2, detrend="constant",
                       return_onesided=True, scaling="density", average="median")
    return freqs, psd


def main():
    out = {}
    for det in ("H1", "L1"):
        strain, meta, dq, dq_desc = load_strain(det)
        print(f"{det}: meta GPSstart={meta['GPSstart']} Duration={meta['Duration']} "
              f"Detector={meta['Detector']} StrainChannel={meta['StrainChannel']}")
        print(f"  strain: n={len(strain)} fs={len(strain)/meta['Duration']:.1f} Hz "
              f"rms={np.nanstd(strain):.3e} max|h|={np.nanmax(np.abs(strain)):.3e}")
        # Data-quality mask: GWOSC simple DQmask, 1 s per entry. Event is at
        # 317.2 s into the file; check seconds 300-340.
        ev_sec = int(TRIGGER - GPS_START)
        print(f"  DQmask around event (s {ev_sec-10}..{ev_sec+10}): "
              f"{np.unique(dq[ev_sec-10:ev_sec+11], return_counts=True)}")
        print(f"  DQmask overall unique: {np.unique(dq, return_counts=True)}")
        print(f"  DQ descriptions: {dq_desc}")
        # Off-source PSD: file seconds 40..296 (256 s), ending 21 s before peak.
        off = strain[int(40 * FS):int(296 * FS)]
        freqs, psd = welch_median_psd(off, FS)
        print(f"  PSD: {len(freqs)} bins, {freqs[0]:.1f}-{freqs[-1]:.1f} Hz; "
              f"PSD(100 Hz)={psd[np.argmin(np.abs(freqs-100))]:.3e}, "
              f"PSD(150 Hz)={psd[np.argmin(np.abs(freqs-150))]:.3e}")
        # 8 s segment, Tukey window (Bilby default roll_off=0.2 s), then FFT.
        i0 = int(round((SEG_START - GPS_START) * FS))
        seg = strain[i0:i0 + int(SEG_DUR * FS)]
        assert len(seg) == int(SEG_DUR * FS), len(seg)
        assert not np.isnan(seg).any(), "NaN in analysis segment"
        window = tukey(len(seg), alpha=2 * 0.2 / SEG_DUR)
        window_factor = float(np.mean(window ** 2))
        fd = np.fft.rfft(seg * window) / FS  # Bilby nfft convention
        farr = np.fft.rfftfreq(len(seg), d=1.0 / FS)
        np.savez(f"data_{det}.npz", freqs=farr, fd_strain=fd,
                 psd_freqs=freqs, psd=psd, seg_start=SEG_START,
                 trigger=TRIGGER, fs=FS, duration=SEG_DUR,
                 window_factor=window_factor, roll_off_s=0.2)
        out[det] = {"file": FILES[det], "seg_start_gps": SEG_START,
                    "trigger_gps": TRIGGER, "duration_s": SEG_DUR,
                    "f_min": F_MIN, "f_max": F_MAX, "fs_hz": FS,
                    "dq_around_event": [int(x) for x in np.unique(dq[ev_sec-10:ev_sec+11])],
                    "psd_100Hz": float(psd[np.argmin(np.abs(freqs-100))]),
                    "psd_150Hz": float(psd[np.argmin(np.abs(freqs-150))]),
                    "seg_rms": float(np.std(seg))}
    json.dump(out, open("data_summary.json", "w"), indent=2)
    print("Wrote data_H1.npz, data_L1.npz, data_summary.json")


if __name__ == "__main__":
    main()
