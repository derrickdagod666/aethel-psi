"""Gate 3 — GW231123 data preparation (REAL GWOSC strain).

Event: GW231123_135430, 23 Nov 2023, 13:54:30 UTC.
GPS peak (geocentre): 1384782888.7 s (GWOSC O4 Discovery Paper Data
Release; NRSur t_pol_peak 1384782888.5998, strain peak 1384782888.6191
H1 / 1384782888.6142 L1 — spread ~0.1 s, covered by the dt scan in
validation/profile, which is a nuisance scan, not a gamma fit).
GWOSC O4 Discovery Paper Data Release (public release 14 Jul 2025,
updated PE 24 Oct 2025); strain begins GPS 1384779776, spans 4096 s,
peak 3112.7 s into the files, H1 + L1, 4 kHz:
  H-H1_GWOSC_O4a_4KHZ_R1-1384779776-4096.hdf5  md5 04e1a72b6cab273e3f4ef473e9186ee4
  L-L1_GWOSC_O4a_4KHZ_R1-1384779776-4096.hdf5  md5 ca94a51e4d7105b353c5fe4bfe9a5a1f
  URLs: https://gwosc.org/archive/data/O4a_4KHZ_R1/1384722432/<file>
  (directory = file GPS start - 57344, as for GW250114's O4b files;
  both URLs returned HTTP 200 and the files were downloaded and
  md5-checked; the GWOSC strain-files API returned 0 results for
  this event's catalog versions, so the O4 Discovery Paper release
  is the source used, as documented on https://gwosc.org/o4_eventdata_docs/)

Published parameters (LVK discovery paper, arXiv:2507.08219, combined
samples, Table 2; GWOSC detail page O4_Discovery_Papers v1):
  source m1=137 (+22/-17), m2=103 (+20/-52) Msun, total 238 (+28/-49),
  final 225 (+26/-43) Msun, DL=2200 (+1900/-1500) Mpc, z=0.39 (+0.27/-0.24),
  chi1=0.90 (+0.10/-0.19), chi2=0.80 (+0.20/-0.51), chi_eff=0.31 (+0.24/-0.39),
  chi_p=0.77 (+0.17/-0.19), inclination theta_JN=1.4 (+0.8/-1.1) rad,
  network matched-filter SNR 22.6 (+0.2/-0.3) (updated Nov 2025: 20.7).
  Signal: ~5 cycles, 30-80 Hz, duration ~0.1 s.
  Sky (RA, dec) is NOT in the paper's Table 2. Fixed sky used here:
  RA=3.02 rad, dec=-0.02 rad (TEOB max-likelihood, Jan et al. Table III,
  an independent reanalysis — NOT the LVK combined sky; a second-sky
  sensitivity check is run in the profile script).
  KNOWN DATA ISSUE (paper Sec. III): a glitch in Hanford 1.7-1.1 s
  before the event, 15-30 Hz. It lies inside the 8 s analysis segment.
  A f_min=30 Hz sensitivity check (excluding the glitch band) is run
  in the profile script; the glitch is NOT removed post-hoc.

Analysis choices (pre-registered here, before any likelihood evaluation):
  - 8 s segment, start GPS = trigger - 6 s, 20-256 Hz (paper Fig. 1
    bandpass 20-256 Hz), sampling 4096 Hz. Same segment design as
    GW250114 for comparability.
  - Tukey window, roll_off=0.2 s, as in gate3_data.py (Bilby default).
  - PSD: Welch median over 256 s off-source (file seconds 40-296,
    ending ~2817 s before the peak — the event is 3112 s into the file,
    so the PSD region is far off-source), 4 s Hann, 50% overlap.
  - Fixed source parameters: detector-frame m1=190.4, m2=143.2 Msun
    (source 137/103 x (1+0.39)); aligned spins a1=a2=0.31 to match
    chi_eff=0.31 (published spins 0.90/0.80 are tilted; setting them
    aligned would give chi_eff=0.86, which is wrong. Precession
    chi_p=0.77 and higher modes are UNMODELLED by IMRPhenomPv2 —
    the dominant limitation for this event, stated in the report).
    DL=2200 Mpc, theta_JN=1.4 rad, RA=3.02, dec=-0.02.
  - f_merger central = 35 Hz (see DESIGN.md): scaled from GW250114
    (150 Hz x 71.5/333.6 = 32 Hz) to the ringdown scale (~35-40 Hz
    for final mass 225 Msun source, 313 Msun detector, spin 0.84).
    width = 0.15 x f_merger = 5.25 Hz, FWHM 12.4 Hz.
    Sensitivity scans (pre-registered): f_merger 25/45 Hz,
    width_frac 0.10/0.25.
"""
import json
import h5py
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import tukey

TRIGGER = 1384782888.7  # GPS, geocentre peak (GWOSC)
GPS_START = 1384779776
FS = 4096.0
SEG_DUR = 8.0
SEG_START = TRIGGER - 6.0
F_MIN, F_MAX = 20.0, 256.0

FILES = {
    "H1": "H-H1_GWOSC_O4a_4KHZ_R1-1384779776-4096.hdf5",
    "L1": "L-L1_GWOSC_O4a_4KHZ_R1-1384779776-4096.hdf5",
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
        ev_sec = int(TRIGGER - GPS_START)
        print(f"  DQmask around event (s {ev_sec-10}..{ev_sec+10}): "
              f"{np.unique(dq[ev_sec-10:ev_sec+11], return_counts=True)}")
        print(f"  DQmask overall unique: {np.unique(dq, return_counts=True)}")
        print(f"  DQ descriptions: {dq_desc}")
        off = strain[int(40 * FS):int(296 * FS)]
        freqs, psd = welch_median_psd(off, FS)
        print(f"  PSD: {len(freqs)} bins, {freqs[0]:.1f}-{freqs[-1]:.1f} Hz; "
              f"PSD(35 Hz)={psd[np.argmin(np.abs(freqs-35))]:.3e}, "
              f"PSD(100 Hz)={psd[np.argmin(np.abs(freqs-100))]:.3e}")
        # Off-source line check in 20-256 Hz: per-bin |d|^2/S in 8 s off-source segments
        for lo, hi in [(20, 256)]:
            pass
        i0 = int(round((SEG_START - GPS_START) * FS))
        seg = strain[i0:i0 + int(SEG_DUR * FS)]
        assert len(seg) == int(SEG_DUR * FS), len(seg)
        assert not np.isnan(seg).any(), "NaN in analysis segment"
        window = tukey(len(seg), alpha=2 * 0.2 / SEG_DUR)
        window_factor = float(np.mean(window ** 2))
        fd = np.fft.rfft(seg * window) / FS
        farr = np.fft.rfftfreq(len(seg), d=1.0 / FS)
        np.savez(f"data_{det}.npz", freqs=farr, fd_strain=fd,
                 psd_freqs=freqs, psd=psd, seg_start=SEG_START,
                 trigger=TRIGGER, fs=FS, duration=SEG_DUR,
                 window_factor=window_factor, roll_off_s=0.2)
        out[det] = {"file": FILES[det], "seg_start_gps": SEG_START,
                    "trigger_gps": TRIGGER, "duration_s": SEG_DUR,
                    "f_min": F_MIN, "f_max": F_MAX, "fs_hz": FS,
                    "dq_around_event": [int(x) for x in np.unique(dq[ev_sec-10:ev_sec+11])],
                    "psd_35Hz": float(psd[np.argmin(np.abs(freqs-35))]),
                    "psd_100Hz": float(psd[np.argmin(np.abs(freqs-100))]),
                    "seg_rms": float(np.std(seg))}
    json.dump(out, open("data_summary.json", "w"), indent=2)
    print("Wrote data_H1.npz, data_L1.npz, data_summary.json")


if __name__ == "__main__":
    main()
