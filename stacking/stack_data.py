"""6a stacking — data preparation for two additional REAL events.

Generic adaptation of gate3_231123_data.py (same protocol):
  - 8 s segment, start GPS = trigger - 6 s, Tukey roll_off=0.2 s,
    sampling 4096 Hz.
  - PSD: Welch median over 256 s off-source (file seconds 40-296),
    4 s Hann, 50% overlap. No on-source data enter the PSD.
  - Band prepared 20-896 Hz; the analysis band / notches are decided
    from OFF-SOURCE diagnostics below, before any gamma likelihood.

Events (pre-registered fixed points, from published medians):
  GW150914_095045: GPS 1126259462.42 (PRL 116, 061102); source
    m1=36.2, m2=29.1 Msun, z=0.09 -> det 39.5/31.7, total det 71.2;
    DL=420 Mpc; chi_eff ~ -0.06 -> aligned spins a1=a2=0 (reduced
    design, as GW250114); network SNR 24.
    f_merger pre-registered = 150 * 71.5/71.2 = 151 Hz.
    Strain: GWOSC O1 4 kHz, GPS start 1126256640, via
    https://gwosc.org/api/v2/events/GW150914/strain-files
  GW200129_065458: GPS 1264316116.43 (GWTC-3); source m1=34.5,
    m2=29.0 Msun, z=0.18 -> det 40.7/34.2, total det 74.9;
    DL=890 Mpc; chi_eff=0.11 -> aligned spins a1=a2=0.11; SNR 26.8.
    f_merger pre-registered = 150 * 71.5/74.9 = 143 Hz.
    Strain: GWOSC O3b 4 kHz, GPS start 1264312320, via
    https://gwosc.org/api/v2/events/GW200129_065458/strain-files
    H1+L1 only (Virgo file exists but is not used, as in the
    GW250114/GW231123 analyses).
  Sky (ra, dec): NOT published as medians in the GWOSC parameters API
    (it gives only sky_area). Determined in the profile script by a
    coarse GR matched-filter maximisation over a fixed grid — a
    nuisance maximisation exactly analogous to the theta_jn scan in
    the earlier profiles, done at gamma=0 before any gamma curve.
"""
import json
import os
import sys
import h5py
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import tukey

EVENTS = {
    "GW150914": dict(
        trigger=1126259462.42, gps_start=1126256640,
        files={"H1": "H-H1_LOSC_4_V1-1126256640-4096.hdf5",
               "L1": "L-L1_LOSC_4_V1-1126256640-4096.hdf5"},
        f_merger=151.0, mass_1_det=39.5, mass_2_det=31.7,
        dl_mpc=420.0, a_1=0.0, a_2=0.0, snr_published=24.0,
        # L1 O1 file has NaN strain in file seconds 0-189 (detector not
        # observing); PSD region moved to 200-456 s, still 256 s and far
        # off-source (event at 2822 s). H1 uses the same region.
        psd_start_s=200.0),
    "GW200129": dict(
        trigger=1264316116.43, gps_start=1264312320,
        files={"H1": "H-H1_GWOSC_O3b_4KHZ_R1-1264312320-4096.hdf5",
               "L1": "L-L1_GWOSC_O3b_4KHZ_R1-1264312320-4096.hdf5"},
        f_merger=143.0, mass_1_det=40.7, mass_2_det=34.2,
        dl_mpc=890.0, a_1=0.11, a_2=0.11, snr_published=26.8,
        psd_start_s=40.0),
}
FS = 4096.0
SEG_DUR = 8.0
F_MIN, F_MAX = 20.0, 896.0


def load_strain(path):
    with h5py.File(path, "r") as h:
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
    ev_name = sys.argv[1]
    cfg = EVENTS[ev_name]
    trig, gps0 = cfg["trigger"], cfg["gps_start"]
    seg_start = trig - 6.0
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ev_name)
    os.makedirs(outdir, exist_ok=True)
    out = {"event": ev_name, "config": {k: v for k, v in cfg.items() if k != "files"}}
    for det in ("H1", "L1"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "events", cfg["files"][det])
        strain, meta, dq, dq_desc = load_strain(path)
        print(f"{ev_name} {det}: meta GPSstart={meta['GPSstart']} "
              f"Duration={meta['Duration']} Detector={meta['Detector']}")
        print(f"  strain: n={len(strain)} rms={np.nanstd(strain):.3e} "
              f"max|h|={np.nanmax(np.abs(strain)):.3e} "
              f"NaNs={int(np.isnan(strain).sum())}")
        ev_sec = int(trig - gps0)
        print(f"  DQmask around event (s {ev_sec-10}..{ev_sec+10}): "
              f"{np.unique(dq[ev_sec-10:ev_sec+11], return_counts=True)}")
        print(f"  DQmask overall unique: {np.unique(dq, return_counts=True)}")
        off = strain[int(cfg.get("psd_start_s", 40.0) * FS):
                      int((cfg.get("psd_start_s", 40.0) + 256) * FS)]
        freqs, psd = welch_median_psd(off, FS)
        print(f"  PSD: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz; "
              f"PSD(35)={psd[np.argmin(np.abs(freqs-35))]:.3e}, "
              f"PSD(100)={psd[np.argmin(np.abs(freqs-100))]:.3e}, "
              f"PSD(150)={psd[np.argmin(np.abs(freqs-150))]:.3e}")
        # Off-source line diagnostic: per-bin <d|d> in 8 s off-source
        # segments (file seconds 400, 800, 1200, 2000), band 20-896 Hz.
        # A line that dominates <d|d> off-source is instrumental and a
        # notch candidate; decision recorded here, before any gamma work.
        df = freqs[1] - freqs[0]
        seg_len = int(SEG_DUR * FS)
        window = tukey(seg_len, alpha=2 * 0.2 / SEG_DUR)
        for start_s in (400, 800, 1200, 2000):
            s = strain[int(start_s * FS):int(start_s * FS) + seg_len]
            fd = np.fft.rfft(s * window) / FS
            farr = np.fft.rfftfreq(seg_len, d=1.0 / FS)
            m = (farr >= 20) & (farr <= 896)
            contrib = 4 * (np.abs(fd[m]) ** 2 / np.interp(farr[m], freqs, psd)) * df
            top = np.argsort(contrib)[-5:][::-1]
            tot = float(contrib.sum())
            print(f"  off-source seg @{start_s}s: <d|d> 20-896Hz={tot:.0f}; "
                  f"top bins: " + ", ".join(
                      f"{farr[m][i]:.1f}Hz:{contrib[i]:.0f}" for i in top))
        # Event segment <d|d> by band (diagnostic only)
        i0 = int(round((seg_start - gps0) * FS))
        seg = strain[i0:i0 + seg_len]
        assert len(seg) == seg_len, len(seg)
        assert not np.isnan(seg).any(), "NaN in analysis segment"
        fd = np.fft.rfft(seg * window) / FS
        farr = np.fft.rfftfreq(seg_len, d=1.0 / FS)
        for lo, hi in [(20, 50), (50, 100), (100, 200), (200, 400), (400, 896)]:
            m = (farr >= lo) & (farr < hi)
            dd = float(4 * np.sum((np.abs(fd[m]) ** 2 / np.interp(farr[m], freqs, psd))) * df)
            print(f"  event seg <d|d> {lo}-{hi}Hz = {dd:.0f}")
        np.savez(os.path.join(outdir, f"data_{det}.npz"), freqs=farr,
                 fd_strain=fd, psd_freqs=freqs, psd=psd,
                 seg_start=seg_start, trigger=trig, fs=FS,
                 duration=SEG_DUR, window_factor=float(np.mean(window ** 2)),
                 roll_off_s=0.2)
        out[det] = {"file": cfg["files"][det], "seg_start_gps": seg_start,
                    "trigger_gps": trig, "duration_s": SEG_DUR,
                    "f_min": F_MIN, "f_max": F_MAX, "fs_hz": FS,
                    "dq_around_event": [int(x) for x in np.unique(dq[ev_sec-10:ev_sec+11])],
                    "psd_35Hz": float(psd[np.argmin(np.abs(freqs-35))]),
                    "psd_100Hz": float(psd[np.argmin(np.abs(freqs-100))]),
                    "psd_150Hz": float(psd[np.argmin(np.abs(freqs-150))]),
                    "seg_rms": float(np.std(seg))}
    json.dump(out, open(os.path.join(outdir, "data_summary.json"), "w"), indent=2)
    print(f"Wrote {outdir}/data_H1.npz, data_L1.npz, data_summary.json")


if __name__ == "__main__":
    main()
