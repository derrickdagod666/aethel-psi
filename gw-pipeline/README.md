# gw-pipeline — Bilby UAF waveform + Gate 3 runs

## What is here

- `uaf_bilby_plugin.py` — canonical copy of the fixed waveform
  (identical to `gate3_pe.py`'s `uaf_blunted_waveform`).
- `gate3_pe.py`, `gate3_common.py`, `gate3_data.py`,
  `gate3_profile.py`, `gate3_validate.py` — GW250114 (the
  reference event).
- `gate3_231123_*.py` — GW231123 equivalents; `precessing_profile.py`
  (XPHM), `tphm_profile.py` (TPHM cross-check).
- `xphm_profile.py`, `tilt_check.py` — GW200129 XPHM profile and
  tilt-robustness check.

## The waveform

```python
def uaf_blunted_waveform(frequency_array, mass_1, mass_2,
                         luminosity_distance, theta_jn, phase, ...,
                         gamma=0.4, width_frac=0.15, f_merger=None, **kwargs):
    # f_merger MUST be passed explicitly (FIX 2)
    # width = width_frac * f_merger  (FIX 1: not 0.15 Hz absolute)
    # modifier = max(1 - gamma*exp(-(f-f_merger)^2/(2*width^2)), floor=0.5)
```

Baseline is `bilby.gw.source.lal_binary_black_hole` with
`IMRPhenomPv2`. The modifier multiplies both polarisations.

**Verified live 2026-11-07** (venv with Bilby 2.8.2):

| γ | modifier at f_merger = 150 Hz |
|---|---|
| 0.0 | 1.000 |
| 0.4 | 0.600 |
| 0.9 | 0.500 (floor) |

## Run instructions

```bash
pip install -r ../requirements.txt   # bilby 2.8.2, lalsuite, numpy, scipy
# 1. Download public GWOSC strain (GW250114 example; md5s in gate3_data.py):
#    https://gwosc.org/api/v2/events/GW250114_082203/strain-files
#    H-H1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5  md5 f0f6c16c419282a26fa7a57810f715ac
#    L-L1_GWOSC_O4b_4KHZ_R1-1420877824-4096.hdf5  md5 bea4dcf16e5dd9d64e425d66f97d5cc6
# 2. Prepare data + PSDs (8 s segment, Tukey 0.2 s, 20-896 Hz,
#    Welch-median PSD from 256 s off-source, notches 504-508 / 510-514 Hz):
python gate3_data.py
# 3. Profile likelihood:
python gate3_profile.py
# 4. Full dynesty PE (nlive=1000, dlogz=0.1, rwalk, walks=25, npool=4;
#    tens of minutes per model on 4 cores):
python gate3_pe.py
```

## Results this pipeline produced (real data)

- GW250114: recovered MF SNR 78.2 (LVK ~80). Profile ΔlnL(0.4) =
  −55.2; full PE γ median 0.0235, 95% upper 0.0813,
  P(γ>0.4) = 0/3697, ln B = −3.06 ± 0.23.
- GW231123: profile ΔlnL(0.4) = +13.0 (Pv2) / +23.0 (XPHM
  precessing); NOT a detection (saturates at the 0.5 floor,
  killed at f_merger = 25 Hz). Full precessing PE **died in a VM
  reboot ~03:46 UTC 2026-11-07** — no posterior/Bayes factor
  exists; checkpoint exists, resume untested.
- GW200129: Pv2 ΔlnL(0.4) = −15.7, but XPHM tilt scan spans
  −16.4…+8.7 — exclusion is **OPEN**, needs a spin-marginalised PE.
- GW150914: +0.1 (flat). GW190412: +5.4 (Pv2 lacks its higher modes).
- 5-event stack: ΔlnL(0.4) = −52.4, Wilks 95% γ < 0.173.

## Known limitations (do not hide these)

- Masses/spins fixed at published medians in the PE; sky fixed;
  no calibration marginalisation. Single approximant per run
  (IMRPhenomPv2 unless XPHM/TPHM named).
- The +7 ms offset between the published NRSur7dq4 peak time and
  the IMRPhenomPv2 best time is a model-definition difference;
  the geocent_time prior (±0.05 s; ±0.15 s for GW231123 XPHM)
  covers it.
- Data files (`.hdf5`, `.npz`) are NOT in the repo — download
  them as above. Never commit strain data.
