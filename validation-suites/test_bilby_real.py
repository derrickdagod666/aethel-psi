"""Real Bilby test of the UAF blunted waveform (Listing 1, reconstructed).
Run with ./venv_qiskit/bin/python test_bilby_real.py
Requires bilby + lalsuite (installed into ./venv_qiskit via pip).

Reconstruction notes (garbled PDF listing -> working code):
- The PDF listing's function signature and body were reassembled from the
  prose + fragments: uaf_blunted_waveform(frequency_array, mass_1, mass_2,
  luminosity_distance, theta_jn, phase, a_1, a_2, tilt_1, tilt_2, phi_12,
  phi_jl, gamma_uaf=0.4, f_merger=None, width=0.15, **kwargs).
- Baseline call in the paper: bilby.gw.source.lal_binary_black_hole(...)
  with default waveform_approximant IMRPhenomPv2. Kept exactly.
- f_merger default: frequency of max |plus| polarisation. Kept.
- BUG FIX (the one material change): `width` is used as an absolute frequency
  in the printed code, but the paper's Figure 2 and the value 0.15 only make
  sense as a FRACTION of f_merger. Both versions are run below so the
  difference is measured, not asserted. The fixed version uses
  width_hz = width * f_merger.
- The paper's `**kwargs` are forwarded to lal_binary_black_hole; in bilby 2.8
  the waveform call signature is lal_binary_black_hole(frequency_array,
  mass_1, mass_2, luminosity_distance, a_1, a_2, tilt_1, tilt_2, phi_12,
  phi_jl, theta_jn, phase, **kwargs) -- argument ORDER differs from the
  listing, so keyword arguments are used throughout to avoid a silent swap.
"""
import numpy as np, json
import bilby

def uaf_blunted_waveform(frequency_array, mass_1, mass_2, luminosity_distance,
                         theta_jn, phase, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                         gamma_uaf=0.4, f_merger=None, width=0.15,
                         width_is_fractional=False, **kwargs):
    polarizations = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, a_1=a_1, a_2=a_2,
        tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl,
        theta_jn=theta_jn, phase=phase, **kwargs)
    if f_merger is None:
        f_merger = frequency_array[np.argmax(np.abs(polarizations['plus']))]
    w = width * f_merger if width_is_fractional else width
    suppression = gamma_uaf * np.exp(-0.5 * ((frequency_array - f_merger) / w) ** 2)
    modifier = np.clip(1.0 - suppression, 0.5, 1.0)
    for mode in ('plus', 'cross'):
        polarizations[mode] = polarizations[mode] * modifier
    return polarizations, f_merger, modifier

freq = np.arange(20, 512, 0.25)  # Hz, LIGO-like grid
args = dict(mass_1=34.0, mass_2=32.0, luminosity_distance=1000.0, theta_jn=0.4,
            phase=0.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
# GW250114-like masses (34 + 32 Msun), the loudest event (SNR ~80) -- used as a
# representative high-SNR template only; NO real detector data are used here.
base = bilby.gw.source.lal_binary_black_hole(freq, **args)
print("Baseline IMRPhenomPv2: max|plus| at f =", freq[np.argmax(np.abs(base['plus']))], "Hz")
print("Baseline max|plus| =", np.abs(base['plus']).max())

results = {}
for label, frac in [('verbatim width=0.15 Hz', False), ('fixed width=0.15*fm', True)]:
    pol, fm, mod = uaf_blunted_waveform(freq, gamma_uaf=0.4, width_is_fractional=frac, **args)
    # matched-filter overlap between baseline and modified (white-noise proxy)
    h0 = base['plus']; h1 = pol['plus']
    overlap = np.vdot(h0, h1) / np.sqrt(np.vdot(h0, h0) * np.vdot(h1, h1))
    # optimal SNR loss at fixed distance: sqrt(<h1|h1>/<h0|h0>)
    snr_ratio = np.sqrt(np.vdot(h1, h1) / np.vdot(h0, h0))
    print(f"\n{label}: inferred f_merger={fm:.2f} Hz, modifier at fm={mod[np.argmin(np.abs(freq-fm))]:.3f}, "
          f"min={mod.min():.3f}")
    print(f"  overlap with GR template = {overlap.real:.6f} (mismatch {1-overlap.real:.2e})")
    print(f"  SNR ratio (modified/GR) = {snr_ratio.real:.4f} -> SNR loss {(1-snr_ratio.real)*100:.2f}%")
    print(f"  cross polarisation also modified: {np.allclose(pol['cross'], base['cross']*mod)}")
    results[label] = {'f_merger_Hz': float(fm), 'modifier_at_merger': float(mod[np.argmin(np.abs(freq-fm))]),
                      'overlap': float(overlap.real), 'mismatch': float(1-overlap.real),
                      'snr_ratio': float(snr_ratio.real)}
    # gamma sweep on the fixed version: how big must gamma be for a 1% mismatch?
    if frac:
        for g in [0.4, 0.6, 0.8, 1.0]:
            p2, _, _ = uaf_blunted_waveform(freq, gamma_uaf=g, width_is_fractional=True, **args)
            ov = np.vdot(h0, p2['plus']) / np.sqrt(np.vdot(h0, h0) * np.vdot(p2['plus'], p2['plus']))
            print(f"  fixed-version gamma={g}: mismatch={1-ov.real:.4f}, "
                  f"modifier at fm={1-g:.2f} (floor clips at 0.5 for gamma>0.5)")
json.dump(results, open('results_bilby_real.json', 'w'), indent=2)
print("\nWrote results_bilby_real.json")
print("NOTE: this tests the waveform GENERATOR only. No LVK strain data were")
print("downloaded or fitted; a real Gate-3 test needs GWOSC posterior/strain data.")

# --- Addendum: the auto-f_merger bug -------------------------------------
# argmax(|plus|) returned 40 Hz, i.e. the low-frequency inspiral edge, NOT the
# merger. For a 66 Msun binary the ISCO GW frequency is:
M_tot = 34.0 + 32.0
f_isco_gw = 4400.0 / M_tot  # Hz, approx: c^3/(3*sqrt(6)*pi*G*M)
print(f"\nADDENDUM: total mass {M_tot} Msun -> f_ISCO(GW) ~ {f_isco_gw:.1f} Hz; "
      f"GW150914-like merger/ringdown ~150 Hz. argmax gave 40 Hz: WRONG by ~4x.")
print("Re-running fixed version with f_merger supplied explicitly at 150 Hz:")
for g in [0.4]:
    p2, fm2, _ = uaf_blunted_waveform(freq, gamma_uaf=g, f_merger=150.0,
                                      width_is_fractional=True, **args)
    ov = np.vdot(base['plus'], p2['plus']) / np.sqrt(np.vdot(base['plus'], base['plus']) * np.vdot(p2['plus'], p2['plus']))
    sr = np.sqrt(np.vdot(p2['plus'], p2['plus']) / np.vdot(base['plus'], base['plus']))
    print(f"  f_merger=150 Hz, gamma={g}: mismatch={1-ov.real:.4f}, SNR ratio={sr.real:.4f}")
    print(f"  (vs dip-at-40Hz mismatch 0.0170: misplacing the dip changes the observable by ~2x)")
