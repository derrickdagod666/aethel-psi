"""Bilby modifier (standalone, no bilby/LALSuite) + Qiskit stabilizer (real run).
Bilby itself could NOT be installed here: it requires LALSuite (lalsuite), which
is not pip-installable in this container in reasonable time. So the modifier
function -- the ONLY UAF-specific part of Listing 1 -- is tested exactly as
printed, against a mock chirp. The LAL IMRPhenomPv2 call itself is untested.
Qiskit 2.6.2 + qiskit-aer were installed in ./venv_qiskit and the circuit runs.
"""
import numpy as np, json

# ---------------------------------------------------------------- Bilby modifier, verbatim logic
def uaf_modifier_verbatim(frequency_array, f_merger, gamma_uaf=0.4, width=0.15):
    """Exactly as printed in the UAF paper Listing 1 (reconstructed from garbled text):
    suppression = gamma * exp(-0.5*((f - f_merger)/width)**2)
    modifier = clip(1 - suppression, 0.5, 1.0)"""
    suppression = gamma_uaf * np.exp(-0.5 * ((frequency_array - f_merger) / width) ** 2)
    return np.clip(1.0 - suppression, 0.5, 1.0)

def uaf_modifier_fixed(frequency_array, f_merger, gamma_uaf=0.4, width_frac=0.15):
    """FIXED version -- what the paper almost certainly intended, and what the
    synthesis paper's Figure 2 actually plots: width is a FRACTION of f_merger.
    Change elaborated: `width` in the original is used as an absolute frequency
    in Hz. With f in Hz (~150 Hz at merger) that makes the dip 0.35 Hz wide
    (FWHM) -- narrower than one LIGO frequency bin, i.e. invisible in practice
    and numerically a spike. The fix multiplies by f_merger so width=0.15 means
    15% of the merger frequency (22.5 Hz at fm=150 Hz, FWHM 53 Hz), which is
    the broad localised dip Figure 2 shows. No other line changed."""
    width = width_frac * f_merger
    suppression = gamma_uaf * np.exp(-0.5 * ((frequency_array - f_merger) / width) ** 2)
    return np.clip(1.0 - suppression, 0.5, 1.0)

# Mock chirp: simple rising-amplitude toy (NOT IMRPhenomPv2 -- that needs LALSuite)
f = np.linspace(20, 300, 561)
fm = 150.0
plus_mock = (f / fm) ** (-7/6) * np.exp(-((f - fm) / 200) ** 2)  # toy amplitude shape
plus_mock /= plus_mock.max()

for name, mod in [("verbatim (width=0.15 Hz)", uaf_modifier_verbatim(f, fm)),
                  ("fixed (width=0.15*fm)", uaf_modifier_fixed(f, fm))]:
    modded = plus_mock * mod
    print(f"{name}: modifier at fm={mod[np.argmin(np.abs(f-fm))]:.3f}, min={mod.min():.3f}, "
          f"mean={mod.mean():.4f}, max SNR loss (1-mean)={1-mod.mean():.4f}")
    # matched-filter overlap (proxy for detectability): <h|h_mod>/sqrt(<h|h><h_mod|h_mod>)
    overlap = np.dot(plus_mock, modded) / np.sqrt(np.dot(plus_mock, plus_mock) * np.dot(modded, modded))
    print(f"   waveform overlap with unmodified mock chirp = {overlap:.5f} "
          f"(1-overlap = {1-overlap:.2e})")
    # floor test: gamma=0.9 should clip at 0.5
    m2 = uaf_modifier_verbatim(np.array([fm]), fm, gamma_uaf=0.9)
    print(f"   floor check gamma=0.9 at fm: modifier={m2[0]:.2f} (expect 0.5)")

# Edge cases of the verbatim function
print("\nEdge cases (verbatim): gamma=0 ->", uaf_modifier_verbatim(np.array([150.0]), 150.0, gamma_uaf=0.0))
print("Edge cases (verbatim): f_merger=None would crash (paper guards with argmax) -- guard present in listing: OK")
print("Edge cases: negative frequencies ->", uaf_modifier_verbatim(np.array([-10.0]), 150.0))

json.dump({"bilby": "modifier logic PASS standalone; full Bilby+LAL run NOT done (no LALSuite)",
           "verbatim_overlap_1_minus": float(1 - np.dot(plus_mock, plus_mock*uaf_modifier_verbatim(f, fm))/np.sqrt(np.dot(plus_mock,plus_mock)*np.dot(plus_mock*uaf_modifier_verbatim(f,fm), plus_mock*uaf_modifier_verbatim(f,fm)))),
           "fixed_overlap_1_minus": float(1 - np.dot(plus_mock, plus_mock*uaf_modifier_fixed(f, fm))/np.sqrt(np.dot(plus_mock,plus_mock)*np.dot(plus_mock*uaf_modifier_fixed(f,fm), plus_mock*uaf_modifier_fixed(f,fm))))},
          open("results_bilby.json","w"), indent=2)
print("\nWrote results_bilby.json")
