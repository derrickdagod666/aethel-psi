# Release notes — v0.1.0 (2026-11-07)

First versioned release of `aethel-psi`. The code is unchanged from the
first commit (9803bd4b) plus the Third Revised Edition PDF (cbb16405);
v0.1.0 packages it — it does not add new science.

## What is in v0.1.0

- **Pip-installable package**: `pip install -e .` (or `pip install .`),
  then `from aethel_psi import uaf_blunted_waveform, dip_modifier`.
  Pinned Bilby 2.8.2 / Qiskit 2.6.2 — the versions used for the
  verified runs.
- **Tests + CI**: `pytest tests/` runs the waveform modifier check
  (1.000 / 0.600 / 0.500 at `f_merger` for γ = 0 / 0.4 / 0.9) and the
  Qiskit stabilizer test (PASS 4/4 live). GitHub Actions runs both
  on every push (Python 3.12).
- Everything from the first commit: Gate 3 pipeline (5 real events),
  validation suites v2–v5 (+ v6 spin-γ), ACT DR6 ripple test,
  stacking/Fisher forecast, Qiskit demo, Drive exploratory code,
  and the Third Revised Edition PDF in `docs/`.

## What v0.1.0 honestly reports

- **γ = 0.4 is excluded** (GW250114, real strain): profile
  ΔlnL = −55.2; full dynesty PE γ median 0.0235, 95% upper 0.0813,
  ln B = −3.06 ± 0.23 vs GR. Prediction 1 is **γ ≲ 0.08 (95%)**.
- **Spin-γ hypothesis CLOSED** (v6): both χ_eff and χ_p branches FAIL.
- **ACT DR6 joint: no ripple**, global p = 0.105.
- v5: PASS 25, FAIL 18, OPEN 6, INFO 8. β_A = 1/14 FITTED, not
  derived; Λ_eff ≡ 0 DROPPED, not proven; phantom crossing not
  reproduced with the stated V(Ψ); ξ = −1/6 is the one closed
  derivation (under a stated D_Ψ, E = −Ψ²).

## What is NOT in v0.1.0

- ApothegmSovereignEngine and the VQE drug-discovery app — source
  not found in Drive or workspace (2026-11-07).
- Aethelred Agent code — Drive copy contains hardcoded API keys;
  deliberately excluded (README + `.env.example` only).
- GWOSC strain / ACT DR6 data files — download via the run scripts.
- No resumed GW231123 precessing PE, no spin-marginalised GW200129
  PE, no replacement γ(M)/γ(spin) mechanism.
- No peer review, no Nobel claim, no valuation. The theory itself
  is not for sale; the pipeline is the product.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install .
python -m pytest tests/ -v
```

```python
from aethel_psi import uaf_blunted_waveform
# uaf_blunted_waveform(freq, mass_1, mass_2, luminosity_distance,
#                      theta_jn, phase, gamma=0.4, f_merger=150.0, ...)
# f_merger MUST be passed explicitly; width = 0.15 * f_merger; floor 0.5.
```

MIT licence. Cite the Third Revised Edition (Derrick Irons, The
Aethelred Institute, Nov 2026, ORCID 0009-0001-7064-6611) and the
results as measured — γ ≲ 0.08, not γ = 0.4.
