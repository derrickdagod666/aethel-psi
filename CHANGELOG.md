# Changelog

All notable changes to the `aethel-psi` repo. The theory in the paper is
**not validated**; this log records what the code does and what was
measured, including the FAILs. Nothing here is a Nobel claim, a
peer-reviewed result, or a valuation of the theory.

## [0.1.0] — 2026-11-07 (first versioned release)

First pip-installable release. The code itself is unchanged from the
first commit (9803bd4b, 49 text files) plus the Third Revised Edition
PDF (cbb16405); v0.1.0 packages it, it does not add new science.

### Added

- **Pip-installable package** (`pyproject.toml`, `src/aethel_psi/`):
  `from aethel_psi import uaf_blunted_waveform, dip_modifier` — the
  fixed Bilby UAF waveform (Gaussian dip at `f_merger`, depth γ,
  width `0.15 × f_merger`, floor 0.5) extracted verbatim in logic
  from `gw-pipeline/uaf_bilby_plugin.py`, decoupled from
  `gate3_common.py` so it imports without GWOSC strain data.
  Pinned: **Bilby 2.8.2, Qiskit 2.6.2** (the versions used for the
  verified runs, 2026-11-07).
- **Tests** (`tests/`): waveform modifier check (1.000 / 0.600 /
  0.500 at `f_merger` for γ = 0 / 0.4 / 0.9) and the Qiskit
  stabilizer test in pytest form (mirrors
  `qiskit-stabilizer/test_qiskit.py`, PASS 4/4 live).
- **CI** (`.github/workflows/ci.yml`): on every push, Python 3.12,
  `pip install -e ".[dev]"`, then `pytest tests/ -v`.
- **Package quickstart** in `README.md`.

### Measured results shipped in v0.1.0 (all previously verified, 2026-11-07)

- γ = 0.4 **excluded** on real GW250114 strain: profile ΔlnL = −55.2
  vs GR; full dynesty PE γ median 0.0235, 95% upper 0.0813,
  P(γ > 0.4) = 0/3697, ln B = −3.06 ± 0.23 (free-γ UAF vs GR).
  Prediction 1 is **γ ≲ 0.08 (95%)** for GW250114-like mergers.
- 5-event stack excludes γ = 0.4 (ΔlnL = −52.4, Wilks 95% γ < 0.173).
- Spin-γ hypothesis **CLOSED** (v6, real LVK spin samples): both the
  χ_eff and χ_p branches FAIL. No replacement γ(M)/γ(spin) mechanism.
- ACT DR6 joint TT+TE+EE: no ripple, global p = 0.105.
- Validation suites: v5 PASS = 25, FAIL = 18, OPEN = 6, INFO = 8;
  v6 PASS = 9, FAIL = 6, OPEN = 2. β_A = 1/14 is FITTED, not derived;
  Λ_eff ≡ 0 was DROPPED, not proven; the phantom crossing at
  a_c = 0.476 is not reproduced with the stated V(Ψ). The one closed
  derivation is ξ = −1/6, under a stated D_Ψ (E = −Ψ², scalar sector).
- Qiskit stabilizer: PASS 4/4, GHZ 4096/4096 in all runs.

### NOT in v0.1.0

- **ApothegmSovereignEngine** (React/TypeScript) and the
  **Dash/Plotly/PennyLane VQE drug-discovery app**: no source files
  were found in Google Drive or the local workspace on 2026-11-07
  (Drive searched by name and full text). They cannot be shipped
  until the source is located.
- **Aethelred Agent code**: exists in Google Drive but contains
  hardcoded API keys; deliberately not copied. `aethelred-agent/`
  holds a README + `.env.example` only, explaining how to sanitize
  it first. Never commit API keys.
- **GWOSC strain files and ACT DR6 data files**: public data,
  downloaded by the run scripts — see the component READMEs.
- **No new science**: no replacement γ(M)/γ(spin) mechanism, no
  resumed GW231123 precessing PE (died in a VM reboot; checkpoint
  exists, resume untested), no spin-marginalised GW200129 PE.
- **No peer review, no Nobel claim, no valuation of the theory.**
  The theory itself is not for sale.
