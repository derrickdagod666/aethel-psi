# aethel-psi

**The Aethel-Psi Synthesis — code, tests, and data-analysis pipeline.**
Derrick Irons / The Aethelred Institute · ORCID 0009-0001-7064-6611

This repository is the *code* behind the paper
[`docs/aethel-psi-synthesis-third-revised-edition.pdf`](docs/aethel-psi-synthesis-third-revised-edition.pdf)
(Third Revised Edition, Nov 2026, 21 pages). The theory in that paper is
**not validated** — this repo is honest about that, and the pipeline is
the product, not a validated theory.

## What this code actually does

- A **Bilby waveform plugin** (`gw-pipeline/`): the fixed UAF blunted
  waveform `uaf_blunted_waveform`, a Gaussian dip at the merger frequency
  with depth γ, width `0.15 × f_merger`, floor 0.5. Two bugs in the
  paper's printed listing are fixed here and documented in the code:
  width is a *fraction* of `f_merger`, and `f_merger` is passed
  explicitly (never `argmax(|h|)`).
- **Gate 3 on real LIGO/Virgo data**: profile-likelihood and full
  dynesty PE runs on GW250114, GW231123, GW200129, GW150914 and
  GW190412 from public GWOSC strain, plus a 4–5 event stack.
- **Validation suites v2–v5** (`validation-suites/`): falsifiable
  checks of every claim in the paper, with PASS / FAIL / OPEN counts.
- **ACT DR6 CMB ripple test** (`act-dr6/`): real DR6.02 TT+TE+EE
  bandpowers with the full covariance.
- **Qiskit stabilizer demo** (`qiskit-stabilizer/`): the paper's
  3-qubit repetition-code circuit, reconstructed for Qiskit 2.x.
- **Quantum-cognitive / PennyLane code from Drive** (`drive-code/`):
  earlier exploratory code, kept as-is, not production code.

## What it does NOT claim — read this first

- **γ = 0.4 is excluded.** On real GW250114 strain, γ = 0.4 gives
  ΔlnL = −55.2 vs GR (profile); the full dynesty PE gives γ median
  0.0235, 95% upper 0.0813, P(γ > 0.4) = 0/3697, ln B = −3.06 ± 0.23
  (free-γ UAF vs GR). **Prediction 1 is γ ≲ 0.08 (95%) for
  GW250114-like mergers.**
- **The 5-event stack excludes γ = 0.4**: ΔlnL = −52.4, Wilks 95%
  γ < 0.173. γ ≈ 0.009 (the γ(M) derivation) is untouched by the
  stack and needs ~20 events at SNR ~80 for 1σ, ~200 for 3σ.
- **The spin-γ hypothesis is closed** (v6, real LVK spin samples):
  both the χ_eff and χ_p branches FAIL. No replacement γ(M)/γ(spin)
  mechanism is in this repo.
- **ACT DR6 joint TT+TE+EE: no ripple.** Global p = 0.105,
  A95 = 1.33 × 10⁻². A ≈ 0.1 ripple excluded; a 0.005-scale ripple
  is not.
- **β_A = 1/14 is FITTED, not derived.** The Chern–Simons
  computation gives integer levels only; no level gives 1/14.
  **Λ_eff ≡ 0 is dropped, not proven.** The phantom crossing at
  a_c = 0.476 is **not reproduced** with the stated V(Ψ). The one
  closed derivation is **ξ = −1/6**, under a stated D_Ψ
  (E = −Ψ², scalar sector).
- **v5 final: PASS = 25, FAIL = 18, OPEN = 6, INFO = 8.**
  **v6: PASS = 9, FAIL = 6, OPEN = 2.** The FAILs are in the
  paper and in `AUDIT.md`. Nothing here is priced on γ = 0.4,
  and nothing here is a Nobel claim, a peer-reviewed result, or
  a valuation of the theory. A timestamp (arXiv/Zenodo) proves
  priority only.

## Quickstart (verified 2026-11-07)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # bilby 2.8.2, lalsuite, numpy, scipy, qiskit 2.6.2, qiskit-aer, sympy
```

```bash
# 1. Waveform modifier check (no data needed) — verified live:
#    modifier at f_merger = 1.000 / 0.600 / 0.500 for gamma = 0 / 0.4 / 0.9
python -c "import sys; sys.path.insert(0, 'gw-pipeline')"
python gw-pipeline/uaf_bilby_plugin.py  # import-only smoke test; see gw-pipeline/README.md

# 2. Qiskit stabilizer — verified live, PASS 4/4, GHZ 4096/4096 in all runs
python qiskit-stabilizer/test_qiskit.py

# 3. Validation suite v5 (full run; exit 1 is correct — it contains FAILs)
python validation-suites/aethel_psi_suite_v5.py
```

Real-data runs (Gate 3, stacking, ACT) need public data downloaded
first — see each component README. **No strain data, no venvs, and
no credentials are in this repo** (see `.gitignore`).

## Repo layout

```
aethel-psi/
├── gw-pipeline/         Bilby UAF waveform + Gate 3 PE/profile scripts (5 events)
├── qiskit-stabilizer/   3-qubit stabilizer demo (Qiskit 2.x)
├── validation-suites/   Falsifiable suites v2–v5 + core/Bilby tests
├── act-dr6/             Real ACT DR6.02 TT+TE+EE ripple test (sacc/cobaya)
├── stacking/            4–5 event profile stack + Fisher forecast
├── drive-code/          Earlier PennyLane / quantum-cognitive code (as-is)
├── aethelred-agent/     Agent backend — README + .env.example ONLY (see below)
├── docs/                Third Revised Edition PDF (reference only)
└── requirements.txt     Pinned versions used for the verified runs
```

## ⚠️ aethelred-agent — NOT included as code

The Aethelred Agent TypeScript backend (Express + Gemini 2.5 Pro +
SQLite + WebSocket, port 8787) exists in Google Drive, but the copy
there contains **hardcoded API keys**. It is deliberately **not**
copied into this package. `aethelred-agent/README.md` explains how
to sanitize it (move keys to environment variables) before it can
be published. Never commit API keys.

## Also not in this repo (and why)

- **ApothegmSovereignEngine** (React/TypeScript, 3Dmol.js) and the
  **Dash/Plotly/PennyLane VQE drug-discovery app**: described in
  the author's notes, but **no source files were found** in Google
  Drive or the local workspace on 2026-11-07 (Drive searched by
  name and full text). They cannot be shipped until the source is
  located. See `AUDIT.md`.
- **GWOSC strain files** (~40 MB each, 5 events) and **ACT DR6
  data files**: public data, downloaded by the run scripts — see
  component READMEs for exact URLs / GWOSC API calls and md5s.

## License

MIT — see `LICENSE`. (Why MIT, not Apache-2.0: see
`LICENSE-NOTE.md`. Short version: MIT is the simplest permissive
licence for a research pipeline others should be able to run and
fork; Apache-2.0's patent grant is a better fit only if/when a
patentable device or method ships, which this repo does not contain.)

## Citation

If you use the pipeline, cite the paper (Third Revised Edition,
Nov 2026, Derrick Irons, The Aethelred Institute, ORCID
0009-0001-7064-6611) and the Zenodo/arXiv record once published —
see `../arxiv-zenodo/` (sibling package) for the timestamp record.
Cite the *results as measured*, including the FAILs: γ ≲ 0.08,
not γ = 0.4.
