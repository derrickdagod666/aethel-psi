# Asset audit — what is shippable, what is missing

Audited 2026-11-07 by reading the local workspace and Google Drive
(read-only; Drive searched by name and full text; no Drive files
edited, deleted, shared, or copied except the two Python files
listed below, downloaded to /tmp for inspection). No secret values
are recorded anywhere in this package.

GitHub connection (checked same day):
- Status: `connected` (OAuth + MCP). Authenticated user:
  **derrickdagod666** (id 80133675, GitHub since 2021-03-05,
  0 public repos, 0 followers). Profile:
  https://github.com/derrickdagod666
- `create_repository` is available through the connector, but **no
  repo was created and nothing was pushed** — that needs Putt's
  choice of repo name + visibility (see `CHECKLIST.md`).
- Repo-name check: GitHub search for `aethel-psi` returned
  **0 results** — no public repo with that name found, so
  `aethel-psi` is proposed. Alternatives if it is taken or
  unwanted: `aethel-psi-synthesis`, `uaf-gw-pipeline`,
  `primordial-waveform`.
- GitHub App installation / repository selection is still the
  pending step: https://github.com/apps/meta-muse-ai/installations/new

## Shippable (in this package, code copied)

| Asset | Where it lives (source) | Stack | Tested? | Missing / caveats |
|---|---|---|---|---|
| Fixed Bilby UAF waveform `uaf_blunted_waveform` | `gate3-gw250114/gate3_pe.py` (canonical), also `gate3-gw231123/gate3_231123_pe.py`, suites v2–v5 | Python, bilby 2.8.2, lalsuite (IMRPhenomPv2) | **Verified live 2026-11-07**: import + call, modifier at f_merger = 1.000 / 0.600 / 0.500 for γ = 0 / 0.4 / 0.9. Real-data: recovered MF SNR 78.2 on GW250114 (LVK ~80). Full PE: γ median 0.0235, 95% upper 0.0813, ln B = −3.06 ± 0.23 | GWOSC strain not included (~40 MB/event); run scripts download it. Masses/spins fixed at published medians in the PE; no calibration marginalisation. |
| Gate 3 pipeline (profile + PE, 5 events) | `gate3-gw250114/`, `gate3-gw231123/`, `gate3-gw200129/xphm/` | Python, bilby, dynesty, numpy/scipy | Real data, 5 events: GW250114 ΔlnL(0.4) = −55.2; GW231123 +13.0/+23.9 (XPHM +23.0); GW200129 −15.7 Pv2 (XPHM tilt-dependent −16.4…+8.7, OPEN); GW150914 +0.1; GW190412 +5.4. 5-event stack ΔlnL(0.4) = −52.4, 95% γ < 0.173 | GW231123 full precessing PE died in a VM reboot ~03:46 UTC 2026-11-07 (checkpoint exists, resume untested, no posterior/Bayes factor). GW200129 needs a spin-marginalised PE. |
| Qiskit stabilizer demo | `new-theory-tests/test_qiskit.py` | Python, Qiskit 2.6.2, qiskit-aer | **Verified live 2026-11-07: PASS 4/4**, GHZ 4096/4096 in no-error and error-on-q0/q1/q2 runs | 3-qubit repetition code only — a demo of the syndrome table, not cosmic unitarity. Reconstructed from a garbled paper listing; `if_test` replaces deprecated `.c_if()`. |
| Validation suites v2–v5 | `new-theory-tests/aethel_psi_suite_v2.py` … `v5.py`, `test_core.py`, `test_bilby_real.py`, `test_bilby_qiskit.py` | Python, numpy/scipy/sympy (+ bilby/qiskit for the Bilby/Qiskit tests) | v2 quick PASS=29 FAIL=1 OPEN=8; v3 full PASS=40 FAIL=5 OPEN=10; v4 PASS=6 FAIL=11 OPEN=8; **v5 full PASS=25 FAIL=18 OPEN=6 INFO=8** (exit 1, correctly). v6 (separate batch, not a suite file): PASS=9 FAIL=6 OPEN=2 | Suites are standalone scripts, not a pip package; v5 reads v3/v4 output files in C0 (copies not bundled — rerun v3/v4 first or edit C0 paths). ACT C7 in v5 is a hook; real ACT code is in `act-dr6/`. |
| ACT DR6 ripple test | `new-theory-tests/act-dr6/a5_real.py`, `a5_joint.py`, `camb_joint_refit.py` | Python, sacc, scipy, cobaya + camb + act_dr6_mflike | **Real DR6.02 data**: TT-only p = 0.241, A95 = 1.60×10⁻²; joint TT+TE+EE p = 0.105, A95 = 1.33×10⁻² (20,000 null sims). No ripple. Diagonal-only p = 0.004 is a documented false positive — never quote it | Needs cobaya install (`cobaya-install`, downloads DR6 data ~GBs); CosmoRec not installed, no MCMC attempted. Data files not included. |
| GW stacking + Fisher forecast | `gate3-gw250114/stacking/stack_profile.py`, `fisher_stacking.py`, `stack_data.py` | Python, numpy | 4-event stack ΔlnL(0.4) = −57.8, 5-event −52.4. Forecast corrected: ~20 events at SNR ~80 for 1σ on γ ≈ 0.009, ~200 for 3σ | Summed profile likelihood, NOT a stacked posterior. |
| Quantum-cognitive / PennyLane code | Google Drive `code.py` (32,303 bytes, 642 lines, 2025-04-16) | Python, torch, PennyLane, faiss, pyvista, transformers | **Not run locally.** No test output exists for it. Imports are heavy (torch + transformers + pyvista) | No Dash/Plotly app in this file despite the description in notes; no VQE drug-discovery code in it. `generate_synthetic_data` only — no real dataset. Treat as exploratory, not production. |
| Harmonic quantum tests | Google Drive `quantum_tests_harmonic_full.py` (3,480 bytes, 68 lines, 2025-07-10) | Python, qutip, qiskit, PennyLane | **Not run locally.** References `system` object defined nowhere in the file — it is a fragment, not runnable as-is | Incomplete: `run_harmonic_oscillator_tests(system)` with no class definition in the file; `qml.device("default.qubit.tf")` is a legacy PennyLane device. Needs reconstruction before shipping as runnable code. |

## NOT shippable as code (missing or blocked)

| Asset | Status | What is missing |
|---|---|---|
| ApothegmSovereignEngine (React/TypeScript, 3Dmol.js, QSAE-QAOA) | **NOT FOUND** in Drive (name + full-text search, 2026-11-07) or local workspace. Described only in the author's notes. | The source itself. Cannot ship until located (check the original machine / repo it was built in). |
| Dash/Plotly/PennyLane VQE drug-discovery app | **NOT FOUND** as code in Drive or workspace. Drive `code.py` is PennyLane + torch but is not a Dash app and not a VQE drug app. | The source itself (and any molecular dataset / backend it used). |
| Aethelred Agent backend (TypeScript, Gemini 2.5 Pro, Express, SQLite, WebSocket, port 8787) | **FOUND in Drive but BLOCKED**: `Aethelred Agent - Final Integrated Script.txt` (15,045 bytes, 318 lines, 2025-08-27) + Google Doc version. **Contains hardcoded API keys** — deliberately NOT copied into this package. | Sanitization: move all keys to environment variables (see `aethelred-agent/README.md` + `.env.example`), then `npm install` + `tsc` check. Also needs a `DOTS.txt` file it reads at runtime and an `agent_data/` dir. No run has been verified locally. |
| Aethelred_Key (Drive doc, 2025-08-08) | **Not opened, not copied.** Credentials-type document. | Must never be committed. If the agent is shipped, keys must be re-issued / rotated and stored in env vars, never in the repo. |

## What was verified live for this package (2026-11-07)

1. GitHub `status`: connected; `get_me`: derrickdagod666.
2. Repo-name search `aethel-psi`: 0 results.
3. `test_qiskit.py` re-run in `venv_qiskit` (Qiskit 2.6.2):
   PASS 4/4, GHZ 4096/4096 in all four runs.
4. `uaf_blunted_waveform` import + call in `venv_qiskit`
   (Bilby 2.8.2): modifier 1.000 / 0.600 / 0.500 at γ = 0 / 0.4 / 0.9.
5. No secret values found by grep in any copied Python file
   (`api_key|secret|token|password` — no hits in the copied sets).
   The Aethelred script was excluded from copying for exactly
   this reason; its keys were never reproduced here.

## Honesty ledger (results the repo must not contradict)

- γ = 0.4 excluded (GW250114 profile −55.2; full PE ln B = −3.06;
  5-event stack −52.4). Prediction 1: γ ≲ 0.08 (95%).
- Spin-γ hypothesis closed in v6 (both branches FAIL on real LVK
  spin samples: GW250114 χ_eff median −0.035, χ_p median 0.114).
- ACT joint null (p = 0.105). β_A fitted, Λ_eff ≡ 0 dropped,
  phantom a_c = 0.476 unreproduced, ξ = −1/6 derived (stated D_Ψ).
- 10 parameters/functions remain unspecified (v5).
