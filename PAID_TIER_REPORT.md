# Paid tier + all outstanding suggestions — report

**Date:** 2026-11-07 (UTC). **Repo:** https://github.com/derrickdagod666/aethel-psi
(branch `main`). **Instruction:** "Do paid tier and all your suggestions"
(Putt, 2026-11-07).

Honesty rule applied throughout: per-item PASS / FAIL / OPEN verdicts,
no post-hoc refit to rescue a prediction, no result quoted from a run
still in progress, no pricing promises, no sale, the theory itself is
not for sale. γ = 0.4 is excluded; γ ≲ 0.08 for GW250114-like mergers;
the spin-γ hypothesis is CLOSED (v6); ACT DR6 joint TT+TE+EE is null
(global p = 0.105, A95 = 1.33×10⁻²).

## Summary table

| # | Item | Verdict |
|---|---|---|
| 1 | Paid-tier packaging (pip package + CI + v0.1.0 files) | **DONE** (release creation still to do — browser step) |
| 2 | Resume GW231123 precessing PE | **RUNNING** (resumed twice; dlogz 6.92, target 0.1) |
| 3 | GW200129 spin-marginalised PE | **RUNNING** (UAF run only; no posterior/Bayes factor yet) |
| 4 | GW190412 XPHM / higher-mode profile | **DONE — FAIL** (the +5.4 is NOT a pure Pv2 artefact; +2.9 under XPHM, still no exclusion) |
| 5 | Locate missing code (ApothegmSovereignEngine, VQE app) | **NOT FOUND** (harder search; Keep notes remain unchecked) |
| 6 | Aethelred Agent sanitization | **BLOCKED on Putt** (key rotation — his step, not automatable) |
| 7 | README + PITCH.md updated, this report written, push to repo | **DONE** (this push; release step outstanding) |

## 1. Paid-tier packaging — DONE (release creation outstanding)

- **Pip-installable package v0.1.0:** `pyproject.toml` (setuptools,
  `src` layout), `src/aethel_psi/__init__.py` + `waveform.py`.
  `from aethel_psi import uaf_blunted_waveform, dip_modifier` — the
  fixed waveform extracted verbatim in logic from
  `gw-pipeline/uaf_bilby_plugin.py`, decoupled from
  `gate3_common.py` so it imports without GWOSC strain. Pinned
  **Bilby 2.8.2 / Qiskit 2.6.2**; `.[dev]` = pytest,
  `.[act]` = cobaya/camb/sacc (opt-in only). Original Gate 3
  scripts untouched.
- **Clean-install test — PASS (10/10).** Fresh venv, Python 3.12.3,
  `pip install -e ".[dev]"` succeeded in 1m 38s; import check
  printed `aethel_psi 0.1.0 / bilby 2.8.2 / qiskit 2.6.2`;
  `pytest tests/ -v` → **10 passed in 12.69s**, exit 0, including
  the live LAL ratio test (1.000 / 0.600 / 0.500 at f_merger for
  γ = 0 / 0.4 / 0.9). Two failures on the way are recorded in
  `PAID_TIER_PACKAGING.md`, not hidden: attempt 1 failed
  `Errno 28 No space left on device` (/tmp is a 512 MB tmpfs —
  reran with venv/cache under `~/workspace`); attempt 2's
  background session vanished mid-install with no pip error, so it
  was resumed rather than claimed as a pass. A setuptools license
  deprecation was fixed (SPDX `license = "MIT"`, setuptools>=77)
  and re-verified: editable reinstall clean, `pip wheel` builds
  `aethel_psi-0.1.0-py3-none-any.whl` (9,328 bytes), waveform
  tests re-passed 6/6. Scoping note: the full test was an editable
  install; a separate non-editable `pip install .` run was not
  done (wheel builds cleanly; CI uses the verified editable
  command).
- **CI:** `.github/workflows/ci.yml` — on push/PR, Python 3.12,
  `pip install -e ".[dev]"`, `pytest tests/ -v`.
- **v0.1.0 release files:** `CHANGELOG.md` and
  `RELEASE_NOTES_v0.1.0.md`, honest throughout: what's in,
  measured results, and what's NOT in (ApothegmSovereignEngine
  and VQE app source not found, Aethelred Agent code excluded —
  hardcoded API keys).
- **PITCH.md:** outline only — Tier 1 free open core / Tier 2
  paid GW plugin-support / Tier 3 scoped consulting, no prices;
  the "package + CI + v0.1.0" gate is marked done, release
  creation still to do.
- Full detail: `PAID_TIER_PACKAGING.md`. No secrets copied
  (grep-checked). Test venv and pip cache deleted; build
  artifacts removed from the tree.

### v0.1.0 release — NOT yet created (browser step, for the parent)

The GitHub connector cannot tag (`create_repository` already
fails 403 for the App token; tagging is the same class of
limitation) and no tag or GitHub release was created. To make
the release, on github.com signed in as derrickdagod666:

1. Push this report's files to `main` first (done in this push),
   so the release target includes the package.
2. Releases → Draft a new release → tag `v0.1.0` (create new tag
   on publish, target `main`), title `v0.1.0`, paste the body
   from `RELEASE_NOTES_v0.1.0.md`, optionally attach
   `docs/aethel-psi-synthesis-third-revised-edition.pdf`, Publish.
   (Alternative: `git tag -a v0.1.0` + push the tag, then create
   the release from it on the web.)
3. Verify the Actions tab shows the CI run green (10 tests)
   before announcing v0.1.0.
4. Only then may Tier 2 be offered; no prices are set here and
   no sale has been made.

## 2. GW231123 precessing PE — RUNNING (resumed twice)

- IMRPhenomXPHM, 11-param UAF, γ~U(0,0.9), geocent_time prior
  widened to ±0.15 s, dynesty nlive=1000/dlogz=0.1. Launched
  ~02:13 UTC 2026-11-07; died in a VM reboot ~03:46 UTC.
- **First resume (06:21 UTC):** worked, did not restart from
  zero. Key catch: the original `precessing_pe.py` passes
  `clean=True`, which in Bilby 2.8.2 forces `resume=False` —
  re-running it as-is would have discarded the checkpoint and
  lost ~1.1M likelihood calls. A dedicated resume script was
  written instead (`precessing_pe_resume.py`, `clean=False`,
  `resume=True`, same outdir/label, 60 s checkpointing; the
  original used 600 s). Checkpoint verified loadable:
  ncall = 1,106,149, it = 16073, nlive = 1000, ndim = 11.
  Launched detached (setsid), PID 22313. Correction: dlogz
  8.57 was the 03:37 UTC value; by the last pre-reboot log line
  (03:49 UTC) it had fallen to 6.99.
- **Second reboot (~06:24 UTC) killed it again** (last line
  16159it, dlogz 7.17). **Second resume (06:28 UTC):** checkpoint
  verified loadable again (ncall = 1,114,164, it = 16133,
  logz-ratio = 200.371 ± 0.12), fresh backup made
  (`GW231123_resume.pickle.bak-20261107-0628UTC`, md5-verified).
  Relaunched detached, **PID 3655** (verified alive), log
  `pe_run4.log`, "Resume file successfully loaded", first line
  16133it / dlogz 7.22 — exactly the checkpoint state.
  Checkpoint rewriting confirmed (06:28:46, 06:30:06 UTC).
- **Latest numbers (06:37 UTC check): 16279it, ncall ≈ 1.1×10⁶,
  eff 1.4%, logz-ratio = 200.53 ± 0.12, dlogz = 6.92 > 0.1.
  RUNNING — no posterior, no lnZ, no Bayes factor yet.**
- Contention: load average 4.55–6.84 on 4 cores — the GW200129
  PE and the GW190412 XPHM profile run concurrently. Rate
  ~0.19 it/s vs ~1.1 it/s after the first resume; no firm ETA,
  should recover when a competing job finishes. UAF run only —
  the GR run was not started (no checkpoint; separate launch
  decision), so no Bayes factor exists until both finish.
- Report: `workspace/gate3-gw231123/precessing-pe/PE_RESUME_REPORT.md`
  (§9 = second resume). Checkpoint:
  `gate3_xphm_uaf/GW231123_resume.pickle`.

## 3. GW200129 spin-marginalised PE — RUNNING

- **Pre-registered priors in writing before any PE likelihood**
  (`workspace/gate3-gw200129/PE_REPORT.md` §1,
  `gw200129_pe.py`): IMRPhenomXPHM, band 20–400 Hz, no
  notches; fixed masses 40.7/34.2 M☉ (det), spin magnitudes
  a1=0.90/a2=0.50, sky ra=1.57/dec=0.80. Sampled UAF (11
  params): γ~U(0,0.9), f_merger~U(100,200) Hz (pre-registered
  central 143 Hz inside), width_frac~U(0.05,0.30),
  DL~U(400,1600) Mpc, θ_JN~U(0,π), phase~U(0,2π),
  geocent_time~TRIGGER±0.05 s, **tilt_1/tilt_2~U(0,π)
  (uniform in tilt, not isotropic) and φ_12/φ_jl~U(0,2π)** —
  the orientations are the marginalised nuisance that flipped
  the profile ΔlnL. GR run: same minus γ/f_merger/width_frac
  (8 params). Dynesty nlive=1000, dlogz=0.1, rwalk, walks=25,
  npool=4 — identical to the GW250114 PE. Honesty note
  recorded: the PE is not blind to the profile result (spin
  priors chosen after the profile showed tilt-fragility); it
  must be judged on its own posterior/Bayes factor.
- **Pre-launch verification passed:** UAF modifier at f=143 Hz:
  1.0000/0.6000/0.5000 at γ=0/0.4/0.9. Likelihood cross-check
  vs the XPHM profile: GR ratio 258.6 (profile 256.1),
  **ΔlnL(γ=0.4) = −7.05 (profile −7.2)** — the PE likelihood
  reproduces the profile. (A fixed-phase smoke difference of
  +116 was traced to a phase artefact, documented in the
  report.)
- Launch: first launch (PID 23341, 06:23 UTC) was killed by the
  ~06:24 UTC VM restart (no traceback, no checkpoint — it had
  only reached dynesty setup). Relaunched unchanged at
  **06:24:46 UTC, PID 2246** (`setsid … python -u
  gw200129_pe.py`, log `full-pe/pe_run.log`). Verified alive.
  At 01:24 elapsed: **1631 iterations, ncall ≈ 4.0×10³,
  eff 40.6%, logz-ratio −68.37 ± 0.07, dlogz = 323**
  (criterion ≤ 0.1 not met — expected this early).
- **Status: RUNNING (UAF run only; GR run queued after it). No
  posterior, no Bayes factor, and no verdict on whether
  γ < 0.065 survives — none is claimed.** Expect several hours
  for the UAF run plus a comparable GR run. Log flooded with
  the known-benign `floor` warning; disk fine (146 GB free).

## 4. GW190412 XPHM / higher-mode profile — DONE, FAIL

Profile only (no PE, no Bayes factor, no marginalised
posterior). Real GWOSC O3a strain, f_merger = 254 Hz
pre-registered, width 0.15, band 20–400 Hz, 0.5 ms dt grid.
Report: `workspace/gate3-gw190412/xphm/XPHM_PROFILE_REPORT.md`.

| Config | GR lnL | \|MF\| / opt SNR | ΔlnL(γ=0.4) | Wilks 95% |
|---|---|---|---|---|
| Pv2 aligned (v6) | 120.2 | 15.5 / 14.4 | **+5.4** | none |
| A: XPHM aligned (control) | 122.7 | 15.7 / 14.8 | **+4.6** | none |
| B: XPHM precessing, a1=0.44, tilt 0.9 (headline; χ_eff=0.206, χ_p=0.345) | 130.1 | 16.2 / 14.5 | **+2.9** | none |
| Tilt 0.5 / 1.4 | 100.3 / 78.9 | 14.2 / 12.7 | **+3.9 / +2.0** | — |

- **Verdict: FAIL** — the +5.4 is NOT a pure Pv2 higher-mode
  artefact. The dip preference survives IMRPhenomXPHM but is
  roughly halved (+5.4 → +2.9). Approximant change alone
  removes only 0.8 log-units (+5.4 → +4.6); precession at the
  published point removes ~2.5 total. Unlike GW200129 (ΔlnL
  spanned −16 to +9 across tilts), the sign **never flips**
  here: +2.0 to +3.9 across tilts 0.5–1.4. The GR fit improves
  +7.4 under precession (130.1 vs 122.7), so part of what the
  dip absorbed was genuine precession/higher-mode mismatch —
  but only about half.
- Still a preference, not a detection: both curves rise
  monotonically to the floor (B: +4.3 at γ=0.9). Placement
  diagnostics under B: 244 Hz +3.0, 264 Hz +2.8, 200 Hz +1.4,
  300 Hz +1.7; width 0.10 +2.1, 0.25 +2.5.
- Recovered |MF| 16.2 vs published 19.8 (82%); L1 still
  dominates and L1 |MF| (15.0) still exceeds its optimal SNR
  (11.5).
- **5-event stack with the XPHM-B number substituted**
  (recomputed, labelled as such): all 5: ΔlnL(0.4) **−54.9**
  (was −52.4), Wilks 95% γ < **0.159** (was 0.173); without
  GW231123: −67.9, 95% γ < 0.105; wide-dt variant: −44.0,
  95% γ < 0.218. Config-A variant: −53.3, 95% γ < 0.169.
  γ ≈ 0.009 still untouched (stack at γ=0.1 is +2.9 / −1.3).
  Per-event pattern by mass still non-monotone (+2.9, −15.7
  Pv2 / −7.2 XPHM-B, +0.1, −55.2, +13.0) — no γ(M) survives.
- Method notes: pre-registration (§1) was written before any
  XPHM likelihood; f_merger, spins, sky (fixed at Pv2 GR max
  ra=4.71, dec=+0.80), and dt grid were not changed after
  seeing results. dt diagnostic: Pv2 reproduced **exactly**
  (120.194); the 5 ms grid under-resolves the XPHM peak at
  254 Hz by 45.8 log-units (76.9 vs 122.7); the 0.5 ms grid is
  verified against 0.25 ms to 0.1 log-units. One background
  exec run died silently mid-way (no traceback, log ended at
  γ=0.2 config A); it was re-run in full from scratch — no
  partial numbers are cited. Environment: Bilby 2.8.2,
  LALSimulation 7.0.0 (note: v6 report said 8.0.0; the venv
  reports 7.0.0 — recorded as run).
- Limitation to flag: the substituted stack mixes approximants
  (XPHM for GW190412, Pv2 for the other four); a fully XPHM
  stack would need XPHM profiles for GW150914 and GW250114
  too. A spin-marginalised PE is still required before
  GW190412 can be quoted for or against γ = 0.4; cite as
  "profile preference at a fixed point, +2.9 under
  XPHM-precessing (+5.4 under Pv2)".

## 5. Locate the missing code — NOT FOUND

Harder search, all read-only on Drive (no edits, no secrets
copied). Full detail: `CODE_HUNT_REPORT.md`.

- **Inventory:** all 1,097 items in
  `google-drive-inventory.csv` by name — 0 hits for Apothegm,
  Sovereign, VQE, PennyLane, Dash, Plotly, 3Dmol, React,
  TypeScript, molecule. No `.ts`/`.tsx`/`.js`/`package.json`
  exists in Drive at all.
- **Live Drive full-text:** `Apothegm` 0, `SovereignEngine` 0,
  `Apothem` 0, `3Dmol`/`3Dmol.js` 0, `VQE` only 2 prose hits
  (both copies of "Aethel-Psi Paradigm: Nobel Path" — VQE is a
  prose claim about θΨ≈π, no code). `Sovereign` only prose
  ("Sovereignty vs. Power", "technological sovereignty").
  Trashed copies checked — 0. No shared drives.
- **Content grep of all 119 downloadable code/text files**
  (75 AI Studio/Colab files + 44 root `.txt`, including all
  ~50 unnamed `code (N).txt` exports), **all 4 ZIPs** (Web.zip
  = Snoritech mockup prose; Health Connect = a `.db` export;
  DerrickReport zips = duplicate PDFs), **all 3 Colab
  notebooks** (two empty, one DeepSeek-Math/Gradio), and **5
  Google Docs read in full** — zero Apothegm/VQE/Dash/Plotly/
  3Dmol code. Every apparent "dash/plotly" hit was LaTeX
  `\addplot[dashed]`; the only real `plotly` in code is a
  `!pip install … plotly …` line in Colab `code (3).py`.
- **Local workspace:** no `*.tsx`/`*.ts`/`package.json` under
  `~/workspace/user`; only mentions in prior deliverables.
- **Related code found, audited, NOT copied** (none is either
  missing app): `code (2).py` ×2 (26 KB, quantum-system/
  Gradio, fails `py_compile` at line 333), `code (3).py`
  (30 KB, same framework + Plotly viz, Colab `!pip` magic),
  `enhanced-ai-assistant.py` (8.8 KB Streamlit coding
  assistant, compiles, token passed as arg), and a complete
  React "Living AI Chat Widget" (626 lines, Keep export
  2025-08-23 — the only React in Drive, but the wrong app).
  All secrets-scanned clean. Left out deliberately to avoid
  muddying the repo; exact Drive locations are in the report.
- **Prime remaining suspect:** the live Google Keep notes —
  the apps are described as code there, but Drive holds only
  ~10 Keep `.txt` exports of 44 notes, and the 7 AI Studio
  apps can't be content-searched via Drive. Both need Putt /
  a browser session to check.

### VQE app — new implementation built from the spec (not the original)

After the search above reported NOT FOUND, a new VQE app was
built in this run window (`vqe-app/`: `app.py`, `vqe_core.py`,
`test_vqe.py`, plus this README) from the Keep-note spec
(Dash + Plotly + PennyLane + VQE + drug discovery). It was
written 2026-11-07 06:35–06:39 UTC; it was not produced by
any of the five subagents in this run, so its authorship
within this run is unattributed in the coordinator's records
— what matters is what the files themselves say, and they
label it honestly: **NEWLY WRITTEN, not Putt's code**, with
exact reuse labels (the learnable-Psi RY pattern from
`drive-code/quantum_cognitive_pennylane.py` used only as an
optional default-OFF ansatz initialisation; the Hamiltonian,
VQE loop, exact-diagonalisation reference, bond scan, and all
Dash/Plotly UI newly written; 3D viewer is Plotly 3D scatter,
not 3Dmol.js). Limits are stated in `vqe-app/README.md`:
simulator only, STO-3G, active spaces stated, no hardware,
no drug-efficacy claims, chemical accuracy NOT claimed.
- **Test status — FAIL (verified by the coordinator).** The
  headless test (`test_vqe.py`: H2, LiH, H2O, NH3 at
  equilibrium + 5-point H2 bond scan, tolerances H2 0.01 /
  LiH 0.02 / H2O,NH3 0.05 Ha) was re-run by the coordinator
  in the app's own venv (PennyLane 0.45.1) on 2026-11-07
  (~12 min wall time) — **RESULT: FAIL.** H2 PASS
  (VQE −1.137261 vs exact −1.137284 Ha, err +0.000022, tol
  0.010); LiH PASS (−7.862125 vs −7.862129, err +0.000004,
  tol 0.020); **H2O FAIL** (−74.761284 vs −74.970810, err
  +0.209526, tol 0.050); **NH3 FAIL** (−55.157837 vs
  −55.465417, err +0.307579, tol 0.050). H2 bond scan (60
  steps each): only r=0.50 Å PASS (err +0.006775); r=0.74,
  1.00, 1.40, 2.00 Å all FAIL (errs +0.036, +0.178, +0.073,
  +0.025). Cause as diagnosed: the 2-layer hardware-efficient
  ansatz at 120 steps is not expressive/optimised enough for
  the 8-qubit molecules — more layers/steps or a better
  ansatz is needed; this is a test failure of the new code,
  not of the spec. The app is pushed with this FAIL recorded
  in its README and here, not hidden. Do NOT cite a PASS for
  this app. If the original VQE app source is ever located,
  it should replace this implementation.
- The 614 MB `vqe-app/venv/` is NOT pushed (excluded by
  `.gitignore`); only the three Python files + README go to
  the repo.

## 6. Aethelred Agent sanitization — BLOCKED on Putt

- No agent code copied; no key value copied anywhere;
  `Aethelred_Key` was never opened.
- `aethelred-agent/README.md` rewritten as a full
  sanitization guide (env-var table, exact de-hardcoding
  pattern, verification commands).
- `.env.example` already existed and was **verified
  names-only** (6 lines, every value <10 chars — a real key
  is 32–39 chars); left unchanged.
- `aethelred-agent/SANITIZATION_CHECKLIST.md` added
  (isolate → de-hardcode → rotate → rebuild → verify →
  publish).
- **Blocked on Putt:** rotate/re-issue the Google,
  OpenWeather, and NewsAPI keys before the agent can ship as
  code. That step is his, not automatable here.
- Scratch copies of the downloaded Drive files (which included
  the agent `.txt`) were deleted after the run; the report is
  the surviving record and contains no secret values.

## 7. What was pushed (this report)

This push to `derrickdagod666/aethel-psi` on `main` contains:
the v0.1.0 package (`pyproject.toml`, `src/aethel_psi/`,
`tests/`), CI (`.github/workflows/ci.yml`), `CHANGELOG.md`,
`RELEASE_NOTES_v0.1.0.md`, `PAID_TIER_PACKAGING.md`,
`CODE_HUNT_REPORT.md`, `aethelred-agent/SANITIZATION_CHECKLIST.md`,
updated `README.md`, `PITCH.md`, `.gitignore`,
`aethelred-agent/README.md`, and this report. Text/UTF-8
files only — no binary, no `__pycache__`, no secrets — per
the snapshot-push procedure; the Third Revised Edition PDF
is already in `docs/` from the earlier browser upload
(commit cbb16405) and was not re-pushed.

## What remains OPEN / blocked on Putt

1. **v0.1.0 GitHub release** — browser step (see §1). No
   prices, no Tier 2 offer until it exists and CI is green.
2. **GW231123 precessing PE** — running (dlogz 6.92); GR run
   still to launch after UAF finishes. No posterior/Bayes
   factor until both finish.
3. **GW200129 spin-marginalised PE** — running (UAF only);
   GR run queued after it. No verdict on γ < 0.065 yet.
4. **GW190412 spin-marginalised PE** — still required before
   GW190412 can be quoted for or against γ = 0.4.
5. **Key rotation** (Google, OpenWeather, NewsAPI) — Putt's
   step; the Aethelred Agent cannot ship as code until done.
6. **Missing code** — ApothegmSovereignEngine and the VQE app:
   prime remaining suspect is the live Google Keep notes
   (needs Putt / a browser session).
7. **VM stability** — two reboots in one morning killed both
   long PEs once each; both were resumed from checkpoints.
   A third reboot would kill them again; checkpoints are
   written every 60 s.
