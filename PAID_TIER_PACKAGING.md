# Paid-tier packaging — v0.1.0 (2026-11-07)

Status: **packaging + CI + v0.1.0 release files are done and verified by a
real clean-install test (10/10 tests pass). Nothing has been pushed.**
The GitHub release itself still has to be created on the web — see
"What remains" below. No prices are set, no sale has been made, and the
theory itself is not for sale.

## 1. Clean-install test — exact results

Environment: Linux VM, system Python **3.12.3**, fresh venvs, pip
upgraded to 26.3 inside the venv. Test command (run from the repo root):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import aethel_psi, bilby, qiskit; print(aethel_psi.__version__, bilby.__version__, qiskit.__version__)"
python -m pytest tests/ -v
```

### Attempt 1 — FAILED (06:20 UTC, venv in /tmp)

`pip install -e ".[dev]"` failed with:

```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

Cause: `/tmp` on this VM is a **512 MB tmpfs**; pip's build env plus the
Bilby/LAL dependency tree does not fit. The subsequent import check
failed as expected (`ModuleNotFoundError: No module named 'aethel_psi'`).
Note: that run printed `--- install exit: 0 ---`, which was the exit
code of `tail`, not of pip — the install genuinely failed. This attempt
is recorded, not hidden.

### Attempt 2 — interrupted, no result (06:22 UTC)

Re-ran with the venv and pip temp/cache dirs under `~/workspace`
(146 GB free): `TMPDIR=~/workspace/.tmp-pip`,
`PIP_CACHE_DIR=~/workspace/.tmp-pip/cache`. Dependency resolution and
all wheel downloads completed (bilby 2.8.2, qiskit 2.6.2, lalsuite
8.0.0, numpy 2.5.4, scipy 1.18.1, …), install was in progress — then
the background session vanished with no pip error and no completion
report. On inspection only 6 small packages had been installed
(sympy, tqdm, mpmath, tzdata, typing_extensions, pip). Cause unknown.
I did **not** treat this as a pass; I resumed the install instead.

### Attempt 3 (resume) — PASS (06:24:38 UTC)

Same venv, same command, wheels served from the pip cache:

- `pip install -e ".[dev]"`: **SUCCESS in 1m 38.153s** (real time).
  `Successfully installed … aethel-psi-0.1.0 bilby-2.8.2 …
  qiskit-2.6.2 qiskit-aer-0.18.2 lalsuite-8.0.0 … pytest-9.2.0`
- Import check printed: `aethel_psi 0.1.0`, `bilby 2.8.2`,
  `qiskit 2.6.2` — all three pinned versions as specified.
- `python -m pytest tests/ -v`: **10 passed in 12.69s** (run wall
  time 15.754s), exit code 0. All 4 Qiskit stabilizer cases pass
  (GHZ recovery) and all 6 waveform checks pass, including the live
  LAL ratio test: modified/baseline at f_merger = 1.000 / 0.600 /
  0.500 for γ = 0 / 0.4 / 0.9.

Resolved versions of the unpinned dependencies at install time:
lalsuite 8.0.0, numpy 2.5.4, scipy 1.18.1, qiskit-aer 0.18.2,
pytest 9.2.0, astropy 8.0.2, pandas 3.1.1, matplotlib 3.11.2,
sympy 1.14.0.

### Post-fix re-verification (license line)

The first successful build printed a setuptools deprecation warning
(`project.license` as a TOML table; also a deprecated MIT classifier).
I changed `pyproject.toml` to SPDX form (`license = "MIT"`,
`requires = ["setuptools>=77"]`, classifier removed) and re-verified:

- `pip install -e . --no-deps --force-reinstall` (build isolation):
  **success, no license warning**.
- `pip wheel . --no-deps`: built `aethel_psi-0.1.0-py3-none-any.whl`
  (9,328 bytes).
- Waveform tests re-run: **6 passed in 6.89s**.

One scoping note: the full test was an **editable** install
(`pip install -e .`). A non-editable `pip install .` end-to-end in a
fresh venv was not run separately; the standalone wheel above builds
cleanly, and CI uses the same editable command that was verified.

## 2. Files created / changed (all under ~/workspace/user/files/github-repo/)

New files:

| File | What it is |
|---|---|
| `pyproject.toml` | Package metadata, v0.1.0, `bilby==2.8.2`, `qiskit==2.6.2` pinned; `.[dev]` extra = pytest; `[act]` extra = cobaya/camb/sacc (heavy, opt-in only) |
| `src/aethel_psi/__init__.py` | Exports `uaf_blunted_waveform`, `dip_modifier`, `WAVEFORM_ARGS`, `F_MIN`, `F_MAX`, `__version__ = "0.1.0"` |
| `src/aethel_psi/waveform.py` | The fixed waveform, logic verbatim from `gw-pipeline/uaf_bilby_plugin.py`, decoupled from `gate3_common.py` (no GWOSC strain needed to import); pure-numpy `dip_modifier()` + LAL-backed `uaf_blunted_waveform()` |
| `tests/test_waveform_modifier.py` | 6 tests: modifier 1.000/0.600/0.500 at f_merger, bounds/floor, fractional-width check (FIX 1), live LAL waveform-ratio check + `f_merger` required (FIX 2) |
| `tests/test_qiskit_stabilizer.py` | 4 pytest cases mirroring `qiskit-stabilizer/test_qiskit.py` (no-error, error on q0/q1/q2; GHZ fraction ≈ 1.0) |
| `.github/workflows/ci.yml` | GitHub Actions: on push/PR, Python 3.12, `pip install -e ".[dev]"`, `pytest tests/ -v` |
| `CHANGELOG.md` | v0.1.0 entry: what is in, measured results, and what is NOT in (see below) |
| `RELEASE_NOTES_v0.1.0.md` | Release body for the GitHub release (paste as-is) |

Changed files:

| File | Change |
|---|---|
| `README.md` | Added "Install as a package (v0.1.0)" quickstart (`pip install -e .`, `pytest tests/ -v`, import example) above the loose-scripts quickstart |
| `PITCH.md` | Outline only: Tier 1/2/3 unchanged in structure; the "what has to exist first" gate is now marked **done 2026-11-07** (package + CI + v0.1.0 files), with the GitHub release creation still to do; still no prices, no sale |
| `.gitignore` | Added `.pytest_cache/`, `build/`, `dist/` (build artifacts from the verification runs were removed from the tree) |

Unchanged on purpose: `gw-pipeline/uaf_bilby_plugin.py` and all Gate 3 /
validation / ACT / stacking scripts (the package extracts the waveform;
the run scripts stay canonical). No secrets were copied or added
(grep-checked the new/changed files for key/token/password/secret).

Test venv (`~/workspace/tmp-aethel-psi-venv-test`) and pip temp/cache
(`~/workspace/.tmp-pip`) were deleted after verification.

## 3. What v0.1.0 honestly contains — and does not

In: pip-installable `aethel_psi` 0.1.0, tests + CI, everything from the
first commit (Gate 3 pipeline, suites v2–v5 + v6, ACT DR6 test,
stacking, Qiskit demo, Third Revised Edition PDF). Measured results
shipped: γ = 0.4 excluded (GW250114 profile ΔlnL = −55.2; full PE γ
median 0.0235, 95% upper 0.0813, ln B = −3.06 ± 0.23); Prediction 1 is
γ ≲ 0.08 (95%); spin-γ hypothesis CLOSED (v6); ACT joint null
p = 0.105; v5 PASS 25 / FAIL 18 / OPEN 6 / INFO 8.

NOT in v0.1.0: ApothegmSovereignEngine and the VQE drug-discovery app
(source not found in Drive or workspace, 2026-11-07); Aethelred Agent
code (Drive copy contains hardcoded API keys — deliberately excluded,
README + `.env.example` only); GWOSC strain / ACT DR6 data files
(downloaded by run scripts); no resumed GW231123 precessing PE; no
spin-marginalised GW200129 PE; no replacement γ(M)/γ(spin) mechanism;
no peer review, no Nobel claim, no valuation. The theory itself is not
for sale.

## 4. What remains — release creation (coordinator / browser)

Nothing was pushed and nothing was tagged. The coordinator must push
the files above to `main` first, then create the release on the web
(the GitHub connector cannot tag; `create_repository` failed 403
earlier, but push-files works):

1. **Push** the new/changed files to `https://github.com/derrickdagod666/aethel-psi`
   branch `main` (local package is `~/workspace/user/files/github-repo/`).
2. **Create the tag + release on github.com** (signed in as
   derrickdagod666): repo → **Releases** → **Draft a new release** →
   **Choose a tag** → type `v0.1.0` → **Create new tag: v0.1.0 on
   publish** → target `main` → Release title `v0.1.0` → paste the
   body from `RELEASE_NOTES_v0.1.0.md` verbatim → optionally attach
   `docs/aethel-psi-synthesis-third-revised-edition.pdf` as a release
   binary (it is already in the repo under `docs/`) → **Publish
   release**. (Equivalent via git CLI, if preferred:
   `git tag -a v0.1.0 -m "v0.1.0 — first packaged release"` then
   `git push origin v0.1.0`, then create the release from that tag on
   the web with the same notes.)
3. **Verify CI**: after the push, open the repo's **Actions** tab —
   the `CI` workflow run on `main` should be green (10 tests). If it
   is red, do not announce v0.1.0 as verified; report the failure
   back.
4. **Then, and only then**, Tier 2 (paid GW plugin/support) may be
   offered to GW groups — per `PITCH.md`, with no prices set here and
   no sale until the release exists.
