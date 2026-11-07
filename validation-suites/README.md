# validation-suites — falsifiable suites v2–v5

Standalone scripts. Every check is pre-registered and able to FAIL;
the scripts exit 1 when any check FAILs (that is correct behaviour,
not a bug).

| Suite | Result | Notes |
|---|---|---|
| v2 (`aethel_psi_suite_v2.py`, `--quick`) | PASS=29 FAIL=1 OPEN=8 | The 1 FAIL was a borderline A2 coverage check (0.78, n=100). |
| v3 (`aethel_psi_suite_v3.py`, full) | PASS=40 FAIL=5 OPEN=10 | A2 fixed with a binomial coverage test. γ(M) flat 0.00903; A4 unobservable. |
| v4 (`aethel_psi_suite_v4.py`, full) | PASS=6 FAIL=11 OPEN=8 | Tests the new LaTeX claims: β_A/β_ε derivations FAIL, phantom 0.476 FAILs vs DESI, Fisher 1.001 FAILs (1.225), Λ ≡ 0 FAILs, Gilkey a4 PASS (−1/6). |
| v5 (`aethel_psi_suite_v5.py`, full) | **PASS=25 FAIL=18 OPEN=6 INFO=8** | The five repairs: only ξ = −1/6 closes (under stated D_Ψ). β_A fitted, phantom FAILED, Λ dropped, A4 narrowed to failed-SN cutoff. |

Also here: `test_core.py` (core math), `test_bilby_real.py`
(Bilby waveform, both the verbatim and fixed width versions),
`test_bilby_qiskit.py` (modifier unit checks).

## Run

```bash
pip install -r ../requirements.txt
python aethel_psi_suite_v5.py        # full run; exit 1 is expected
python test_core.py
python test_bilby_real.py
python qiskit-stabilizer/test_qiskit.py   # from repo root; see its README
```

## Caveat

v5 section C0 reads the v3/v4 output files from disk, so run v3
and v4 first (or adjust the C0 paths) before treating a v5 run as
a clean reproduction. v2/v3/v4 files are never overwritten by v5.
The v6 batch (PASS=9 FAIL=6 OPEN=2: spin posteriors, GW200129
XPHM, ACT joint, GW190412 stack) is a separate report, not a
suite file — see the paper's addendum and `AUDIT.md`.
