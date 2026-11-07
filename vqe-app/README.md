# VQE drug-discovery app — Dash + Plotly + PennyLane

## Provenance — read this first

**This is NOT Putt's original VQE app.** The original Dash/Plotly/
PennyLane VQE drug-discovery app was searched for hard in Google
Drive and the local workspace on 2026-11-07 (`CODE_HUNT_REPORT.md`:
all 1,097 Drive items by name, full-text search, content grep of all
119 downloadable code/text files, all 4 ZIPs, all 3 Colab notebooks)
and was **not found**. The only record of it is the description in
his Google Keep note "yes 🚨" (Dash + Plotly + PennyLane + VQE +
drug discovery).

What is in this directory is a **new implementation built from that
spec** (written 2026-11-07, during the paid-tier run). Its headers
label exactly what is reused from Putt's Drive code and what is
newly written:

- **Reused pattern (labelled in `vqe_core.py`):** the learnable-Psi
  idea from `drive-code/quantum_cognitive_pennylane.py` (a PennyLane
  QNode with a learnable `psi` encoded as `qml.RY(psi)`) — used here
  ONLY as an optional, default-OFF initialisation of the VQE ansatz
  weights (`psi_overlay`, seeding the first layer's RY angles from
  Psi(r) = 0.1·cos(470·r)). An initialisation choice, not a physical
  coupling; the Hamiltonian is standard quantum chemistry.
- **Newly written:** molecular Hamiltonians via
  `qml.qchem.molecular_hamiltonian` (STO-3G, active spaces stated
  per molecule), the VQE loop, the exact-diagonalisation reference,
  the bond-length scan, and all Dash/Plotly UI code in `app.py`.
- **Not reused:** `drive-code/quantum_tests_harmonic_full.py`
  contributed nothing runnable (its `system` object is undefined);
  recorded in `vqe_core.py` so the reuse claim is checkable.
- **3D viewer:** Plotly 3D scatter of atom positions, labelled as
  such. No 3Dmol.js is used — no 3Dmol.js code or pattern was found
  in Putt's files (the Apothegm note "0001" describes a 3Dmol.js
  viewer, but that code was not found either).

## What it does

Pick a small molecule (H2, LiH, H2O, NH3), run a PennyLane VQE on
the analytic simulator, compare to exact diagonalisation, scan the
bond length, and view the molecule in 3D.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install pennylane dash plotly numpy
python test_vqe.py   # headless test, no Dash server needed
python app.py        # then open http://127.0.0.1:8050
```

## Honest limits

- Simulator only (PennyLane `default.qubit`, analytic,
  shots=None). No real quantum hardware, no drug-efficacy claims.
- STO-3G minimal basis, active spaces stated per molecule in
  `vqe_core.py` (H2 full STO-3G 4 qubits; LiH (2e,2o) 4 qubits;
  H2O/NH3 (4e,4o) 8 qubits). Energies are electronic
  ground-state estimates, not production drug-discovery results.
- Tolerances in `test_vqe.py` are honest for a hardware-efficient
  ansatz on an analytic simulator (H2 0.01, LiH 0.02, H2O/NH3
  0.05 Ha); chemical accuracy (0.0016 Ha) is NOT claimed.
- The Ψ overlay is a note-inspired initialisation/visualisation
  only (default OFF). It does not change any Hamiltonian or energy
  except through the initial ansatz weights when enabled.

## Test status — FAIL (verified 2026-11-07, do not cite as passing)

Headless test re-run by the coordinator in this app's own venv
(PennyLane 0.45.1, ~12 min): **RESULT: FAIL.** H2 PASS (err
+0.000022 Ha, tol 0.010); LiH PASS (err +0.000004 Ha, tol
0.020); **H2O FAIL** (err +0.209526 Ha, tol 0.050); **NH3 FAIL**
(err +0.307579 Ha, tol 0.050). H2 bond scan: only r=0.50 Å
PASS; r=0.74, 1.00, 1.40, 2.00 Å FAIL. The 2-layer
hardware-efficient ansatz at 120 steps is not expressive
enough for the 8-qubit molecules — more layers/steps or a
better ansatz is needed. This is recorded here and in
`PAID_TIER_REPORT.md` §5, not hidden.

If the original VQE app source is ever located, it should
replace this implementation and this README must be updated
to say so.
