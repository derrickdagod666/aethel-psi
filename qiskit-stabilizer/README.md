# qiskit-stabilizer — 3-qubit stabilizer demo

Reconstruction of Listing 2 from the UAF paper: a 3-qubit GHZ
("spacetime") register, 2 ancilla parity checks, a single bit-flip
error, syndrome-conditional recovery.

## What was fixed vs the printed listing

The PDF text extraction garbles the listing. The logic was
reconstructed from the prose, and one API change was made:
`qc.x(q).c_if(cr_syn, v)` (deprecated/removed in Qiskit 2.x) was
replaced with `with qc.if_test((cr_syn, v)):` blocks — same truth
table. The syndrome mapping was verified correct and kept:
ancilla0 = q0⊕q1, ancilla1 = q1⊕q2, so error on q0 → syndrome 01,
on q1 → 11, on q2 → 10.

## Run

```bash
pip install qiskit==2.6.2 qiskit-aer
python test_qiskit.py
```

## Verified result (re-run 2026-11-07)

```
no-error control:      GHZ 4096/4096 = 1.000
error on q1 (paper):   GHZ 4096/4096 = 1.000
error on q0:           GHZ 4096/4096 = 1.000
error on q2:           GHZ 4096/4096 = 1.000
Qiskit stabilizer test: PASS -- all four runs recover the GHZ state
```

## Honesty note

This is a 3-qubit repetition code. It proves the syndrome table,
not cosmic unitarity, not a "holographic cosmic stabilizer".
Do not describe it as more than a demo.
