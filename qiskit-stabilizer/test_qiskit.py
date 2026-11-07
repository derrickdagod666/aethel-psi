"""Qiskit stabilizer test -- reconstructed Listing 2 from the UAF paper.
Run with ./venv_qiskit/bin/python test_qiskit.py

What was WRONG / garbled in the printed listing and what I changed:
1. The PDF text extraction mangles the listing (lines fused, `qc.x(...).c_if`
   fragments out of order). The logic was reconstructed from the prose:
   3-qubit GHZ spacetime register, 2 ancilla parity checks, X error on qubit 1,
   syndrome-conditional recovery.
2. `qc.x(q).c_if(cr_syn, v)` -- the old Qiskit conditional-gate API -- is
   deprecated/removed in Qiskit 2.x. Replaced with `with qc.if_test((cr_syn, v)):`
   blocks, which are the supported Qiskit 2.x form. Same truth table.
3. The listing's syndrome mapping is in fact CORRECT and was kept:
   ancilla0 = q0 XOR q1, ancilla1 = q1 XOR q2, so
     error on q0 -> syndrome 01 (1), error on q1 -> syndrome 11 (3),
     error on q2 -> syndrome 10 (2). Recovery X's the matching qubit.
4. Added: a no-error control run, and error-on-q0 / error-on-q2 variants,
   which the paper does not run. These prove the syndrome table, not just one case.
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import json

def build(error_qubit=None):
    qr_space = QuantumRegister(3, 'spacetime')
    qr_psi = QuantumRegister(2, 'stabilizer')
    cr_syn = ClassicalRegister(2, 'syndrome')
    cr_out = ClassicalRegister(3, 'geometry')
    qc = QuantumCircuit(qr_space, qr_psi, cr_syn, cr_out)
    # GHZ-like coherent state
    qc.h(qr_space[0])
    qc.cx(qr_space[0], qr_space[1])
    qc.cx(qr_space[0], qr_space[2])
    qc.barrier(label='Coherent')
    # injected bit-flip error
    if error_qubit is not None:
        qc.x(qr_space[error_qubit])
    qc.barrier(label='Error')
    # parity syndromes
    qc.cx(qr_space[0], qr_psi[0])
    qc.cx(qr_space[1], qr_psi[0])
    qc.cx(qr_space[1], qr_psi[1])
    qc.cx(qr_space[2], qr_psi[1])
    qc.measure(qr_psi, cr_syn)
    qc.barrier(label='Syndrome')
    # conditional recovery (Qiskit 2.x if_test form)
    with qc.if_test((cr_syn, 1)):
        qc.x(qr_space[0])
    with qc.if_test((cr_syn, 3)):
        qc.x(qr_space[1])
    with qc.if_test((cr_syn, 2)):
        qc.x(qr_space[2])
    qc.barrier(label='Healed')
    qc.measure(qr_space, cr_out)
    return qc

sim = AerSimulator()
results = {}
for label, err in [('no-error control', None), ('error on q1 (paper case)', 1),
                   ('error on q0', 0), ('error on q2', 2)]:
    qc = build(err)
    counts = sim.run(transpile(qc, sim), shots=4096).result().get_counts()
    # counts keys are 'geometry syndrome' (space-separated); geometry is the left field
    geo = {}
    for k, v in counts.items():
        g = k.split()[0]
        geo[g] = geo.get(g, 0) + v
    good = geo.get('000', 0) + geo.get('111', 0)
    print(f"{label}: geometry counts {dict(sorted(geo.items()))} -> "
          f"GHZ states 000+111 = {good}/4096 = {good/4096:.3f}")
    results[label] = {'geometry': geo, 'ghz_fraction': good/4096}

# Verdict: GHZ fraction should be ~1.0 in all four runs if correction works.
ok = all(abs(r['ghz_fraction'] - 1.0) < 0.01 for r in results.values())
print("\nQiskit stabilizer test:", "PASS" if ok else "FAIL",
      "-- all four runs recover the GHZ state" if ok else "-- see fractions above")
json.dump(results, open('results_qiskit.json', 'w'), indent=2)
print("Wrote results_qiskit.json")
