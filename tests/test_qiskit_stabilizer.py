"""Qiskit stabilizer test — pytest form of qiskit-stabilizer/test_qiskit.py.

Mirrors the verified script (reconstructed Listing 2 from the UAF paper):
3-qubit GHZ spacetime register, 2 ancilla parity checks, X error on one
qubit, syndrome-conditional recovery via Qiskit 2.x ``if_test`` blocks.
Verified live 2026-11-07, Qiskit 2.6.2: PASS 4/4, GHZ 4096/4096 in all
runs. The original script is kept as-is in qiskit-stabilizer/; this file
is the CI-runnable pytest copy (the script runs its simulation at import
time, so it cannot be imported directly).
"""
import pytest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


def build(error_qubit=None):
    qr_space = QuantumRegister(3, 'spacetime')
    qr_psi = QuantumRegister(2, 'stabilizer')
    cr_syn = ClassicalRegister(2, 'syndrome')
    cr_out = ClassicalRegister(3, 'geometry')
    qc = QuantumCircuit(qr_space, qr_psi, cr_syn, cr_out)
    qc.h(qr_space[0])
    qc.cx(qr_space[0], qr_space[1])
    qc.cx(qr_space[0], qr_space[2])
    qc.barrier(label='Coherent')
    if error_qubit is not None:
        qc.x(qr_space[error_qubit])
    qc.barrier(label='Error')
    qc.cx(qr_space[0], qr_psi[0])
    qc.cx(qr_space[1], qr_psi[0])
    qc.cx(qr_space[1], qr_psi[1])
    qc.cx(qr_space[2], qr_psi[1])
    qc.measure(qr_psi, cr_syn)
    qc.barrier(label='Syndrome')
    with qc.if_test((cr_syn, 1)):
        qc.x(qr_space[0])
    with qc.if_test((cr_syn, 3)):
        qc.x(qr_space[1])
    with qc.if_test((cr_syn, 2)):
        qc.x(qr_space[2])
    qc.barrier(label='Healed')
    qc.measure(qr_space, cr_out)
    return qc


@pytest.mark.parametrize(
    "label,error_qubit",
    [
        ("no-error control", None),
        ("error on q1 (paper case)", 1),
        ("error on q0", 0),
        ("error on q2", 2),
    ],
)
def test_stabilizer_recovers_ghz(label, error_qubit):
    sim = AerSimulator()
    qc = build(error_qubit)
    counts = sim.run(transpile(qc, sim), shots=4096).result().get_counts()
    geo = {}
    for k, v in counts.items():
        g = k.split()[0]
        geo[g] = geo.get(g, 0) + v
    good = geo.get('000', 0) + geo.get('111', 0)
    assert good / 4096 == pytest.approx(1.0, abs=0.01), f"{label}: {geo}"
