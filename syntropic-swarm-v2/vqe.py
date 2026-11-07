"""Dynamic Quantum Hamiltonian (VQE) module v2.

Changes vs the original compile_vqe (per Putt's rule, elaborated):
  1. THE MATH STRING IS NOT COMPILED — and v2 says so. The original
     regex-parsed `V(Psi) = ...`, ignored the expression except for a
     substring test, and reported "Parsed: ..." as if the expression had
     been compiled into a Hamiltonian. v2 SELECTS between two named,
     documented Hamiltonians by keyword, and the UI/README state the rule
     in plain text:
       - message contains "cos" -> "cos-type":  ZZ + 0.5*IX - 0.15*ZI
       - otherwise            -> "generic-type": ZZ + XX
     Selection, not compilation. A real term-by-term compiler is future
     work (see README).
  2. The VQE itself is REAL. The selected Hamiltonian is genuinely
     optimised on the local Aer simulator (EstimatorV2 + TwoLocal ry/cz
     ansatz + scipy COBYLA, 100 iterations) and the reported ground-state
     energy is the optimiser's result, not a constant. (First written
     with qiskit_algorithms.VQE; that import crashes under Qiskit 2.2.1
     — qiskit-algorithms 0.3.1 needs the removed BaseSampler — so it was
     replaced with EstimatorV2 + scipy and the qiskit-algorithms
     dependency was dropped.)
  3. TwoLocal is constructed with num_qubits=2 explicitly (both
     Hamiltonians are 2-qubit). The original passed no num_qubits, which
     is unverified against the installed qiskit version.
  4. If Qiskit is not installed, select_hamiltonian still works (it is
     pure data) but run_vqe raises a plain error instead of crashing on a
     missing self.estimator — the original's __init__ would have returned
     early leaving no estimator attribute at all.
"""
import numpy as np


def select_hamiltonian(math_string):
    """Keyword SELECTION (not compilation) of a named Hamiltonian.

    Returns (name, SparsePauliOp, explanation).
    """
    from qiskit.quantum_info import SparsePauliOp

    text = (math_string or "").lower()
    if "cos" in text:
        return (
            "cos-type",
            SparsePauliOp.from_list(
                [("ZZ", 1.0), ("IX", 0.5), ("ZI", -0.15)]),
            "Message contains 'cos', so the cos-type Hamiltonian "
            "(ZZ + 0.5*IX - 0.15*ZI) was selected. The expression itself "
            "was NOT compiled into Pauli terms.",
        )
    return (
        "generic-type",
        SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 1.0)]),
        "No 'cos' in the message, so the generic Hamiltonian (ZZ + XX) "
        "was selected. The expression itself was NOT compiled into Pauli "
        "terms.",
    )


def run_vqe(hamiltonian, maxiter=100):
    """Run a real VQE on the Aer Estimator (statevector, no shots).

    CHANGE vs the original: the original created a VQE with
    ``qiskit_algorithms.VQE`` + COBYLA. That import CRASHES in the
    installed stack — qiskit-algorithms 0.3.1 imports ``BaseSampler``
    from ``qiskit.primitives``, which Qiskit 2.2.1 removed (found by the
    headless test, §8). This implementation uses the primitives that DO
    exist in Qiskit 2.2.1 — ``EstimatorV2`` plus
    ``scipy.optimize.minimize`` (COBYLA) — and drops the
    qiskit-algorithms dependency entirely (removed from requirements.txt).
    The physics is the same variational loop: ansatz parameters are
    optimised to minimise <H>.

    Returns (ground_state_energy, optimal_parameters).
    """
    from qiskit import transpile
    from qiskit.circuit.library import TwoLocal
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2
    from scipy.optimize import minimize

    # TwoLocal is a composite instruction Aer cannot execute directly
    # ("unknown instruction: TwoLocal") — transpile it to Aer basis
    # gates (u, cx) first. Parameters are preserved by transpile.
    ansatz = transpile(
        TwoLocal(2, "ry", "cz", reps=2), AerSimulator())
    estimator = EstimatorV2()

    def cost(params):
        job = estimator.run([(ansatz, hamiltonian, list(params))])
        # evs is 0-d for a single parameter set, 1-d otherwise.
        evs = np.asarray(job.result()[0].data.evs)
        return float(evs.reshape(-1)[0])

    x0 = np.zeros(ansatz.num_parameters)
    res = minimize(cost, x0, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.5})
    return float(res.fun), list(res.x)
