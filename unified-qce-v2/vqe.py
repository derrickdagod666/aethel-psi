"""Dynamic Quantum Hamiltonian (VQE) module — UNIFIED QCE v2.

This module finally does what the originals only claimed: it COMPILES a
math string into a Hamiltonian. Two input languages are supported, and
the module says which one it used for every expression:

A) Psi-function syntax — e.g. ``V(Psi) = cos(Psi)`` or ``V(Psi) = Psi**2
   + 0.5*Psi``.
   ENCODING (a stated modelling choice, not a law of nature): the
   waveform variable Psi is encoded as the Pauli Z operator on qubit 0.
   Because Z has eigenvalues +1/-1 and Z**2 = I, ANY function f of Psi
   compiles exactly to a*I + b*Z, where
       a = (f(+1) + f(-1)) / 2,   b = (f(+1) - f(-1)) / 2.
   Examples, verified analytically in the headless test:
       cos(Psi)        -> 0.5403*I            (ground state +0.5403)
       Psi**2          -> 1.0000*I            (ground state +1.0000)
       sin(Psi)        -> 0.8415*Z            (ground state -0.8415)
       0.5*Psi + 1     -> 1.0000*I + 0.5*Z    (ground state +0.5000)
       exp(Psi)        -> 1.5431*I + 1.1752*Z (ground state +0.3679)
   Supported: numbers, Psi, +, -, *, /, **, sin, cos, exp, pi.
   The originals did NOT do this: they regex-matched "cos" in the string
   and returned a fixed 2-qubit Hamiltonian while reporting "Parsed: ...".
   That behaviour is kept below as NAMED PRESETS, not as compilation.

B) Pauli-term syntax — e.g. ``0.5*ZZ + 1.0*IX - 0.15*ZI`` or ``ZZ + XX``.
   Each term is coefficient * Pauli string (letters I, X, Y, Z, 1-3
   qubits). Terms of different lengths are padded on the left with I.
   This is the honest way to write a multi-qubit Hamiltonian as text.

C) Named presets (from the originals, kept for compatibility):
       "cos-type":     ZZ + 0.5*IX - 0.15*ZI   (2 qubits)
       "generic-type": ZZ + XX                 (2 qubits)
   These are SELECTED by name in the Quantum Lab tab. They are not the
   result of compiling anything, and the UI says so.

The VQE itself is real in all cases: Aer EstimatorV2 + TwoLocal + COBYLA
on the local simulator, reporting the optimiser's ground-state energy.

NOTE (dependency fix found while testing this build): the pinned stack
inherited from swarm v2 — qiskit 2.2.1 + qiskit-algorithms 0.3.1 — does
NOT work: 0.3.1 imports BaseSampler, which Qiskit 2.x removed. And
0.4.0's VQE calls the estimator with V2-style pubs, so the Aer V1
Estimator fails; EstimatorV2 is required. This build therefore pins
qiskit-algorithms==0.4.0 and uses EstimatorV2 (see requirements.txt).
"""
import re

import numpy as np

_PSI = None  # lazy sympy symbol, created in _psi_symbol()


def _psi_symbol():
    global _PSI
    if _PSI is None:
        import sympy as sp

        _PSI = sp.Symbol("Psi")
    return _PSI


_PAULI_TERM_RE = re.compile(
    r"^\s*([+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)?\s*\*?\s*)?([IXYZ]{1,3})\s*$",
    re.IGNORECASE,
)


def _looks_like_pauli_syntax(expr):
    """True if every +-separated term is coefficient * Pauli string."""
    text = (expr or "").strip()
    if not text:
        return False
    # Split on + / - that are not part of an exponent.
    parts = re.split(r"(?<!e)(?<!E)(?=[+-])", text)
    if not any(p.strip() for p in parts):
        return False
    for part in parts:
        if not part.strip():
            continue
        if not _PAULI_TERM_RE.match(part):
            return False
    return True


def _compile_pauli_syntax(expr):
    from qiskit.quantum_info import SparsePauliOp

    parts = re.split(r"(?<!e)(?<!E)(?=[+-])", expr.strip())
    terms = []
    max_len = 0
    parsed = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = _PAULI_TERM_RE.match(part)
        coeff_str, pauli = m.group(1), m.group(2).upper()
        coeff_str = (coeff_str or "").replace("*", "").strip()
        coeff = float(coeff_str) if coeff_str not in ("", "+", "-") else (
            -1.0 if coeff_str.strip() == "-" else 1.0)
        parsed.append((coeff, pauli))
        max_len = max(max_len, len(pauli))
    for coeff, pauli in parsed:
        terms.append((pauli.rjust(max_len, "I"), coeff))
    op = SparsePauliOp.from_list(terms)
    expl = ("Pauli-term syntax: parsed term-by-term into "
            f"{op.to_list()!r} ({max_len} qubits).")
    return op, expl


def _compile_psi_function(expr):
    import sympy as sp

    from qiskit.quantum_info import SparsePauliOp

    Psi = _psi_symbol()
    try:
        f = sp.sympify(expr, locals={"Psi": Psi, "sin": sp.sin,
                                     "cos": sp.cos, "exp": sp.exp,
                                     "pi": sp.pi})
    except Exception as exc:
        raise ValueError(
            f"Could not parse V(Psi) expression {expr!r}: {exc}. "
            "Use a function of Psi (e.g. cos(Psi), Psi**2 + 0.5*Psi) or "
            "Pauli-term syntax (e.g. 0.5*ZZ + 1.0*IX).")
    if f.free_symbols - {Psi}:
        raise ValueError(
            f"Expression {expr!r} uses symbols other than Psi: "
            f"{sorted(str(s) for s in f.free_symbols - {Psi})}. "
            "Only Psi is supported in function syntax.")
    # Spectral mapping: Z has eigenvalues +/-1, Z**2 = I, so
    # f(Z) = a*I + b*Z with a, b from f evaluated at +/-1. Exact.
    f_plus = sp.N(f.subs(Psi, 1))
    f_minus = sp.N(f.subs(Psi, -1))
    a = float(sp.N((f_plus + f_minus) / 2))
    b = float(sp.N((f_plus - f_minus) / 2))
    terms = []
    if abs(a) > 1e-12:
        terms.append(("I", a))
    if abs(b) > 1e-12:
        terms.append(("Z", b))
    if not terms:
        terms = [("I", 0.0)]
    op = SparsePauliOp.from_list(terms)
    expl = (
        f"Compiled V(Psi) = {expr} with Psi encoded as Pauli Z on qubit 0. "
        f"f(+1) = {float(f_plus):.4f}, f(-1) = {float(f_minus):.4f}, so "
        f"f(Z) = {a:.4f}*I + {b:.4f}*Z (exact: Z**2 = I). "
        f"Analytic ground state: {a - abs(b):.4f} Hartree.")
    return op, expl


def compile_expression(expr):
    """Compile a math string into a Hamiltonian.

    Returns (SparsePauliOp, explanation). Raises ValueError if the
    string is not in either supported syntax.
    """
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Empty expression — nothing to compile.")
    if _looks_like_pauli_syntax(expr):
        return _compile_pauli_syntax(expr)
    return _compile_psi_function(expr)


def select_hamiltonian(math_string):
    """Named PRESET selection (not compilation), kept from the originals.

    Returns (name, SparsePauliOp, explanation).
    """
    from qiskit.quantum_info import SparsePauliOp

    text = (math_string or "").lower()
    if "cos" in text:
        return (
            "cos-type",
            SparsePauliOp.from_list(
                [("ZZ", 1.0), ("IX", 0.5), ("ZI", -0.15)]),
            "PRESET (not compiled): message contains 'cos', so the "
            "cos-type Hamiltonian (ZZ + 0.5*IX - 0.15*ZI) was selected. "
            "To compile an expression for real, use the Quantum Lab tab.",
        )
    return (
        "generic-type",
        SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 1.0)]),
        "PRESET (not compiled): no 'cos' in the message, so the generic "
        "Hamiltonian (ZZ + XX) was selected. To compile an expression for "
        "real, use the Quantum Lab tab.",
    )


def run_vqe(hamiltonian, maxiter=100):
    """Run a real VQE on the Aer simulator. Returns (eigenvalue, result).

    Raises RuntimeError if Qiskit/Aer is unavailable.
    """
    try:
        from qiskit_aer.primitives import EstimatorV2 as Estimator
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.circuit.library import TwoLocal
    except ImportError as exc:
        raise RuntimeError(f"VQE unavailable (Qiskit not installed): {exc}")
    n_qubits = hamiltonian.num_qubits
    estimator = Estimator()
    # .decompose() is required: Aer rejects the un-decomposed TwoLocal
    # library instruction ("unknown instruction: TwoLocal"); the
    # decomposed form is plain RY/CZ gates, which Aer runs natively.
    ansatz = TwoLocal(num_qubits=n_qubits, rotation_blocks="ry",
                      entanglement_blocks="cz").decompose()
    optimizer = COBYLA(maxiter=maxiter)
    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
    return float(np.real(result.eigenvalue)), result
