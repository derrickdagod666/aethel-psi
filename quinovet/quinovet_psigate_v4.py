"""
Quinovet / Universal PsiGate v4
-------------------------------
A NumPy-only research prototype for:
  * a parameterized SU(2) PsiGate,
  * pure-state quantum Fisher information (QFIM),
  * the standard nine-qubit Shor [[9,1,3]] code,
  * single-qubit Pauli syndrome decoding with repeated syndrome rounds,
  * transversal logical CNOT between two Shor-code blocks,
  * an ancilla-referenced geometric-phase interferometry primitive,
  * the Berry phase of the PsiGate control loop, computed directly.

Scope:
  This is a statevector/reference simulator, not a hardware controller.
  It does not implement lattice surgery, fault-tolerant syndrome extraction,
  pulse-level control, leakage, or a physical quantum device.

Changes in v4 (over the audited v3):
  * The Berry phase of the canonical (A, Theta) control loop is now
    actually computed (Pancharatnam product over the loop's state path),
    with a convergence self-test. Reference value: gamma = +2.2214 rad,
    the converged simulation value; the research note's area-integral
    figure (+2.2234 rad) carries ~0.002 rad of grid error and is not
    used as the test target.
  * The ancilla interferometry primitive is unchanged, but its role is
    stated precisely where it is defined: it measures <psi|U|psi> for a
    supplied propagator. By itself it is NOT a Berry-phase measurement;
    a hardware Berry test needs an adiabatic controlled propagator,
    which this module does not build.
  * psigate_qfim_closed_form() added: the symbolically verified metric
    of the PsiGate family, with a self-test that the numerical QFIM
    agrees with it. The generic Euler SU(2) gate is renamed euler_su2
    (psi_gate kept as an alias) and psigate() is an explicit alias of
    axis_rotation, so the PsiGate can no longer be confused with an
    arbitrary SU(2) gate.
  * correct_single_pauli_error() now VERIFIES its own result: the
    corrected state must lie in the code space (every stabilizer acts
    as +1), and CorrectionResult.corrected reports that verification
    instead of being hard-coded True.
  * Stronger self-tests: the transversal CNOT's logical truth table
    (including entanglement generation), and an end-to-end noisy-rounds
    correction test at a fixed seed.
  * run_tests() added in the qce-improved package format, alongside the
    original run_self_tests().
  * Multi-error extension (added after an audit of claims made for
    9-qubit codes): a collision-checked decoder for the provably
    correctable "favourable distribution" family -- X errors at most
    one per 3-qubit block, Z errors supported inside a single block --
    with self-tests proving the patterns AND the boundary where
    correctability ends. The single-error decoder is unchanged; these
    patterns were always correctable by the code, but the v3 lookup
    could not decode them (it raised, correctly, rather than guess).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math
import numpy as np


# ---------------------------------------------------------------------------
# 1. PsiGate: arbitrary single-qubit SU(2) rotations
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def euler_su2(theta: float, phi: float, lam: float) -> np.ndarray:
    """U = Rz(phi) Ry(theta) Rz(lam), an arbitrary SU(2) gate.

    This is a generic Euler gate, NOT the PsiGate. It is kept for
    completeness; psi_gate remains as a backwards-compatible alias.
    """
    return rz(phi) @ ry(theta) @ rz(lam)


# Backwards-compatible alias (v3 name).
psi_gate = euler_su2


def axis_rotation(angle: float, polar_angle: float) -> np.ndarray:
    """Rotation exp[-i angle/2 (sin(polar) X + cos(polar) Z)]."""
    n = math.sin(polar_angle) * X + math.cos(polar_angle) * Z
    return math.cos(angle / 2) * I2 - 1j * math.sin(angle / 2) * n


# The PsiGate of the Aethel-Psi work: PsiGate(A, Theta) is exactly this
# axis rotation -- angle A about n(Theta) = (sin Theta, 0, cos Theta).
psigate = axis_rotation


def assert_unitary(U: np.ndarray, atol: float = 1e-10) -> None:
    if U.shape != (2, 2) or not np.allclose(U.conj().T @ U, I2, atol=atol):
        raise ValueError("Expected a 2x2 unitary matrix.")


# ---------------------------------------------------------------------------
# 2. Pure-state QFIM and task-aware objective
# ---------------------------------------------------------------------------

def pure_state_qfim(
    state_fn,
    parameters: Sequence[float],
    step: float = 1e-6,
) -> np.ndarray:
    """
    Numerically evaluate the pure-state QFIM:
      F_ij = 4 Re(<d_i psi|d_j psi> -
                  <d_i psi|psi><psi|d_j psi>).
    The input state_fn(parameters) must return a normalized statevector.
    """
    p = np.asarray(parameters, dtype=float)
    psi = np.asarray(state_fn(p), dtype=complex)
    if not np.isclose(np.vdot(psi, psi).real, 1.0, atol=1e-7):
        raise ValueError("state_fn must return a normalized statevector.")

    derivatives = []
    for i in range(len(p)):
        plus, minus = p.copy(), p.copy()
        plus[i] += step
        minus[i] -= step
        derivatives.append(
            (np.asarray(state_fn(plus)) - np.asarray(state_fn(minus))) / (2 * step)
        )

    F = np.zeros((len(p), len(p)), dtype=float)
    for i, di in enumerate(derivatives):
        for j, dj in enumerate(derivatives):
            cov = np.vdot(di, dj) - np.vdot(di, psi) * np.vdot(psi, dj)
            F[i, j] = 4.0 * cov.real
    return 0.5 * (F + F.T)


def psigate_qfim_closed_form(A: float, Theta: float) -> np.ndarray:
    """
    Symbolically verified QFIM of |psi(A, Theta)> = PsiGate(A, Theta)|0>,
    in the parameter order (A, Theta):

        G_AA     = sin^2(Theta)
        G_ATheta = sin(A) sin(Theta) cos(Theta)
        G_ThetaTheta = 2(1 - cos A) - sin^2(A) sin^2(Theta)

    Verified analytically and against finite-difference SLD evaluation;
    det G = 4 sin^2(Theta) sin^4(A/2), so the family is unidentifiable
    on the loci sin(Theta) = 0 and A = 0 (mod 2 pi).
    """
    sA, cA = math.sin(A), math.cos(A)
    sT, cT = math.sin(Theta), math.cos(Theta)
    return np.array(
        [[sT * sT, sA * sT * cT],
         [sA * sT * cT, 2.0 * (1.0 - cA) - sA * sA * sT * sT]],
        dtype=float,
    )


def qfi_gate_objective(
    gate_fidelity: float,
    qfi_gain: float,
    leakage: float = 0.0,
    crosstalk: float = 0.0,
    duration: float = 0.0,
    weights: Dict[str, float] | None = None,
    minimum_fidelity: float = 0.99,
) -> float:
    """
    Lower is better. QFI is rewarded only after a hard fidelity constraint.
    Inputs are normalized scores in [0,1], except duration (nonnegative).
    """
    w = weights or {
        "fidelity": 1.0, "leakage": 1.0, "crosstalk": 0.5,
        "duration": 0.01, "qfi": 0.1
    }
    if not 0.0 <= gate_fidelity <= 1.0:
        raise ValueError("gate_fidelity must be in [0,1].")
    if gate_fidelity < minimum_fidelity:
        return float("inf")
    return (
        w["fidelity"] * (1.0 - gate_fidelity)
        + w["leakage"] * leakage
        + w["crosstalk"] * crosstalk
        + w["duration"] * duration
        - w["qfi"] * qfi_gain
    )


# ---------------------------------------------------------------------------
# 3. Nine-qubit Shor code [[9,1,3]]
# ---------------------------------------------------------------------------

N_QUBITS = 9

# Binary symplectic Pauli convention:
# P = i^k X^x Z^z; x,z are 9-bit tuples. Global phase is irrelevant
# for stabilizer commutation and syndrome calculation.
Pauli = Tuple[Tuple[int, ...], Tuple[int, ...]]


def pauli_from_ops(ops: Dict[int, str]) -> Pauli:
    x, z = [0] * N_QUBITS, [0] * N_QUBITS
    for q, op in ops.items():
        if op in ("X", "Y"):
            x[q] = 1
        if op in ("Z", "Y"):
            z[q] = 1
        if op not in ("X", "Y", "Z"):
            raise ValueError(f"Unsupported Pauli {op!r}")
    return tuple(x), tuple(z)


def pauli_weight(p: Pauli) -> int:
    x, z = p
    return sum(1 for a, b in zip(x, z) if a or b)


def symplectic_product(a: Pauli, b: Pauli) -> int:
    ax, az = a
    bx, bz = b
    return (sum(i * j for i, j in zip(ax, bz))
            + sum(i * j for i, j in zip(az, bx))) % 2


STABILIZERS: Tuple[Pauli, ...] = (
    pauli_from_ops({0: "Z", 1: "Z"}),
    pauli_from_ops({1: "Z", 2: "Z"}),
    pauli_from_ops({3: "Z", 4: "Z"}),
    pauli_from_ops({4: "Z", 5: "Z"}),
    pauli_from_ops({6: "Z", 7: "Z"}),
    pauli_from_ops({7: "Z", 8: "Z"}),
    pauli_from_ops({0: "X", 1: "X", 2: "X", 3: "X", 4: "X", 5: "X"}),
    pauli_from_ops({3: "X", 4: "X", 5: "X", 6: "X", 7: "X", 8: "X"}),
)

# For |0_L> = |GHZ+>^tensor3 and |1_L> = |GHZ->^tensor3:
# logical X = Z^tensor9; logical Z = X^tensor9.
LOGICAL_X = pauli_from_ops({q: "Z" for q in range(9)})
LOGICAL_Z = pauli_from_ops({q: "X" for q in range(9)})


def syndrome(error: Pauli) -> Tuple[int, ...]:
    return tuple(symplectic_product(error, s) for s in STABILIZERS)


def stabilizer_row_span() -> set:
    """All binary Pauli vectors generated by the eight stabilizers."""
    span = set()
    for mask in range(1 << len(STABILIZERS)):
        x = [0] * N_QUBITS
        z = [0] * N_QUBITS
        for j, generator in enumerate(STABILIZERS):
            if (mask >> j) & 1:
                gx, gz = generator
                x = [a ^ b for a, b in zip(x, gx)]
                z = [a ^ b for a, b in zip(z, gz)]
        span.add((tuple(x), tuple(z)))
    return span


STABILIZER_SPAN = stabilizer_row_span()


def pauli_difference(a: Pauli, b: Pauli) -> Pauli:
    """Binary representation of a*b, ignoring its irrelevant global phase."""
    ax, az = a
    bx, bz = b
    return (tuple(x ^ y for x, y in zip(ax, bx)),
            tuple(x ^ y for x, y in zip(az, bz)))


def build_single_error_lookup() -> Dict[Tuple[int, ...], Pauli]:
    """
    Derive a minimum-weight representative for every correctable syndrome.
    Some physical errors intentionally share a syndrome: they differ by a
    stabilizer and therefore act identically on the code space up to phase.
    """
    identity = pauli_from_ops({})
    lookup: Dict[Tuple[int, ...], Pauli] = {syndrome(identity): identity}
    for q in range(N_QUBITS):
        for op in ("X", "Y", "Z"):
            err = pauli_from_ops({q: op})
            syn = syndrome(err)
            if syn in lookup:
                if pauli_difference(err, lookup[syn]) not in STABILIZER_SPAN:
                    raise RuntimeError(
                        "Syndrome collision between inequivalent errors; "
                        "the proposed decoder is not valid."
                    )
                continue
            lookup[syn] = err
    return lookup


SINGLE_ERROR_LOOKUP = build_single_error_lookup()


# ---------------------------------------------------------------------------
# 3b. The favourable multi-error family (beyond the single-error lookup)
# ---------------------------------------------------------------------------

def favourable_family_patterns() -> List[Pauli]:
    """
    The provably correctable "favourable distribution" family:

      * X part: at most one X in each 3-qubit block (4^3 = 64 patterns,
        weights 0..3) -- bit flips are located block by block from the
        six Z-type stabilizers, so one flip per block is identifiable
        even when all three blocks are hit at once.
      * Z part: supported inside a SINGLE block, any subset of it
        (22 distinct patterns including the identity) -- within one
        block, every Z pattern is stabilizer-equivalent to the identity
        (even weight) or to a single Z (odd weight). That equivalence
        is Shor degeneracy, and it is what "multiple phase errors in
        one block" really reduces to.

    Combined: 64 x 22 = 1408 patterns (X and Z parts merge into Y on
    shared qubits). NOT claimed correctable: anything whose Z part
    spreads across two or three blocks -- one Z in each block is a
    logical operator, and patterns there collide with single-block
    patterns modulo a logical (see the boundary self-test). The
    distance-3 guarantee is untouched by all of this: it promises
    correction of ANY single error, and this family is the extra,
    distribution-dependent structure on top of it.
    """
    blocks = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    x_choices = [[None, *block] for block in blocks]
    z_parts = {pauli_from_ops({})}
    for block in blocks:
        for mask in range(1, 1 << 3):
            ops = {block[i]: "Z" for i in range(3) if (mask >> i) & 1}
            z_parts.add(pauli_from_ops(ops))
    patterns: List[Pauli] = []
    for c0 in x_choices[0]:
        for c1 in x_choices[1]:
            for c2 in x_choices[2]:
                ops = {}
                for c in (c0, c1, c2):
                    if c is not None:
                        ops[c] = "X"
                x_part = pauli_from_ops(ops)
                for z_part in z_parts:
                    patterns.append(pauli_difference(x_part, z_part))
    return patterns


def build_favourable_family_lookup() -> Dict[Tuple[int, ...], Pauli]:
    """
    Minimum-weight representative per syndrome over the favourable
    family, under the same discipline as the single-error lookup: if
    two family members share a syndrome but are NOT stabilizer-
    equivalent, the family is not uniquely decodable and this RAISES
    rather than guessing. The builder succeeding at import is itself
    the proof that the family is correctable.
    """
    lookup: Dict[Tuple[int, ...], Pauli] = {}
    for err in favourable_family_patterns():
        syn = syndrome(err)
        if syn in lookup:
            if pauli_difference(err, lookup[syn]) not in STABILIZER_SPAN:
                raise RuntimeError(
                    "Favourable-family syndrome collision between "
                    "inequivalent errors; the family is not decodable."
                )
            if pauli_weight(err) < pauli_weight(lookup[syn]):
                lookup[syn] = err
            continue
        lookup[syn] = err
    return lookup


FAVOURABLE_FAMILY_LOOKUP = build_favourable_family_lookup()


def apply_pauli(state: np.ndarray, pauli: Pauli) -> np.ndarray:
    """Apply a 9-qubit Pauli to a statevector using little-endian qubit indices."""
    state = np.asarray(state, dtype=complex)
    if state.size != 2**N_QUBITS:
        raise ValueError("Expected a 9-qubit statevector of length 512.")
    x, z = pauli
    out = np.empty_like(state)
    for basis in range(state.size):
        target = basis
        parity = 0
        for q in range(N_QUBITS):
            bit = (basis >> q) & 1
            if x[q]:
                target ^= (1 << q)
            if z[q] and bit:
                parity ^= 1
        out[target] = (-1 if parity else 1) * state[basis]
    return out


def ghz_state(sign: int) -> np.ndarray:
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1.")
    state = np.zeros(8, dtype=complex)
    state[0] = 1 / math.sqrt(2)
    state[7] = sign / math.sqrt(2)
    return state


def shor_logical_basis() -> Tuple[np.ndarray, np.ndarray]:
    """Return the two 512-amplitude logical basis states."""
    zero3, one3 = ghz_state(+1), ghz_state(-1)
    zero_l = np.kron(np.kron(zero3, zero3), zero3)
    one_l = np.kron(np.kron(one3, one3), one3)
    return zero_l, one_l


def shor_encode(alpha: complex = 1.0, beta: complex = 0.0) -> np.ndarray:
    norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
    if norm == 0:
        raise ValueError("Logical amplitudes cannot both be zero.")
    zero_l, one_l = shor_logical_basis()
    return (alpha * zero_l + beta * one_l) / norm


def decode_single_qubit_error(
    measured_syndrome: Sequence[int],
) -> Pauli:
    key = tuple(int(v) for v in measured_syndrome)
    if key not in SINGLE_ERROR_LOOKUP:
        raise ValueError(
            "Syndrome is outside the identity/single-Pauli lookup; "
            "do not silently guess a correction."
        )
    return SINGLE_ERROR_LOOKUP[key]


def majority_vote_syndromes(
    rounds: Sequence[Sequence[int]],
) -> Tuple[int, ...]:
    """Bitwise majority vote over an odd number of complete syndrome rounds."""
    if not rounds or len(rounds) % 2 == 0:
        raise ValueError("Supply a nonempty odd number of syndrome rounds.")
    width = len(STABILIZERS)
    if any(len(r) != width for r in rounds):
        raise ValueError("Each round must contain eight stabilizer outcomes.")
    return tuple(
        int(sum(int(r[j]) for r in rounds) > len(rounds) // 2)
        for j in range(width)
    )


def simulate_syndrome_measurement(
    true_syndrome: Sequence[int],
    measurement_error_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[int, ...]:
    if not 0 <= measurement_error_probability <= 1:
        raise ValueError("measurement_error_probability must be in [0,1].")
    rng = rng or np.random.default_rng()
    return tuple(
        int(bit) ^ int(rng.random() < measurement_error_probability)
        for bit in true_syndrome
    )


@dataclass
class CorrectionResult:
    state: np.ndarray
    measured_syndrome: Tuple[int, ...]
    correction: Pauli
    corrected: bool


def state_in_codespace(state: np.ndarray) -> bool:
    """
    True iff every stabilizer acts as +1 on the state, i.e. the state
    lies in the code space. Note what this does and does not prove: a
    residual LOGICAL operator also passes this check, so it verifies
    return to the code space, not recovery of the original logical
    state. Only a logical-state comparison (possible in simulation,
    impossible from syndromes alone) proves the latter.
    """
    return all(
        np.allclose(apply_pauli(state, s), state, atol=1e-10)
        for s in STABILIZERS
    )


def correct_single_pauli_error(
    state: np.ndarray,
    rounds: Sequence[Sequence[int]],
) -> CorrectionResult:
    voted = majority_vote_syndromes(rounds)
    correction = decode_single_qubit_error(voted)
    corrected_state = apply_pauli(state, correction)
    # v4: `corrected` is the result of verifying the corrected state is
    # back in the code space -- it is no longer hard-coded True. An
    # unknown syndrome still raises (fail loud, see the decoder).
    verified = state_in_codespace(corrected_state)
    return CorrectionResult(corrected_state, voted, correction, verified)


def decode_favourable_family(measured_syndrome: Sequence[int]) -> Pauli:
    key = tuple(int(v) for v in measured_syndrome)
    if key not in FAVOURABLE_FAMILY_LOOKUP:
        raise ValueError(
            "Syndrome is outside the favourable multi-error family; "
            "do not silently guess a correction."
        )
    return FAVOURABLE_FAMILY_LOOKUP[key]


def correct_favourable_pattern(
    state: np.ndarray,
    rounds: Sequence[Sequence[int]],
) -> CorrectionResult:
    """
    Majority-vote the rounds, decode within the favourable multi-error
    family, correct, and verify the result lies in the code space --
    the same verification (and the same residual-logical caveat) as
    correct_single_pauli_error.
    """
    voted = majority_vote_syndromes(rounds)
    correction = decode_favourable_family(voted)
    corrected_state = apply_pauli(state, correction)
    verified = state_in_codespace(corrected_state)
    return CorrectionResult(corrected_state, voted, correction, verified)


# ---------------------------------------------------------------------------
# 4. Two-block transversal logical CNOT (not lattice surgery)
# ---------------------------------------------------------------------------

def apply_cnot_statevector(
    state: np.ndarray, control: int, target: int, n_qubits: int
) -> np.ndarray:
    if control == target or min(control, target) < 0 or max(control, target) >= n_qubits:
        raise ValueError("Invalid CNOT qubit indices.")
    if state.size != 2**n_qubits:
        raise ValueError("Statevector length does not match n_qubits.")
    out = state.copy()
    c_mask, t_mask = 1 << control, 1 << target
    for basis in range(state.size):
        if (basis & c_mask) and not (basis & t_mask):
            partner = basis | t_mask
            out[basis], out[partner] = out[partner], out[basis]
    return out


def transversal_shor_logical_cnot(
    two_block_state: np.ndarray,
) -> np.ndarray:
    """
    Apply physical CNOTs q -> q+9 for q=0..8.
    This is the transversal logical CNOT for the CSS Shor code.
    Input is an 18-qubit statevector (262144 amplitudes).
    """
    state = np.asarray(two_block_state, dtype=complex)
    if state.size != 2**18:
        raise ValueError("Expected an 18-qubit / 262144-amplitude statevector.")
    out = state.copy()
    for q in range(9):
        out = apply_cnot_statevector(out, q, q + 9, 18)
    return out


# ---------------------------------------------------------------------------
# 5. Ancilla-referenced phase interferometry
# ---------------------------------------------------------------------------

@dataclass
class InterferometryResult:
    relative_phase_rad: float
    visibility: float
    p_plus_x: float
    p_plus_y: float
    overlap: complex


def ancilla_phase_reference(
    initial_system_state: np.ndarray,
    loop_unitary: np.ndarray,
) -> InterferometryResult:
    """
    Ideal ancilla-controlled-U interferometry.
    The ancilla coherence is <psi|U|psi>; its argument is a relative phase.
    This does not claim that an isolated global phase is observable.

    Role, stated precisely (v4): this measures the coherence of a
    SUPPLIED propagator U. It is not by itself a Berry-phase
    measurement -- no loop propagator is constructed here. For the
    Berry phase of the PsiGate control loop as a state path, use
    berry_phase_pancharatnam() below; a hardware Berry test would need
    an adiabatic controlled propagator, which this module does not
    build.
    """
    psi = np.asarray(initial_system_state, dtype=complex)
    U = np.asarray(loop_unitary, dtype=complex)
    if U.shape != (psi.size, psi.size):
        raise ValueError("loop_unitary dimension must match the system state.")
    if not np.isclose(np.vdot(psi, psi).real, 1.0, atol=1e-8):
        raise ValueError("Initial system state must be normalized.")
    if not np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-8):
        raise ValueError("loop_unitary must be unitary.")
    overlap = np.vdot(psi, U @ psi)
    return InterferometryResult(
        relative_phase_rad=float(np.angle(overlap)),
        visibility=float(abs(overlap)),
        p_plus_x=float((1.0 + overlap.real) / 2.0),
        p_plus_y=float((1.0 + overlap.imag) / 2.0),
        overlap=complex(overlap),
    )


# ---------------------------------------------------------------------------
# 6. Berry phase of the PsiGate control loop
# ---------------------------------------------------------------------------

# Converged simulation reference for the loop below (Pancharatnam
# product, stable from 1600 to 6400 loop states). The research note's
# area-integral value is +2.2234 rad; the ~0.002 rad difference is the
# note's grid error, documented when this loop was first simulated.
BERRY_PHASE_REFERENCE_RAD = 2.2214


def psigate_loop_points(n_per_edge: int) -> List[Tuple[float, float]]:
    """
    (A, Theta) points around the canonical control loop,
    counterclockwise: Theta = pi/4 with A: 0 -> pi; A = pi with
    Theta: pi/4 -> 3pi/4; Theta = 3pi/4 with A: pi -> 0; A = 0 with
    Theta: 3pi/4 -> pi/4 (the A = 0 edge collapses to |0>, because
    A = 0 is an unidentifiability locus of the family).
    """
    if n_per_edge < 4:
        raise ValueError("n_per_edge must be at least 4.")
    pts: List[Tuple[float, float]] = []
    for A in np.linspace(0, np.pi, n_per_edge, endpoint=False):
        pts.append((float(A), np.pi / 4))
    for T in np.linspace(np.pi / 4, 3 * np.pi / 4, n_per_edge, endpoint=False):
        pts.append((np.pi, float(T)))
    for A in np.linspace(np.pi, 0, n_per_edge, endpoint=False):
        pts.append((float(A), 3 * np.pi / 4))
    for T in np.linspace(3 * np.pi / 4, np.pi / 4, n_per_edge, endpoint=False):
        pts.append((0.0, float(T)))
    return pts


def berry_phase_pancharatnam(n_per_edge: int = 400) -> float:
    """
    Berry phase of |psi(A, Theta)> = PsiGate(A, Theta)|0> around the
    canonical loop, from the gauge-invariant Pancharatnam product

        gamma = -arg prod_k <psi_k | psi_{k+1}>     (cyclic).

    No area integral is taken anywhere: this is the path definition of
    the phase. It converges to +2.2214 rad for this loop (see
    BERRY_PHASE_REFERENCE_RAD). Scope: exact statevector simulation;
    no hardware Berry run has been performed.
    """
    zero = np.array([1.0, 0.0], dtype=complex)
    states = [psigate(A, T) @ zero for A, T in psigate_loop_points(n_per_edge)]
    prod = 1.0 + 0.0j
    n = len(states)
    for k in range(n):
        prod *= np.vdot(states[k], states[(k + 1) % n])
    return float(-np.angle(prod))


# ---------------------------------------------------------------------------
# 7. Self-checks / reproducibility
# ---------------------------------------------------------------------------

def run_self_tests() -> Dict[str, bool]:
    tests: Dict[str, bool] = {}

    # PsiGate unitarity and determinant-one condition.
    U = psi_gate(0.37, -0.91, 1.22)
    tests["psi_gate_unitary"] = bool(np.allclose(U.conj().T @ U, I2, atol=1e-10))
    tests["psi_gate_det_one"] = bool(np.isclose(np.linalg.det(U), 1.0, atol=1e-10))

    # Stabilizer commutation and logical operator algebra.
    tests["stabilizers_commute"] = all(
        symplectic_product(STABILIZERS[i], STABILIZERS[j]) == 0
        for i in range(8) for j in range(i + 1, 8)
    )
    tests["logical_ops_commute_with_stabilizers"] = all(
        symplectic_product(op, s) == 0
        for op in (LOGICAL_X, LOGICAL_Z) for s in STABILIZERS
    )
    tests["logical_ops_anticommute"] = (
        symplectic_product(LOGICAL_X, LOGICAL_Z) == 1
    )
    tests["single_error_syndromes_cover_equivalence_classes"] = len(SINGLE_ERROR_LOOKUP) == 22

    zero_l, one_l = shor_logical_basis()
    tests["logical_states_orthonormal"] = (
        np.isclose(np.vdot(zero_l, zero_l), 1.0)
        and np.isclose(np.vdot(one_l, one_l), 1.0)
        and np.isclose(np.vdot(zero_l, one_l), 0.0)
    )
    tests["logical_zero_stabilized"] = all(
        np.allclose(apply_pauli(zero_l, s), zero_l, atol=1e-10)
        for s in STABILIZERS
    )
    tests["logical_one_stabilized"] = all(
        np.allclose(apply_pauli(one_l, s), one_l, atol=1e-10)
        for s in STABILIZERS
    )

    # Exhaustive correction of all 27 single-qubit Pauli errors.
    logical_zero = shor_encode()
    all_single_errors_pass = True
    for q in range(N_QUBITS):
        for op in ("X", "Y", "Z"):
            err = pauli_from_ops({q: op})
            corrupted = apply_pauli(logical_zero, err)
            syn = syndrome(err)
            recovered = correct_single_pauli_error(
                corrupted, [syn, syn, syn]
            ).state
            fidelity = abs(np.vdot(logical_zero, recovered)) ** 2
            all_single_errors_pass &= np.isclose(fidelity, 1.0, atol=1e-10)
    tests["all_27_single_pauli_errors_corrected"] = bool(all_single_errors_pass)

    # Three-round majority vote tolerates one faulty readout per stabilizer.
    test_syn = syndrome(pauli_from_ops({2: "Y"}))
    noisy_rounds = [
        tuple(1 - b for b in test_syn),
        test_syn,
        test_syn,
    ]
    tests["three_round_measurement_majority"] = (
        majority_vote_syndromes(noisy_rounds) == test_syn
    )

    # Logical operator action on the selected encoded basis.
    lx_zero = apply_pauli(zero_l, LOGICAL_X)
    lz_one = apply_pauli(one_l, LOGICAL_Z)
    tests["logical_operator_action"] = (
        np.isclose(abs(np.vdot(one_l, lx_zero)), 1.0, atol=1e-10)
        and np.isclose(abs(np.vdot(-one_l, lz_one)), 1.0, atol=1e-10)
    )

    # Ancilla phase reference: U = exp(i*phi) I gives the expected relative phase
    # only because it is controlled relative to the identity branch.
    phi = 0.43
    phase_result = ancilla_phase_reference(
        np.array([1.0, 0.0], dtype=complex),
        np.exp(1j * phi) * np.eye(2, dtype=complex),
    )
    tests["ancilla_phase_reference"] = bool(
        np.isclose(phase_result.relative_phase_rad, phi, atol=1e-10)
        and np.isclose(phase_result.visibility, 1.0, atol=1e-10)
    )

    # v4: numerical QFIM agrees with the closed-form PsiGate metric.
    def _psi(p):
        return psigate(float(p[0]), float(p[1])) @ np.array([1.0, 0.0], dtype=complex)

    tests["qfim_matches_closed_form"] = all(
        np.allclose(
            pure_state_qfim(_psi, [A0, T0]),
            psigate_qfim_closed_form(A0, T0),
            atol=1e-6,
        )
        for A0, T0 in [(0.7, 1.1), (2.0, 0.5), (1.3, 2.2)]
    )

    # v4: Berry phase of the canonical loop (Pancharatnam product),
    # against the converged simulation reference -- not the note's
    # grid-error-carrying +2.2234 rad.
    tests["berry_phase_pancharatnam"] = bool(
        abs(berry_phase_pancharatnam(400) - BERRY_PHASE_REFERENCE_RAD) < 1e-3
    )

    # v4: transversal CNOT logical truth table, including entanglement
    # generation from a logical superposition control.
    def _fid(a, b):
        return abs(np.vdot(a, b)) ** 2

    plus_l = shor_encode(1.0, 1.0)
    tests["transversal_cnot_truth_table"] = bool(
        np.isclose(
            _fid(transversal_shor_logical_cnot(np.kron(zero_l, zero_l)),
                 np.kron(zero_l, zero_l)),
            1.0, atol=1e-10,
        )
        and np.isclose(
            _fid(transversal_shor_logical_cnot(np.kron(one_l, zero_l)),
                 np.kron(one_l, one_l)),
            1.0, atol=1e-10,
        )
        and np.isclose(
            _fid(
                transversal_shor_logical_cnot(np.kron(plus_l, zero_l)),
                (np.kron(zero_l, zero_l) + np.kron(one_l, one_l)) / math.sqrt(2),
            ),
            1.0, atol=1e-10,
        )
    )

    # v4: end-to-end correction under faulty readout, exhaustively:
    # every single-qubit Pauli error x every syndrome bit x every
    # round, with that one bit flipped in that one round. Majority
    # voting must absorb the fault and the verified correction must
    # restore the logical state exactly (27 x 8 x 3 = 648 cases).
    exhaustive_ok = True
    for q in range(N_QUBITS):
        for op in ("X", "Y", "Z"):
            err = pauli_from_ops({q: op})
            corrupted = apply_pauli(logical_zero, err)
            true_syn = syndrome(err)
            for r_flip in range(3):
                for b_flip in range(8):
                    rounds = []
                    for r in range(3):
                        row = list(true_syn)
                        if r == r_flip:
                            row[b_flip] ^= 1
                        rounds.append(tuple(row))
                    res = correct_single_pauli_error(corrupted, rounds)
                    exhaustive_ok &= bool(
                        res.corrected
                        and np.isclose(
                            abs(np.vdot(logical_zero, res.state)) ** 2,
                            1.0, atol=1e-10,
                        )
                    )
    tests["noisy_rounds_end_to_end"] = bool(exhaustive_ok)

    # Multi-error extension, part 1 -- degeneracy inside one block:
    # every nonempty Z pattern inside a single block is stabilizer-
    # equivalent to the identity (even weight) or to a single Z in
    # that block (odd weight). 21 patterns checked against the span.
    degeneracy_ok = True
    for block in [(0, 1, 2), (3, 4, 5), (6, 7, 8)]:
        for mask in range(1, 1 << 3):
            pat = pauli_from_ops(
                {block[i]: "Z" for i in range(3) if (mask >> i) & 1}
            )
            rep = (
                pauli_from_ops({})
                if bin(mask).count("1") % 2 == 0
                else pauli_from_ops({block[0]: "Z"})
            )
            degeneracy_ok &= (
                pauli_difference(pat, rep) in STABILIZER_SPAN
            )
    tests["multi_error_degeneracy_within_block"] = bool(degeneracy_ok)

    # Multi-error extension, part 2 -- one X per block: all 27
    # weight-3 patterns (three simultaneous bit flips, favourably
    # distributed) decoded and corrected exactly on a general logical
    # state, beyond the distance-3 single-error guarantee.
    general_state = shor_encode(0.6, 0.8)
    spread_ok = True
    for q0 in (0, 1, 2):
        for q1 in (3, 4, 5):
            for q2 in (6, 7, 8):
                err = pauli_from_ops({q0: "X", q1: "X", q2: "X"})
                corrupted = apply_pauli(general_state, err)
                syn = syndrome(err)
                res = correct_favourable_pattern(corrupted, [syn, syn, syn])
                spread_ok &= bool(
                    res.corrected
                    and np.isclose(
                        abs(np.vdot(general_state, res.state)) ** 2,
                        1.0, atol=1e-10,
                    )
                )
    tests["multi_error_one_x_per_block"] = bool(spread_ok)

    # Multi-error extension, part 3 -- the family is uniquely
    # decodable: 1408 patterns, and the collision-checked builder
    # resolved them onto syndrome classes without raising (a complete
    # set of coset representatives: all 256 classes are covered).
    tests["favourable_family_decodable"] = bool(
        len(favourable_family_patterns()) == 1408
        and len(FAVOURABLE_FAMILY_LOOKUP) == 256
    )

    # Multi-error extension, part 4 -- the boundary, as a test: one Z
    # in an outer block (Z0) and one Z in each of the other two blocks
    # (Z3 Z6) produce the SAME syndrome, and they differ by Z0 Z3 Z6,
    # which is a logical operator, not a stabilizer. So Z-spread
    # patterns are NOT uniquely decodable -- the favourable region
    # provably ends where the single-block condition does.
    err_a = pauli_from_ops({0: "Z"})
    err_b = pauli_from_ops({3: "Z", 6: "Z"})
    tests["multi_error_boundary_is_logical"] = bool(
        syndrome(err_a) == syndrome(err_b)
        and pauli_difference(err_a, err_b) not in STABILIZER_SPAN
    )

    return tests


def run_tests() -> List[Dict[str, Any]]:
    """qce-improved package format: [{name, pass, detail}, ...]."""
    results = run_self_tests()
    out: List[Dict[str, Any]] = []
    for name, passed in results.items():
        detail = "quinovet v4 self-test"
        if name == "berry_phase_pancharatnam":
            detail = (
                f"gamma = {berry_phase_pancharatnam(400):+.4f} rad "
                f"(converged reference +{BERRY_PHASE_REFERENCE_RAD:.4f} rad)"
            )
        elif name == "qfim_matches_closed_form":
            detail = "numerical QFIM vs closed form at 3 points, atol 1e-6"
        elif name == "noisy_rounds_end_to_end":
            detail = "648 exhaustive single-fault readout cases, all restored"
        out.append({
            "name": f"quinovet_{name}",
            "pass": bool(passed),
            "detail": detail,
        })
    return out


if __name__ == "__main__":
    results = run_self_tests()
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    failures = [name for name, passed in results.items() if not passed]
    print(f"\n{len(results) - len(failures)}/{len(results)} self-tests passed.")
    if failures:
        raise SystemExit("Self-test failure(s): " + ", ".join(failures))
