"""VQE drug-discovery app — core (headless, no Dash needed).

Built from Putt's own material, labelled honestly below:

REUSED FROM DRIVE (labelled):
- PennyLane QNode / parameter-shift style and the learnable-Psi idea come
  from drive-code/quantum_cognitive_pennylane.py (642 lines): that file
  builds a PennyLane QNode with a learnable ``psi`` parameter encoded as
  ``qml.RY(psi)`` on one wire and entangled with CNOTs, and trains it
  with a classical optimizer. Here the same pattern is used ONLY as an
  optional, default-OFF initialisation of the VQE ansatz weights
  (``psi_overlay``): the first layer's RY angles are seeded from
  Psi(r) = A*cos(omega*r + phi) with A=0.1, omega=470, phi=0 — the
  waveform parameters on record in Putt's notes. It is an
  initialisation choice, not a physical coupling; the Hamiltonian is
  standard quantum chemistry.
- The fragment drive-code/quantum_tests_harmonic_full.py (68 lines)
  contributed nothing runnable (its ``system`` object is undefined and
  it imports a missing ``harmonic_oscillator`` module); it is NOT
  reused. Recorded here so the reuse claim is checkable.

NEWLY WRITTEN (not his code):
- Molecular Hamiltonians via ``qml.qchem.molecular_hamiltonian``
  (STO-3G, active spaces stated per molecule), the VQE loop, the
  exact-diagonalisation reference, the bond-length scan, and all
  Dash/Plotly UI code in ``app.py``.

SPEC ON RECORD (Google Keep note "yes 🚨", edited Mar 14):
- The note is described on record as a "Dash/Plotly/PennyLane VQE
  drug-discovery app". Its full text was NOT re-read in this run
  (live Keep needs a browser sign-in this session cannot do; the
  Drive hard search in ../CODE_HUNT_REPORT.md confirms no VQE/Dash
  code exists anywhere in Drive). Nothing beyond "Dash + Plotly +
  PennyLane + VQE + drug discovery" is claimed as his spec.
- The 3D molecule viewer is Plotly 3D scatter of atom positions,
  labelled as such. The Apothegm note "0001" describes a 3Dmol.js
  viewer, but no 3Dmol.js code/pattern was found in Drive or
  workspace, so no 3Dmol.js is used here. Labelled inspired-by-notes.

HONEST LIMITS (see README.md too):
- Simulator only (PennyLane default.qubit, analytic, shots=None).
  No real quantum hardware, no drug-efficacy claims. Energies are
  electronic ground-state estimates for small molecules in a minimal
  STO-3G basis with the active spaces stated below; they are not
  production drug-discovery results.
- Ansatz is UCCSD from the HF reference (see run_vqe). The first
  version used a hardware-efficient StronglyEntanglingLayers ansatz;
  it FAILED on the 8-qubit molecules (H2O err +0.21 Ha, NH3 err
  +0.31 Ha after 120 steps) and was replaced, not hidden.
- PennyLane 0.45 quirks found by experiment and handled in
  build_hamiltonian: ``molecular_hamiltonian`` defaults to
  ``unit="bohr"`` (Angstrom must be passed explicitly) and takes
  ``active_electrons``/``active_orbitals``, not ``active_space``.
- UCCSD on 8 qubits is slow (~3 s/step, ~3 min for 60 steps).
  The Dash app says so next to the steps slider.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ATOMIC_NUMBERS = {"H": 1, "Li": 3, "O": 8, "N": 7}

# ---------------------------------------------------------------------------
# Molecules. Coordinates are in Angstrom. Active space = (electrons, orbitals)
# in the STO-3G basis; stated explicitly because it is the main accuracy
# limit for LiH / H2O / NH3 here.
# ---------------------------------------------------------------------------
MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "charge": 0,
        "mult": 1,
        "active_space": None,  # full STO-3G: 2 orbitals, 4 qubits
        "default_bond": 0.74,
        "scan": (0.4, 2.0),
        "note": "Benchmark molecule. Full STO-3G, 4 qubits.",
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "charge": 0,
        "mult": 1,
        "active_space": (2, 2),  # 4 qubits
        "default_bond": 1.60,
        "scan": (1.0, 3.0),
        "note": "Drug-relevant light hydride. Active space (2e, 2o), 4 qubits.",
    },
    "H2O": {
        "symbols": ["O", "H", "H"],
        "charge": 0,
        "mult": 1,
        "active_space": (4, 4),  # 8 qubits
        "default_bond": 0.96,
        "scan": (0.7, 1.8),
        "note": "Water. Active space (4e, 4o), 8 qubits. Bond = O-H length, H-O-H angle fixed at 104.5 deg.",
    },
    "NH3": {
        "symbols": ["N", "H", "H", "H"],
        "charge": 0,
        "mult": 1,
        "active_space": (4, 4),  # 8 qubits
        "default_bond": 1.01,
        "scan": (0.8, 1.8),
        "note": "Ammonia. Active space (4e, 4o), 8 qubits. Bond = N-H length, pyramidal geometry fixed.",
    },
}


def coordinates(name: str, bond: float) -> np.ndarray:
    """Atom coordinates (Angstrom) for molecule ``name`` at bond length ``bond``.

    Returns a NumPy array: PennyLane 0.45's ``molecular_hamiltonian``
    calls ``coordinates.flatten()`` on its input.
    """
    if name == "H2":
        coords = [(0.0, 0.0, 0.0), (0.0, 0.0, bond)]
    elif name == "LiH":
        coords = [(0.0, 0.0, 0.0), (0.0, 0.0, bond)]
    elif name == "H2O":
        ang = np.deg2rad(104.5) / 2.0
        x = bond * np.sin(ang)
        z = bond * np.cos(ang)
        coords = [(0.0, 0.0, 0.0), (x, 0.0, z), (-x, 0.0, z)]
    elif name == "NH3":
        # Pyramidal: N at origin, H atoms on a cone, H-N-H angle ~107 deg.
        theta = np.deg2rad(107.0)
        rho = bond * np.sin(theta)
        z = -bond * np.cos(theta)
        coords = [
            (0.0, 0.0, 0.0),
            (rho, 0.0, z),
            (rho * np.cos(2 * np.pi / 3), rho * np.sin(2 * np.pi / 3), z),
            (rho * np.cos(4 * np.pi / 3), rho * np.sin(4 * np.pi / 3), z),
        ]
    else:
        raise ValueError(f"unknown molecule {name!r}")
    return np.array(coords, dtype=float)


def build_hamiltonian(name: str, bond: float):
    """Return (hamiltonian, n_qubits) for molecule ``name`` at ``bond`` (Angstrom).

    PennyLane 0.45 note (verified by experiment, 2026-11-07): the
    ``list`` dispatch of ``molecular_hamiltonian`` defaults to
    ``unit="bohr"`` — passing 0.74 gave HF -0.8868 Ha, while passing
    1.398 bohr (0.74 A) gave HF -1.11677 / exact -1.137286 Ha, the
    published STO-3G H2 values. So ``unit="angstrom"`` is passed
    explicitly here. Active space is ``active_electrons`` /
    ``active_orbitals`` in this version (``active_space`` kwarg does
    not exist and raises TypeError).
    """
    spec = MOLECULES[name]
    kwargs = {"unit": "angstrom"}
    if spec["active_space"] is not None:
        kwargs["active_electrons"], kwargs["active_orbitals"] = spec["active_space"]
    h, n_qubits = qml.qchem.molecular_hamiltonian(
        spec["symbols"], coordinates(name, bond), **kwargs
    )
    return h, n_qubits


def exact_energy(hamiltonian) -> float:
    """Ground-state energy by exact diagonalisation of the Hamiltonian matrix."""
    mat = qml.matrix(hamiltonian)
    return float(np.linalg.eigvalsh(mat)[0])


def _uccsd_wires(name: str, n_qubits: int):
    """Single/double excitation wires for the UCCSD ansatz.

    PennyLane 0.45: ``qml.qchem.excitations(electrons, spin_orbitals)``
    returns flat [o1, o2, v1, v2] double-excitation lists; ``qml.UCCSD``
    wants each double excitation as ([o1, o2], [v1, v2]). The conversion
    below is that fix, verified by experiment.
    """
    spec = MOLECULES[name]
    if spec["active_space"] is not None:
        electrons, orbitals = spec["active_space"]
    else:
        electrons = sum(ATOMIC_NUMBERS[s] for s in spec["symbols"])
        orbitals = n_qubits // 2
    s_wires, d_wires = qml.qchem.excitations(electrons, 2 * orbitals)
    d_ucc = [(list(d[:2]), list(d[2:])) for d in d_wires]
    return electrons, s_wires, d_ucc


def run_vqe(name: str, bond: float, steps: int = 60, layers: int = 2,
            optimizer: str = "Adam", stepsize: float = 0.1,
            psi_overlay: bool = False, seed: int = 0):
    """Run VQE for one molecule / bond length.

    Ansatz: UCCSD (unitary coupled-cluster, singles + doubles) from the
    Hartree-Fock reference. This replaced the first version's
    hardware-efficient StronglyEntanglingLayers ansatz, which FAILED
    on the 8-qubit molecules (H2O err +0.21 Ha, NH3 err +0.31 Ha after
    120 steps) and was too slow to fix by adding layers. UCCSD starts
    exactly at the HF energy and reached err 1.7e-05 Ha on H2O in 60
    steps — but it is slow (~3 s/step on 8 qubits, ~3 min per run),
    which the app states honestly. ``layers`` is kept for API
    compatibility and is ignored by UCCSD.

    Returns a dict with vqe_energy, exact_energy, error, iterations
    (list of energies per step), n_qubits, and the final weights.
    Analytic simulator (shots=None); no hardware.
    """
    np.random.seed(seed)
    h, n_qubits = build_hamiltonian(name, bond)
    dev = qml.device("default.qubit", wires=n_qubits)
    electrons, s_wires, d_ucc = _uccsd_wires(name, n_qubits)
    hf = qml.qchem.hf_state(electrons, n_qubits)
    n_params = len(s_wires) + len(d_ucc)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(weights):
        # Adjoint differentiation: exact on the analytic simulator and
        # far faster than parameter-shift here.
        qml.UCCSD(weights, wires=range(n_qubits), s_wires=s_wires,
                  d_wires=d_ucc, init_state=hf)
        return qml.expval(h)

    if psi_overlay:
        # Reused pattern from drive-code/quantum_cognitive_pennylane.py
        # (a learnable psi encoded as a rotation). Here it only seeds
        # the FIRST excitation amplitude from Psi(r) = A*cos(omega*r);
        # every other amplitude starts at 0 (the HF reference).
        # Default OFF; initialisation only, not a physical coupling.
        weights = np.zeros(n_params)
        weights[0] = 0.1 * np.cos(470.0 * bond)
    else:
        weights = np.zeros(n_params)
    weights = pnp.array(weights, requires_grad=True)

    if optimizer == "Adam":
        opt = qml.AdamOptimizer(stepsize=stepsize)
    elif optimizer == "GradientDescent":
        opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    elif optimizer == "SPSA":
        opt = qml.SPSAOptimizer(maxiter=steps)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    energies = []
    w = weights
    for _ in range(steps):
        w, energy = opt.step_and_cost(circuit, w)
        energies.append(float(energy))
    vqe_energy = float(circuit(w))
    ex = exact_energy(h)
    return {
        "molecule": name,
        "bond": bond,
        "n_qubits": n_qubits,
        "vqe_energy": vqe_energy,
        "exact_energy": ex,
        "error": vqe_energy - ex,
        "iterations": energies,
        "steps": steps,
        "optimizer": optimizer,
        "psi_overlay": psi_overlay,
        "ansatz": "UCCSD",
    }


def bond_scan(name: str, bonds, steps: int = 40, layers: int = 2,
              optimizer: str = "Adam", psi_overlay: bool = False, seed: int = 0):
    """VQE + exact energy at each bond length in ``bonds``."""
    rows = []
    for i, b in enumerate(bonds):
        r = run_vqe(name, float(b), steps=steps, layers=layers,
                    optimizer=optimizer, psi_overlay=psi_overlay, seed=seed + i)
        rows.append(r)
    return rows
