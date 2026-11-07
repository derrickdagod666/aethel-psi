"""Quantum execution layer for the Syntropic Swarm v2.

Changes vs the original QuantumHardwareLink (per Putt's rule, elaborated):
  1. SIMULATOR IS THE DEFAULT. The original only targeted real IBM hardware
     (least_busy(simulator=False)) and was dead on any machine without saved
     IBM credentials. v2 runs the same 3-qubit Aethel-Psi circuit on a local
     Qiskit Aer simulator by default: free, instant, no account, no queue.
  2. IBM RUNTIME IS OPT-IN AND GATED TWICE. It is used only when
     (a) QiskitRuntimeService() finds saved credentials on this machine AND
     (b) the env var AOR_ENABLE_IBM=1 is set. Without both, IBM mode reports
     itself unavailable and nothing is submitted. No account is ever created
     here, no token is ever asked for, and no job is ever submitted without
     the explicit flag. A real-hardware run also prints the backend name and
     the shot count so the user sees what is being spent.
  3. OFF mode exists: quantum execution disabled entirely (chat still works).
  4. The entropy is labelled honestly: it is the SHANNON entropy of the
     measured outcome distribution, not a Von Neumann entropy of a state
     (the original's comment said "Von Neumann / Shannon" — with only
     measurement counts available, only Shannon is computed).
  5. Mode is chosen at construction and reported in .describe(); a failed
     IBM probe degrades to SIMULATOR with a printed reason instead of
     leaving .backend = None and raising on every Enterprise request.

Modes (env AOR_QUANTUM_MODE, default "simulator"):
  simulator — AerSimulator, 1024 shots, local.
  ibm       — real hardware via QiskitRuntimeService, only with credentials
              AND AOR_ENABLE_IBM=1; otherwise falls back to simulator.
  off       — quantum execution disabled; execute() raises a plain error.
"""
import os

import numpy as np

SHOTS_DEFAULT = 1024


def shannon_entropy(counts):
    """Shannon entropy (bits) of a measurement-counts dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def build_aethel_psi_circuit(phase_parameter):
    """The original's 3-qubit circuit, unchanged: H, CX, CX, RZ(phase*pi)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(float(phase_parameter) * np.pi, 2)
    qc.measure_all()
    return qc


class QuantumLink:
    def __init__(self, mode=None):
        self.requested_mode = (mode or os.environ.get("AOR_QUANTUM_MODE")
                               or "simulator").lower()
        self.mode = "off"
        self.backend = None
        self.backend_name = None
        self.status = ""
        self._init_backend()

    def _init_backend(self):
        if self.requested_mode == "off":
            self.mode = "off"
            self.status = "Quantum execution OFF (AOR_QUANTUM_MODE=off)."
            print(self.status)
            return
        if self.requested_mode == "ibm":
            self._try_ibm()
            if self.mode == "ibm":
                return
            print(f"IBM mode unavailable ({self.status}) — falling back "
                  "to the local simulator.")
        # Default and fallback: local Aer simulator.
        try:
            from qiskit_aer import AerSimulator

            self.backend = AerSimulator()
            self.backend_name = "aer_simulator (local)"
            self.mode = "simulator"
            self.status = "Simulator ready (Aer, local, free)."
            print(self.status)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            self.mode = "off"
            self.status = f"Simulator unavailable ({exc})."
            print(self.status)

    def _try_ibm(self):
        """Use real hardware only with saved credentials AND explicit opt-in."""
        if os.environ.get("AOR_ENABLE_IBM") != "1":
            self.status = ("IBM Runtime not enabled: set AOR_ENABLE_IBM=1 to "
                           "opt in (saved IBM Quantum credentials also "
                           "required). No job will be submitted.")
            return
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            self.status = "qiskit-ibm-runtime is not installed."
            return
        try:
            service = QiskitRuntimeService()
            backend = service.least_busy(simulator=False, operational=True)
        except Exception as exc:  # noqa: BLE001 - credentials/connection
            self.status = f"IBM Quantum service unreachable: {exc}"
            return
        self.backend = backend
        self.backend_name = backend.name
        self.mode = "ibm"
        self.status = (f"IBM Quantum link established: {backend.name} "
                       "(real hardware — jobs consume quota/queue time).")
        print(self.status)

    def describe(self):
        return {"mode": self.mode, "backend": self.backend_name,
                "status": self.status}

    def execute_topological_collapse(self, phase_parameter,
                                     shots=SHOTS_DEFAULT):
        """Run the 3-qubit circuit; return (circuit, counts, shannon_bits).

        Raises ValueError in OFF mode (nothing to run on).
        """
        if self.mode == "off" or self.backend is None:
            raise ValueError("Quantum execution unavailable: " + self.status)
        qc = build_aethel_psi_circuit(phase_parameter)
        if self.mode == "ibm":
            from qiskit_ibm_runtime import SamplerV2 as Sampler

            print(f"Submitting real-hardware job to {self.backend_name} "
                  f"({shots} shots) — this consumes IBM Quantum quota.")
            sampler = Sampler(backend=self.backend)
            job = sampler.run([qc], shots=shots)
            result = job.result()[0].data.meas.get_counts()
            return qc, dict(result), shannon_entropy(result)
        # Simulator path (Aer).
        from qiskit_aer import AerSimulator

        sim = self.backend if isinstance(self.backend, AerSimulator) \
            else AerSimulator()
        job = sim.run(qc, shots=shots)
        result = job.result().get_counts(qc)
        counts = dict(result)
        return qc, counts, shannon_entropy(counts)
