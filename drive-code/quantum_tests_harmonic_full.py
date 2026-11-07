
import numpy as np
import qutip as qt
from qiskit.quantum_info import concurrence
from qiskit import QuantumCircuit
import pennylane as qml
import pennylane.numpy as pnp
import logging

logging.basicConfig(filename='quantum_sim.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_harmonic_oscillator_tests(system):
    results = []
    
    logging.info("Starting Decoherence Mitigation Test")
    initial_state_1 = np.array([1, 0] * 500, dtype=complex)
    try:
        states_standard = system.time_evolve(initial_state_1, 50, 0.1)
        states_mitigated = system.time_evolve(initial_state_1, 50, 0.1, feedback_loop=True, adaptive_tuning=True)
        amp_standard = np.mean(np.abs(states_standard[-1].full().flatten())**2)
        amp_mitigated = np.mean(np.abs(states_mitigated[-1].full().flatten())**2)
        results.append(f"Decoherence Test: Standard Amplitude = {amp_standard:.4f}, Mitigated Amplitude = {amp_mitigated:.4f}")
        logging.info(f"Decoherence Test completed: Standard={amp_standard:.4f}, Mitigated={amp_mitigated:.4f}")
    except Exception as e:
        results.append(f"Decoherence Test: Error - {str(e)}")
        logging.error(f"Decoherence Test failed: {str(e)}")

    logging.info("Starting Multi-Qubit Scalability Test")
    for n in [1, 2, 3]:
        try:
            system.num_qubits = n
            system.qc = QuantumCircuit(n + 1)
            system.dev = qml.device("default.qubit.tf", wires=n * 2)
            system.vqc_params = pnp.array([0.0] * (12 * n), requires_grad=True)
            initial_state_n = np.array([1, 0] * (500 * n), dtype=complex)
            system.time_evolve(initial_state_n, 10, 0.1)
            counts = system.run_qiskit_circuit(10, 0.1)
            concurrence_val, _ = system.analyze_qubit_state(0.0)
            results.append(f"Scalability Test ({n} qubits): Concurrence = {concurrence_val:.4f}, Counts Sample = {list(counts.items())[:3]}")
            logging.info(f"Scalability Test ({n} qubits): Concurrence={concurrence_val:.4f}")
        except Exception as e:
            results.append(f"Scalability Test ({n} qubits): Error - {str(e)}")
            logging.error(f"Scalability Test ({n} qubits) failed: {str(e)}")

    logging.info("Starting Adaptive Tuning and Feedback Test")
    try:
        system.num_qubits = 1
        system.qc = QuantumCircuit(2)
        system.dev = qml.device("default.qubit.tf", wires=2)
        system.vqc_params = pnp.array([0.0] * 12, requires_grad=True)
        initial_state_1 = np.array([1, 0] * 500, dtype=complex)
        states_adaptive = system.time_evolve(initial_state_1, 20, 0.1, feedback_loop=True, adaptive_tuning=True)
        conc_0, _ = system.analyze_qubit_state(0.0)
        conc_05, _ = system.analyze_qubit_state(0.5)
        conc_1, _ = system.analyze_qubit_state(1.0)
        results.append(f"Adaptive Tuning Test: Concurrence t=0: {conc_0:.4f}, t=0.5: {conc_05:.4f}, t=1.0: {conc_1:.4f}")
        logging.info(f"Adaptive Tuning Test completed: t=0={conc_0:.4f}, t=0.5={conc_05:.4f}, t=1.0={conc_1:.4f}")
    except Exception as e:
        results.append(f"Adaptive Tuning Test: Error - {str(e)}")
        logging.error(f"Adaptive Tuning Test failed: {str(e)}")

    return "\n".join(results)

if __name__ == "__main__":
    from harmonic_oscillator import HarmonicOscillator  # Adjust this import path
    system = HarmonicOscillator()
    print(run_harmonic_oscillator_tests(system))
