# drive-code — earlier PennyLane / quantum-cognitive code (as-is)

Copies of two Python files from the author's Google Drive,
downloaded read-only 2026-11-07. **Kept exactly as found — not
cleaned, not run, not production code.**

## Files

- `quantum_cognitive_pennylane.py` — Drive `code.py` (32,303 bytes,
  642 lines, 2025-04-16). "Quantum-Cognitive AI Assistant": a
  PennyLane QNode feature map (4 qubits, 2 layers) + a PyTorch
  neural error-mitigation model + FAISS memory + a Hugging Face
  transformer for "insights". Imports: numpy, torch, PennyLane,
  faiss, pyvista, transformers, matplotlib, loguru.
- `quantum_tests_harmonic_full.py` — Drive file (3,480 bytes,
  68 lines, 2025-07-10). Harmonic-oscillator / decoherence /
  multi-qubit tests using qutip, Qiskit, PennyLane.

## Status: NOT verified, NOT runnable as-is

- Neither file was run locally. No test output exists for them.
- `quantum_tests_harmonic_full.py` is a **fragment**: it calls
  `run_harmonic_oscillator_tests(system)` but the `system` class
  is defined nowhere in the file, and it uses the legacy
  PennyLane device `"default.qubit.tf"`. It needs reconstruction
  before it runs.
- `quantum_cognitive_pennylane.py` has heavy dependencies
  (torch + transformers + pyvista) and generates synthetic data
  only — no real dataset, no Dash/Plotly interface.
- **There is no Dash/Plotly VQE drug-discovery app in these
  files**, despite that description in the author's notes. The
  VQE app source was not found in Drive or the local workspace.
  Do not claim this folder contains it.

## If you want to run them

```bash
pip install torch pennylane faiss-cpu pyvista transformers loguru qutip qiskit
python quantum_cognitive_pennylane.py   # expect heavy downloads (transformer model)
```

No secrets were found in these files by grep (no API keys /
tokens / passwords). If you extend them, keep credentials in
environment variables — never in the code.
