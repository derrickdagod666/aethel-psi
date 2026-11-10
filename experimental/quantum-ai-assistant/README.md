# experimental/quantum-ai-assistant — archival code, NOT validated results

**Status: ARCHIVAL / EXPERIMENTAL.** These files were salvaged from the
author's Google Drive on 2026-11-09 during the Drive science audit
(`DRIVE_SCIENCE_AUDIT.md`). They are kept **exactly as found** — not
cleaned, not repaired, and not part of the validated aethel-psi results
(Gate 3 pipeline, Qiskit stabilizer, validation suites). Nothing in this
folder has been run end-to-end by the repo maintainers, and no claim in
the papers rests on this code.

## Provenance

All files were downloaded read-only from Derrick Irons' Google Drive
("Google AI Studio" and "Colab Notebooks" folders), where they formed
the 2025 "Quantum AI Assistant" code family — the ancestor of the
workspace's Unified QCE / cognitive-companion apps. The Drive originals
were never the source of the versions in this repo, which were rebuilt
separately. Before copying, every file was scanned for hardcoded API
keys, tokens, passwords, and credential patterns — **none were found**.

Drive held duplicates, collapsed here to one copy each (byte-identical,
md5-verified): `code_converted.txt` is an exact duplicate of
`code (3).py`, and Drive held two identical copies of `code (2).py`.

## Files

| Repo file | Drive original | Drive date | Size | What it is | Syntax check (py_compile, 2026-11-09) |
|---|---|---|---|---|---|
| `quantum_cognitive_assistant.py` | `code.py` | 2025-04-16 | 32,303 B, 642 lines | "Quantum-Cognitive AI Assistant: Unified Implementation" — PennyLane + PyTorch hybrid quantum neural network (`QuantumProcessor`, `QuantumCognitiveNN`, `HybridTrainer`, `QuantumMonitor`), FAISS, Hugging Face transformers (gpt2 insights). | **PASS** |
| `oscillator_simulator_colab.py` | `code (3).py` (≡ `code_converted.txt`) | 2025-03-02 | 30,182 B, 648 lines | Colab script: QuTip/Qiskit harmonic-oscillator simulation — `QuantumSystem` ABC, `psi_waveform(x, A, omega, phi)`, `HarmonicOscillator`, Gradio UI, Gemma LLM hooks, TTS. The fullest single-file version of the simulator. | **FAIL** — begins with Colab `!pip install …` shell-escape lines, which are not valid Python outside a notebook. Archival as-is; expected. |
| `quantum_assistant_project_dump.py` | `code (2).py` (×2 in Drive) | 2025-07-10 | 26,064 B, 577 lines | A multi-file project concatenated into one dump (`QuantumSystem`, `HarmonicOscillator`, `CoupledOscillators`, `HydrogenAtom`, `QuantumAssistant`). A half-merged project: parts are commented-out skeleton. **Origin of the A=54 / λ=945 parameter set — see the conflict note below.** | **FAIL** — `IndentationError` at line 333 (unindent mismatch), consistent with its half-merged state. Archival as-is. |
| `quantum_tests_harmonic_full.py` | `quantum_tests_harmonic_full.py` | 2025-07-10 | 3,480 B, 68 lines | Test-harness fragment (QuTip + Qiskit concurrence + PennyLane) calling `system.time_evolve(...)` on classes defined in the files above. **Not standalone** — a tests file separated from its implementation. | **PASS** (syntax only; it cannot run without the classes above) |

Heavy dependency stack across the family: PennyLane, PyTorch, QuTip,
Qiskit, transformers, FAISS, Gradio, gTTS. No dependency list was
provided in Drive and none has been reconstructed yet.

Note: `quantum_cognitive_assistant.py` and
`quantum_tests_harmonic_full.py` also exist elsewhere in this repo under
`drive-code/` (archived 2026-11-07 under different names). This folder
is the complete family in one place, including the two files
`drive-code/` does not have.

## ⚠ Parameter conflict — flagged, not resolved

`quantum_assistant_project_dump.py` carries the constants **A = 54** and
**ω = 2π/945** (commented constant block, lines 11–12, and live default
arguments `A: float = 54, omega: float = 2*np.pi/945` at line 76). This
is the origin of the A=54 / λ=945 parameter set.

That set **conflicts** with the 2025 "Quantum and Geometric Harmonic
Resonance" paper's parameter set (**A ≈ 0.1**, coupling **ε ≈ 10⁻²⁹**,
derived there from a Calabi–Yau Euler characteristic). Neither set is
derived from the other, and the published Third Revised Edition does
not use the A=54 / λ=945 values. The code's constants are preserved
here for provenance only — **do not merge them into the paper's
parameter set, and do not cite them as the theory's parameters.**

## What would be needed to use this code

A real debugging pass (notebook escapes, the dump's indentation break),
a reconstructed dependency list, and re-derived — not inherited —
parameters. Until then this folder is a historical record of where the
project's code began, nothing more.
