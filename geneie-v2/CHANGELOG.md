# GENEIE v2 — change log vs the original (as pasted)

## v2.0.0 — 2026-11-07 (this build)

### Kept (from the original, now working)
- The vision: genomic research + drug discovery, AI + quantum, in one app.
- The Keras MLP shape 128 → Dropout 0.2 → 64 → BatchNorm → sigmoid, reimplemented
  as sklearn `MLPClassifier(hidden_layer_sizes=(128, 64))` with early stopping.
- PCA + K-means + t-SNE analysis idea, now actually running (t-SNE gated on n ≥ 31).
- The VQE idea, now real PennyLane VQE instead of a nonexistent PyQuil object.
- The GUI layout idea (open CSV, filter, table, analysis selector), now as Dash tabs.
- The 3D molecule viewer idea, now Plotly 3D instead of nonexistent VR engines.
- DNA sequence analysis idea, now `Bio.Seq` GC / k-mer stats.

### Fixed (were broken in the original)
- 3 syntax errors that stopped all three files compiling (unterminated string, line 291
  of geneie_main.py; invalid import line 10 of geneie_script.py; IndentationError line 174
  of geneie_gui.py).
- Missing imports (`requests`, `os`; `Sequential`/callbacks/`SeqIO` in the second copy).
- Undefined names (`backend`, `blockchain`, `encrypt`, `decrypt`, `database`, `analysis`,
  `secure_delete`, plus `encrypt_data` / `decrypt_data` / `perform_analysis`).
- `SeqIO.read()` called on a string; `X = drugs['Features']` treated as 2-D;
  t-SNE with default perplexity on unknown n; training inside a constructor;
  `geneie.run()` called on a QMainWindow that has no `run()` method.

### Dropped (vision only, no code/data/credentials in the pasted material)
- Federated learning, blockchain, Clara/Parabricks, DeepVariant, Pyteomics, Scanpy,
  DNABERT/ProtBERT/BioBERT, QSVM/VQC, GNNs, diffusion/QGAN drug generation,
  Kinesis, FHIR, Riva, PyVista/VR/AR. Listed as roadmap in README.md.
- TensorFlow, PyQt5, Cirq, PyQuil, pyvcf, rdkit, deepchem, pyvirtualreality,
  pyaugmentedreality — removed from requirements.
- The Cirq one-qubit coin flip and the hardcoded "cat" image label.
