# GENEIE v2 — change log vs the original (as pasted)

## v2.2.0 — 2026-11-16 — provenance record + research-only boundary

The two cheapest credibility wins from the GENEIE evidence report
(273-paper synthesis, filed with the company papers), built into the app:

- **Per-run provenance record (`geneie_core.build_provenance`).** Every
  PCA + K-means run — the initial demo view, an upload, or a Re-run — now
  produces a record: timestamp (UTC), app version, dataset identity (demo,
  or the upload's filename), input shape before cleaning, analysed shape
  after cleaning, the full `prepare_numeric_frame` cleaning report for
  uploads (columns dropped and why, cells imputed, duplicates), analysis
  parameters (PCA components, K-means k, seed, n_init, explained variance,
  inertia), the installed versions of numpy / pandas / scikit-learn / dash,
  and a SHA-256 of the analysed frame's CSV bytes so a run can be identified
  and repeated. In the UI it appears as a "Run provenance" panel under the
  graph (one-line summary + full JSON in a details fold) with
  download-as-JSON and download-as-Markdown buttons
  (`core.provenance_markdown`). Kept in session memory only — nothing is
  written to disk server-side, same as the upload flow. The upload callback
  also stores the cleaning report with the cleaned records, so a Re-run's
  record still shows how the data was cleaned. For the demo data the
  cleaning report is recorded as none (it is used as generated).
- **Research-only boundary in the UI.** A persistent banner sits above the
  tabs at all times (not a modal, not dismissible): research use only,
  outputs are hypotheses — not clinical diagnosis, treatment, or therapeutic
  recommendations — and the same notice appears as a footer and inside every
  provenance record (`core.RESEARCH_ONLY_NOTICE`), so it travels with any
  exported result. The app header and browser title now carry the version
  (v2.2.0, from `core.APP_VERSION`).
- **Honest limits unchanged:** uploads still get PCA + K-means only; the
  drug-response model does not retrain on uploads and no upload label column
  is used — the provenance record's analysis note says so explicitly.
- Tests: provenance checks added to `test_headless.py` (demo + upload
  records, hash reproducibility, parameters/versions, notice, Markdown
  rendering) and a new fast suite `test_fast.py` (core cleaning cases + the
  actual Dash callbacks + layout banner) — fast suite 16/16 pass, full
  headless suite ALL PASS.

## v2.1.0 — 2026-11-16 — upload fix

- **Uploaded CSVs are now analysed.** Before: the upload callback parsed the CSV,
  printed its shape, and admitted in its own status line that the table still
  showed demo data and re-running PCA on an upload was a TODO — the headline
  gap in the consulting offer. Now the upload drives the tab: table, PCA +
  K-means graph, and the Re-run button all operate on the uploaded file.
- **`geneie_core.prepare_numeric_frame()` added:** decides which columns are
  usable numeric features (≥50% parse as numbers), drops and names text and
  constant columns, drops all-empty rows, median-imputes missing cells and
  counts them, counts duplicate rows, and raises readable errors for too few
  rows/features. Systemic impact: every future upload-shaped input gets the
  same auditable cleaning instead of ad-hoc handling in the callback.
  One ordering bug was caught by test during the build (constant columns had
  to be dropped before empty-row removal, or a row empty in every varying
  feature survived on its constant value) — fixed and covered by tests.
- **Re-run button fixed as a side effect:** it previously re-ran on the demo
  frame only and never updated the graph; it now re-runs on whichever dataset
  is active (upload if one is loaded, else demo) and redraws the graph.
- **Limits and privacy stated in the UI:** 5 MB / 5,000 rows, in-memory only,
  nothing saved to disk. (The GENEIE_unified starter uploaded the same day
  writes uploads to `data/uploads/` on disk and only summarises them — its
  missing/duplicate summary counts were adopted into the status report, its
  storage choice and summary-only analysis were not.)
- Tests: upload cases added to `test_headless.py` (demo-shaped upload,
  constant/missing handling, rejection cases), plus callback-level tests —
  18/18 pass in the fast suite.

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
