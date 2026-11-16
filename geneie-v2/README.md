# GENEIE v2 (v2.2.0)

A working, runnable rebuild of Putt's GENEIE — genomic research + drug discovery demo —
built from his own pasted material. The original is preserved verbatim in
`../GENEIE_original.md` and `../original/`; this is the version that actually runs.

## What it does (all verified by `test_headless.py`)

- **Genomic data tab** — upload a CSV or use built-in synthetic demo data (120 samples,
  12 features, two planted clusters). Table view with filter/sort, PCA + K-means with
  real plots. t-SNE only runs when n ≥ 31 (the original crashed below that).
  **Uploads are analysed, not just acknowledged (fixed 2026-11-16):** on upload the
  table and the PCA + K-means graph switch to your file, and "Re-run PCA + K-means"
  re-runs on the upload. Cleaning rules, all reported in the status line: numeric
  feature columns only (a column counts when at least half its values parse as
  numbers; text/ID columns are dropped and named), at least 4 rows and 2 varying
  numeric features, missing cells median-imputed and counted, constant columns
  dropped and named, duplicate rows counted. Limits: 5 MB, 5,000 rows. Uploads are
  processed in memory only — never written to disk or sent anywhere.
- **Run provenance (v2.2.0)** — every analysis run (demo view, upload, or Re-run)
  records a provenance record under the graph: timestamp, app version, dataset
  identity, shapes before/after cleaning, the full cleaning report, analysis
  parameters, library versions, and a SHA-256 of the analysed data so a run can
  be identified and repeated. Downloadable as JSON or Markdown; held in session
  memory only.
- **Research-only banner (v2.2.0)** — a persistent banner (and footer) states
  that GENEIE is a research tool: outputs are hypotheses for research use, not
  clinical diagnosis, treatment, or therapeutic recommendations. The same
  notice is embedded in every provenance record.
- **Drug response tab** — sklearn MLP (128 → 64, the original's Keras shape, in sklearn),
  trained on synthetic data, train/test split, test accuracy reported in the UI.
  **Not a real drug-efficacy model.**
- **VQE tab** — real PennyLane VQE (UCCSD + Adam) on H2 / LiH, with an H2 bond-length
  scan, compared against exact diagonalisation of the same Hamiltonian.
- **DNA sequence tab** — Bio.Seq GC content and top k-mers on demo fragments
  (BRCA1 / TP53 / EGFR) or pasted sequence.
- **3D molecule tab** — Plotly 3D atom positions for H2, LiH, H2O, NH3
  (schematic geometry, labelled as such — not PyVista VR).

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python test_fast.py       # fast suite: core cleaning + Dash callbacks + layout (seconds)
python test_headless.py   # full headless test, no server needed (takes ~5 min, mostly VQE)
python app.py             # then open http://127.0.0.1:8050 (first start takes ~4 min)
```

## Honest limits

- **Synthetic demo data only.** No real patient data, no real compounds, no clinical
  claims. The drug model cannot predict real drug response; the genomic clusters are
  planted in the demo data on purpose.
- **Research use only.** GENEIE is a research tool; every output is a hypothesis
  for research use — not clinical diagnosis, treatment, or a therapeutic
  recommendation. The app says so in a persistent banner, and the notice is
  embedded in every run's provenance record.
- **Simulator only.** VQE runs on PennyLane's `default.qubit` simulator. No real
  quantum hardware, no QPU access.
- **Small molecules only.** VQE is H2 / LiH; H2O / NH3 appear only in the 3D viewer.
  UCCSD on larger molecules gets slow fast on a simulator.
- Uploaded CSVs get PCA + K-means only — the drug-response model is still trained
  on synthetic data and does NOT retrain on uploads, and no label/target column in
  an upload is used. An upload is a clustering view of your numeric features,
  not a validated analysis of your study.

## What was dropped, and why (roadmap, not features)

From the original vision doc — none of these has working code, data, or credentials in
the pasted material, so v2 does not pretend to do them:

- Federated learning, blockchain data sharing / provenance / tokenization
- NVIDIA Clara Discovery / Parabricks, DeepVariant, Pyteomics, Scanpy
- DNABERT / ProtBERT / BioBERT, QSVM / VQC, GNN drug-target prediction
- Diffusion / Stable Diffusion / QGAN drug generation
- Amazon Kinesis streaming, FHIR EHR integration, NVIDIA Riva ASR/TTS/NMT
- PyVista / VR / AR 3D visualization (replaced by Plotly 3D)
- Multi-omics integration (no multi-omics data was supplied)

What each would need next: see `../GENEIE_v2_SUMMARY.md`.

## What changed vs the original (Putt's rule: elaborate on every change)

- **PyQt5 desktop GUI → Dash web app.** The original GUI's buttons did nothing
  (`pass` stubs); every v2 button runs real code, and a browser app can be demoed.
- **Keras/TensorFlow → sklearn MLP.** Same 128 → 64 shape, a fraction of the install
  weight, and it trains on demand instead of inside a window constructor.
- **PyQuil `QuantumAlgorithm` (does not exist) → PennyLane VQE** (UCCSD + Adam, exact
  diagonalisation comparison). The one-qubit Cirq coin flip is dropped.
- **`SeqIO.read(string)` (raises) → `Bio.Seq` string handling** for GC / k-mers.
- **`X = drugs['Features']` (1-D Series) → a real numeric feature matrix** built by
  the synthetic generator, with a named descriptor list.
- **Blockchain / encrypt / database / analysis placeholders → deleted.** They were
  undefined names; v2 has no fake security theatre.
- **Hardcoded `genome.csv` / `drugs.csv` / `image.png` → synthetic demo data** generated
  in code, clearly labelled, plus a real CSV upload path.
- **100-epoch fit in `__init__` → train on demand**, with test accuracy reported.
- **Duplicate, disagreeing code copies → one codebase** (`geneie_core.py` + `app.py`),
  with the originals preserved untouched in `../original/`.
