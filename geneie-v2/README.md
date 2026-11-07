# GENEIE v2

A working, runnable rebuild of Putt's GENEIE — genomic research + drug discovery demo —
built from his own pasted material. The original is preserved verbatim in
`../GENEIE_original.md` and `../original/`; this is the version that actually runs.

## What it does (all verified by `test_headless.py`)

- **Genomic data tab** — upload a CSV or use built-in synthetic demo data (120 samples,
  12 features, two planted clusters). Table view with filter/sort, PCA + K-means with
  real plots. t-SNE only runs when n ≥ 31 (the original crashed below that).
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
python test_headless.py   # headless test, no server needed (takes ~5 min, mostly VQE)
python app.py             # then open http://127.0.0.1:8050 (first start takes ~4 min)
```

## Honest limits

- **Synthetic demo data only.** No real patient data, no real compounds, no clinical
  claims. The drug model cannot predict real drug response; the genomic clusters are
  planted in the demo data on purpose.
- **Simulator only.** VQE runs on PennyLane's `default.qubit` simulator. No real
  quantum hardware, no QPU access.
- **Small molecules only.** VQE is H2 / LiH; H2O / NH3 appear only in the 3D viewer.
  UCCSD on larger molecules gets slow fast on a simulator.
- Uploaded CSVs are shown as uploaded; re-running PCA on an upload is a TODO
  (the table currently keeps showing demo data after an upload — the status line says so).

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
