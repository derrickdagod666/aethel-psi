"""GENEIE v2 — Dash + Plotly web app (replaces the PyQt5 desktop GUI).

Change vs the original, per Putt's rule: the original was a PyQt5 QMainWindow whose
buttons did nothing (all analysis methods were `pass`), whose VR/AR engines were
imports of packages that do not exist, and which trained a model inside the window
constructor. v2 is a browser app: every button runs real code (see geneie_core.py),
training happens on demand, and VR/AR are dropped and listed as roadmap only.
Run: python app.py  (then open http://127.0.0.1:8050)
"""
import base64
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table

import geneie_core as core

app = Dash(__name__, title="GENEIE v2")
server = app.server

# Pre-train the drug model once at startup on synthetic data (labelled as such in the UI).
_DRUG_X, _DRUG_Y = core.synthetic_drug_data()
_DRUG_MODEL = core.train_drug_model(_DRUG_X, _DRUG_Y)
_GEN_DF, _GEN_Y = core.synthetic_genomic_data()

TAB_STYLE = {"padding": "20px"}

def _pca_fig(df):
    res = core.pca_kmeans(df)
    pcs, labels = res["pcs"], res["labels"]
    fig = px.scatter(x=pcs[:, 0], y=pcs[:, 1], color=labels.astype(str),
                     labels={"x": "PC1", "y": "PC2", "color": "K-means cluster"},
                     title=f"PCA + K-means (2 clusters) — variance explained "
                           f"PC1 {res['explained_variance_ratio'][0]:.2f}, "
                           f"PC2 {res['explained_variance_ratio'][1]:.2f}")
    return fig, res

def _mol_fig(name):
    mol = core.MOLECULES[name]
    xs = [c[0] for c in mol["coords"]]; ys = [c[1] for c in mol["coords"]]; zs = [c[2] for c in mol["coords"]]
    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="markers+text",
                                       text=mol["symbols"], textposition="top center",
                                       marker=dict(size=12))])
    fig.update_layout(title=f"{name} — atom positions (angstrom, schematic, not a VR render)",
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return fig

genomic_tab = html.Div([
    html.H3("Genomic data — synthetic demo (NOT real patient data)"),
    html.P("Upload a CSV or use the built-in synthetic demo table (120 samples, 12 features, "
           "two planted clusters). PCA + K-means run for real; the table and filter work."),
    dcc.Upload(id="upload-csv", children=html.Button("Upload CSV"), multiple=False),
    html.Div(id="upload-status"),
    dash_table.DataTable(id="genomic-table", page_size=10,
                         data=_GEN_DF.head(50).to_dict("records"),
                         columns=[{"name": c, "id": c} for c in _GEN_DF.columns],
                         filter_action="native", sort_action="native"),
    dcc.Graph(id="pca-graph", figure=_pca_fig(_GEN_DF)[0]),
    html.Button("Re-run PCA + K-means", id="rerun-pca", n_clicks=0),
    html.Div(id="pca-status"),
], style=TAB_STYLE)

drug_tab = html.Div([
    html.H3("Drug response prediction — synthetic demo model"),
    html.P(f"sklearn MLP (128 -> 64, the original's Keras shape, in sklearn), trained on "
           f"synthetic data. Test accuracy: {_DRUG_MODEL['test_accuracy']:.3f} "
           f"(n_test={_DRUG_MODEL['n_test']}). This is NOT a real drug-efficacy model — "
           "no real compounds, no clinical claims."),
    html.P("Enter the 10 descriptor values (or leave the defaults) and predict:"),
    html.Div([dcc.Input(id=f"feat-{i}", type="number", value=0.0, step=0.1,
                        style={"width": "90px", "margin": "3px"})
              for i in range(10)]),
    html.Br(), html.Button("Predict response", id="predict-btn", n_clicks=0),
    html.Div(id="predict-out"),
], style=TAB_STYLE)

vqe_tab = html.Div([
    html.H3("VQE drug discovery — PennyLane, simulator only"),
    html.P("Real VQE (UCCSD + Adam) on H2 / LiH, compared with exact diagonalisation of "
           "the same Hamiltonian. Click to run — H2 takes ~10-30 s, LiH longer."),
    dcc.Dropdown(id="vqe-mol", options=[{"label": m, "value": m} for m in ("H2", "LiH")],
                 value="H2"),
    html.Button("Run VQE", id="vqe-btn", n_clicks=0),
    html.Button("Run H2 bond scan", id="scan-btn", n_clicks=0),
    html.Div(id="vqe-out"),
    dcc.Graph(id="vqe-graph"),
], style=TAB_STYLE)

dna_tab = html.Div([
    html.H3("DNA sequence"),
    html.P("Paste a DNA sequence (or use a demo fragment). GC content and top k-mers "
           "are computed with Bio.Seq — the original called SeqIO.read() on a string, "
           "which raises; v2 does not."),
    dcc.Dropdown(id="dna-demo", options=[{"label": k, "value": k} for k in core.DEMO_SEQUENCES],
                 value="BRCA1-fragment"),
    dcc.Textarea(id="dna-input", value=core.DEMO_SEQUENCES["BRCA1-fragment"],
                 style={"width": "100%", "height": 80}),
    html.Button("Analyse", id="dna-btn", n_clicks=0),
    html.Div(id="dna-out"),
], style=TAB_STYLE)

mol_tab = html.Div([
    html.H3("3D molecule viewer — Plotly 3D (not PyVista VR)"),
    dcc.Dropdown(id="mol-pick", options=[{"label": m, "value": m} for m in core.MOLECULES],
                 value="H2O"),
    dcc.Graph(id="mol-graph", figure=_mol_fig("H2O")),
], style=TAB_STYLE)

app.layout = html.Div([
    html.H1("GENEIE v2"),
    html.P("Genomic research + drug discovery demo. Synthetic demo data only. "
           "No real patient data, no clinical claims, simulator only. "
           "Federated learning, blockchain, Clara/Parabricks, DeepVariant, Riva, "
           "FHIR, Kinesis, VR/AR and QGAN drug generation are roadmap items, not features — "
           "see README.md."),
    dcc.Tabs([
        dcc.Tab(label="Genomic data", children=[genomic_tab]),
        dcc.Tab(label="Drug response", children=[drug_tab]),
        dcc.Tab(label="VQE", children=[vqe_tab]),
        dcc.Tab(label="DNA sequence", children=[dna_tab]),
        dcc.Tab(label="3D molecule", children=[mol_tab]),
    ]),
])

@app.callback(Output("pca-status", "children"), Input("rerun-pca", "n_clicks"))
def rerun_pca(n):
    if not n:
        return ""
    res = core.pca_kmeans(_GEN_DF)
    return (f"Re-ran PCA + K-means: PC1 {res['explained_variance_ratio'][0]:.2f}, "
            f"PC2 {res['explained_variance_ratio'][1]:.2f}, inertia {res['inertia']:.1f}.")

@app.callback(Output("predict-out", "children"), Input("predict-btn", "n_clicks"),
              [State(f"feat-{i}", "value") for i in range(10)])
def predict(n, *vals):
    if not n:
        return ""
    vals = [0.0 if v is None else v for v in vals]
    r = core.predict_response(_DRUG_MODEL, vals)
    return (f"Prediction: {'response' if r['prediction'] else 'no response'} "
            f"(P(response) = {r['p_response']:.3f}). Synthetic demo model — not a clinical result.")

@app.callback(Output("vqe-out", "children"), Output("vqe-graph", "figure"),
              Input("vqe-btn", "n_clicks"), Input("scan-btn", "n_clicks"),
              State("vqe-mol", "value"))
def run_vqe(n1, n2, mol):
    from dash import ctx
    if not ctx.triggered_id:
        return "", go.Figure()
    if ctx.triggered_id == "scan-btn":
        scan = core.h2_bond_scan(max_steps=40)
        fig = px.line(x=[s["bond_length_bohr"] for s in scan],
                      y=[s["vqe_energy"] for s in scan], markers=True,
                      labels={"x": "H-H distance (bohr)", "y": "Energy (Ha)"},
                      title="H2 VQE bond scan (vs exact: errors shown in text)")
        txt = " | ".join(f"r={s['bond_length_bohr']}: E={s['vqe_energy']:.4f} "
                         f"(exact {s['exact_energy']:.4f}, err {s['error']:.4f})" for s in scan)
        return txt, fig
    if mol == "H2":
        coords = [(0.0, 0.0, -0.7), (0.0, 0.0, 0.7)]
        res = core.vqe_energy(["H", "H"], coords, max_steps=60)
    else:
        res = core.lih_energy(max_steps=60)
    return (f"{mol}: VQE E = {res['vqe_energy']:.4f} Ha, exact = {res['exact_energy']:.4f} Ha, "
            f"|error| = {res['error']:.4f} Ha, qubits = {res['n_qubits']}, "
            f"electrons = {res['n_electrons']}."), go.Figure()

@app.callback(Output("dna-out", "children"), Input("dna-btn", "n_clicks"),
              State("dna-input", "value"))
def analyse_dna(n, seq):
    if not n or not seq:
        return ""
    s = core.dna_stats(seq)
    kmers = ", ".join(f"{k}:{v}" for k, v in s["top_kmers"][:6])
    return (f"Length {s['length']} bp, GC {s['gc_percent']}%, "
            f"A {s['a']} T {s['t']} G {s['g']} C {s['c']}. Top k-mers: {kmers}.")

@app.callback(Output("mol-graph", "figure"), Input("mol-pick", "value"))
def pick_mol(name):
    return _mol_fig(name)

@app.callback(Output("dna-input", "value"), Input("dna-demo", "value"))
def load_demo(name):
    return core.DEMO_SEQUENCES[name]

@app.callback(Output("upload-status", "children"), Input("upload-csv", "contents"),
              State("upload-csv", "filename"))
def upload(contents, filename):
    if not contents:
        return ""
    try:
        _, b64 = contents.split(",", 1)
        df = pd.read_csv(io.StringIO(base64.b64decode(b64).decode("utf-8")))
        return f"Uploaded {filename}: {len(df)} rows, {len(df.columns)} columns. (Table still shows demo data; re-running PCA on uploads is a TODO — see README.)"
    except Exception as e:  # noqa: BLE001 - surface the error to the user
        return f"Upload failed: {e}"

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
