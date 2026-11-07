"""Dash + Plotly + PennyLane VQE drug-discovery app.

Run:  pip install -r requirements.txt && python app.py
Then open http://127.0.0.1:8050

What it does: pick a small molecule, run a PennyLane VQE on the
analytic simulator, compare to exact diagonalisation, scan the
bond length, and view the molecule in 3D (Plotly 3D scatter of atom
positions — NOT 3Dmol.js; no 3Dmol.js pattern was found in Putt's
files, see vqe_core.py).

Honest limits: simulator only, STO-3G minimal basis, active spaces
stated in vqe_core.py. No real quantum hardware, no drug-efficacy
claims. Energies are electronic ground-state estimates.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

from vqe_core import MOLECULES, bond_scan, coordinates, run_vqe

ATOM_COLORS = {"H": "white", "Li": "purple", "O": "red", "N": "blue"}
ATOM_SIZES = {"H": 12, "Li": 22, "O": 26, "N": 24}

app = Dash(__name__, title="Aethel-Psi VQE — Drug Discovery")
server = app.server


def molecule_figure(name, bond):
    """Plotly 3D scatter of atom positions (labelled as such in the UI)."""
    coords = coordinates(name, bond)
    symbols = MOLECULES[name]["symbols"]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[c[0] for c in coords], y=[c[1] for c in coords], z=[c[2] for c in coords],
        mode="markers+text",
        marker=dict(size=[ATOM_SIZES[s] for s in symbols],
                    color=[ATOM_COLORS[s] for s in symbols],
                    line=dict(color="black", width=1)),
        text=symbols, textposition="top center",
        name=name,
    ))
    # Bonds as lines between atoms within 2.2 A of each other.
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            if d < 2.2:
                fig.add_trace(go.Scatter3d(
                    x=[coords[i][0], coords[j][0]],
                    y=[coords[i][1], coords[j][1]],
                    z=[coords[i][2], coords[j][2]],
                    mode="lines", line=dict(color="grey", width=4),
                    showlegend=False,
                ))
    fig.update_layout(
        title=f"{name} at bond length {bond:.2f} A (Plotly 3D atom positions)",
        scene=dict(xaxis_title="x (A)", yaxis_title="y (A)", zaxis_title="z (A)",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


app.layout = html.Div([
    html.H1("VQE Drug-Discovery App — Aethel-Psi"),
    html.P("PennyLane VQE on the analytic simulator (default.qubit, shots=None), "
           "STO-3G minimal basis. Ground-state electronic energies for small "
           "molecules. No real hardware, no drug-efficacy claims. "
           "Built from Putt's Keep note 'yes 🚨' spec (Dash + Plotly + PennyLane) "
           "and his Drive PennyLane code — see README.md for the reuse labels."),
    html.Div([
        html.Label("Molecule"),
        dcc.Dropdown(id="molecule", options=[
            {"label": f"{m} — {MOLECULES[m]['note']}", "value": m}
            for m in MOLECULES
        ], value="H2"),
        html.Label("Bond length (Angstrom)"),
        dcc.Slider(id="bond", min=0.4, max=3.0, step=0.02, value=0.74,
                   marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0", 3.0: "3.0"}),
        html.Label("Optimizer"),
        dcc.Dropdown(id="optimizer", options=[
            {"label": "Adam", "value": "Adam"},
            {"label": "GradientDescent", "value": "GradientDescent"},
        ], value="Adam"),
        html.Label("VQE steps (UCCSD; 8-qubit molecules take ~3 s/step)"),
        dcc.Slider(id="steps", min=20, max=100, step=20, value=60,
                   marks={20: "20", 60: "60", 100: "100"}),
        dcc.Checklist(id="psi-overlay", options=[
            {"label": " Ψ harmonic overlay — seed ansatz angles from Psi(r) = 0.1·cos(470·r) "
                      "(note-inspired initialisation only, default OFF)", "value": "on"}
        ], value=[]),
        html.Br(),
        html.Button("Run VQE at this bond length", id="run-btn", n_clicks=0),
        html.Button("Run bond-length scan (5 points)", id="scan-btn", n_clicks=0),
        html.Div(id="status"),
    ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),
    html.Div([
        dcc.Graph(id="mol-3d"),
        dcc.Graph(id="iter-plot"),
        dcc.Graph(id="scan-plot"),
        html.Div(id="result-table"),
    ], style={"width": "52%", "display": "inline-block", "verticalAlign": "top"}),
])


@app.callback(
    Output("mol-3d", "figure"),
    Input("molecule", "value"), Input("bond", "value"),
)
def update_mol(name, bond):
    return molecule_figure(name, bond)


@app.callback(
    Output("iter-plot", "figure"), Output("result-table", "children"),
    Output("status", "children"),
    Input("run-btn", "n_clicks"),
    State("molecule", "value"), State("bond", "value"),
    State("optimizer", "value"), State("steps", "value"),
    State("psi-overlay", "value"),
    prevent_initial_call=True,
)
def run_single(n_clicks, name, bond, optimizer, steps, psi_overlay):
    r = run_vqe(name, bond, steps=int(steps), layers=2, optimizer=optimizer,
                psi_overlay=("on" in (psi_overlay or [])), seed=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=r["iterations"], mode="lines", name="VQE energy"))
    fig.add_hline(y=r["exact_energy"], line_dash="dash",
                  annotation_text=f"exact {r['exact_energy']:.6f} Ha")
    fig.update_layout(title=f"VQE convergence — {name} at {bond:.2f} A",
                      xaxis_title="step", yaxis_title="energy (Ha)")
    table = html.Table([
        html.Tr([html.Th("Molecule"), html.Td(name)]),
        html.Tr([html.Th("Bond length"), html.Td(f"{bond:.2f} A")]),
        html.Tr([html.Th("Qubits"), html.Td(str(r["n_qubits"]))]),
        html.Tr([html.Th("VQE energy"), html.Td(f"{r['vqe_energy']:.6f} Ha")]),
        html.Tr([html.Th("Exact energy"), html.Td(f"{r['exact_energy']:.6f} Ha")]),
        html.Tr([html.Th("Error"), html.Td(f"{r['error']:+.6f} Ha")]),
        html.Tr([html.Th("Optimizer"), html.Td(r["optimizer"])]),
        html.Tr([html.Th("Psi overlay"), html.Td(str(r["psi_overlay"]))]),
    ])
    status = (f"Done: {name} VQE {r['vqe_energy']:.6f} Ha vs exact "
              f"{r['exact_energy']:.6f} Ha (err {r['error']:+.6f} Ha, "
              f"{r['n_qubits']} qubits, {steps} steps).")
    return fig, table, status


@app.callback(
    Output("scan-plot", "figure"),
    Input("scan-btn", "n_clicks"),
    State("molecule", "value"), State("optimizer", "value"),
    State("psi-overlay", "value"),
    prevent_initial_call=True,
)
def run_scan(n_clicks, name, optimizer, psi_overlay):
    lo, hi = MOLECULES[name]["scan"]
    bonds = list(np.linspace(lo, hi, 5))
    rows = bond_scan(name, bonds, steps=60, layers=2, optimizer=optimizer,
                     psi_overlay=("on" in (psi_overlay or [])), seed=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["bond"] for r in rows],
                             y=[r["vqe_energy"] for r in rows],
                             mode="lines+markers", name="VQE"))
    fig.add_trace(go.Scatter(x=[r["bond"] for r in rows],
                             y=[r["exact_energy"] for r in rows],
                             mode="lines+markers", name="exact"))
    if "on" in (psi_overlay or []):
        # Note-inspired visualisation ONLY: a cos curve drawn alongside
        # the scan. It is not physics and does not change any energy.
        xs = np.linspace(lo, hi, 50)
        fig.add_trace(go.Scatter(x=xs, y=0.1 * np.cos(470 * xs),
                                 mode="lines", name="Psi overlay (illustrative)",
                                 line=dict(dash="dot")))
    fig.update_layout(title=f"Bond-length scan — {name} (STO-3G)",
                      xaxis_title="bond length (A)", yaxis_title="energy (Ha)")
    return fig


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
