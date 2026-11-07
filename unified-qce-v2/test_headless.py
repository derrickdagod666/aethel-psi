#!/usr/bin/env python3
"""Headless test for the UNIFIED QCE v2 (companion + swarm in one app).

Runs, in order:
  1. Index build (build_index.py --force) — chunk count, dim, time.
     Corpus: Putt's own papers (Third Revised Edition + UAF paper).
  2. Retrieval — gamma limit on GW250114; top L2, chunks, sources.
  3. Simulator quantum job — 3-qubit circuit on Aer, 1024 shots:
     counts, Shannon entropy in [0, 3] bits, GHZ outcomes dominate.
  4. Biometrics — 30 simulated RR intervals: RMSSD plausible, bio
     score in [0.1, 1.0], temperature = 1 - score.
  5. Ledger — commit/query/persistence round-trip on temp files;
     disabled ledger refuses; reload confirms the insight survives.
  6. Demo response — unified engine in demo mode (no LM Studio):
     cites a source, reports measured telemetry.
  7. Tier-3 quantum intercept — "entangle" runs the circuit, labelled
     SIMULATOR, never "IBM QUANTUM EXECUTION".
  8. VQE compiler — REAL compilation (the headline change of this
     build): cos(Psi) -> 0.5403*I, Psi**2 -> I, sin(Psi) -> 0.8415*Z,
     0.5*Psi+1 -> I + 0.5*Z, Pauli syntax "ZZ + XX" -> 2-qubit op.
     Each compiled Hamiltonian is run through a real VQE and the ground
     state is checked against the analytic value. Named presets still
     work and are labelled as presets. A bad expression fails cleanly.
  9. Tier gating — Tier 1 does NOT get the VQE intercept (quantum_access
     False); Tier 3 does.
  10. 3D visualizer — real embeddings only, deterministic (not random).
  11. EEG tracker — synthetic, labelled, value in [0.1, 1.0], stops.
  12. Serve check — launch app.py on 127.0.0.1:7862, GET /, expect 200.

Exit 0 only if every check passes. Prints measured numbers, not opinions.
Usage: python test_headless.py
"""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    print("== 1. Index build ==")
    import build_index

    t0 = time.time()
    build_index.main()
    build_s = time.time() - t0
    import faiss

    idx = faiss.read_index(os.path.join(HERE, "syntropic.index"))
    check("index builds", idx.ntotal > 0,
          f"ntotal={idx.ntotal} dim={idx.d} in {build_s:.1f}s")

    print("== 2. Retrieval ==")
    from qce import QuantumCognitiveEngine

    engine = QuantumCognitiveEngine()
    chunks, dists = engine.retrieve(
        "What is the gamma limit from GW250114?", top_k=3)
    check("retrieval returns chunks", len(chunks) > 0,
          f"n={len(chunks)} top_L2={dists[0]:.2f} "
          f"sources={[c['source'] for c in chunks]}")
    check("top distance is finite", len(dists) > 0 and dists[0] < 1e6,
          f"top_L2={dists[0]:.2f}")

    print("== 3. Simulator quantum job ==")
    from quantum import QuantumLink

    qlink = QuantumLink(mode="simulator")
    check("quantum mode is simulator", qlink.mode == "simulator",
          f"backend={qlink.backend_name}")
    qc, counts, entropy = qlink.execute_topological_collapse(0.5)
    total = sum(counts.values())
    check("1024 shots measured", total == 1024, f"total={total}")
    check("entropy in [0, 3] bits", 0.0 <= entropy <= 3.0,
          f"entropy={entropy:.4f} bits counts={counts}")
    ghz = counts.get("000", 0) + counts.get("111", 0)
    check("GHZ outcomes dominate", ghz > total * 0.5,
          f"000+111={ghz}/{total}")

    print("== 4. Biometrics (simulated) ==")
    from biometrics import BiometricLayer

    bio = BiometricLayer(mode="simulated", seed=7)
    for _ in range(30):
        bio.simulate_step()
    d = bio.describe()
    check("RMSSD plausible (5-200 ms)", 5.0 <= d["rmssd_ms"] <= 200.0,
          f"rmssd={d['rmssd_ms']} ms n={d['rr_window_n']}")
    check("bio score in [0.1, 1.0]", 0.1 <= d["bio_score"] <= 1.0,
          f"score={d['bio_score']}")
    check("temperature = 1 - score",
          abs(d["temperature"] - (1.0 - d["bio_score"])) < 0.01,
          f"temp={d['temperature']}")

    print("== 5. Ledger round-trip ==")
    from ledger import SyntropicLedger

    tmp = tempfile.mkdtemp()
    idx_path = os.path.join(tmp, "ledger.index")
    mem_path = os.path.join(tmp, "ledger.json")
    led = SyntropicLedger(engine.embed_model, enabled=False,
                          index_path=idx_path, memory_path=mem_path)
    committed, reason = led.commit_resonance(
        "test query", "test synthesis", 0.2)
    check("disabled ledger refuses commit", not committed, reason)
    led.enabled = True
    committed, reason = led.commit_resonance(
        "gamma limit GW250114", "gamma is less than 0.08", 0.2)
    check("enabled ledger commits", committed, reason)
    check("ledger size 1", led.size() == 1, f"size={led.size()}")
    led2 = SyntropicLedger(engine.embed_model, enabled=True,
                           index_path=idx_path, memory_path=mem_path)
    check("ledger persists across reload", led2.size() == 1,
          f"reloaded size={led2.size()}")
    hit = led2.query_swarm("gamma limit GW250114")
    check("ledger query returns the insight", "gamma is less" in hit,
          f"hit={hit[:60]!r}")

    print("== 6. Demo response (unified engine) ==")
    from engine import SwarmEngine

    swarm = SwarmEngine(quantum_mode="simulator", ledger_enabled=False)
    gen = swarm.generate_response(
        "What is the gamma limit from GW250114?", [], "tier2")
    text, telemetry = None, None
    for text, telemetry, _fig in gen:
        pass
    check("demo mode (no LM Studio)", telemetry["mode"] == "demo",
          f"mode={telemetry['mode']}")
    check("demo answer cites a source",
          text is not None and ("Source:" in text or "L2" in text),
          f"words={telemetry['words_streamed']} "
          f"latency={telemetry['latency_s']}s")
    check("telemetry has quantum + bio fields",
          telemetry["quantum_mode"] == "simulator"
          and telemetry["bio_score"] is not None,
          f"quantum={telemetry['quantum_mode']} "
          f"bio={telemetry['bio_score']}")

    print("== 7. Quantum intercept (Tier 3) ==")
    gen = swarm.generate_response("please entangle the qubits", [], "tier3")
    qtext, qtel = None, None
    for qtext, qtel, _fig in gen:
        pass
    check("tier-3 entangle runs the circuit",
          qtext is not None and "QUANTUM EXECUTION COMPLETE" in qtext,
          f"entropy_bits={qtel['quantum_entropy_bits']}")
    check("no mock counts presented as IBM",
          qtext is not None and "SIMULATOR" in qtext
          and "IBM QUANTUM EXECUTION" not in qtext,
          "simulator labelled as simulator")

    print("== 8. VQE compiler (real compilation) ==")
    from vqe import compile_expression, run_vqe, select_hamiltonian

    cases = [
        ("cos(Psi)", 0.5403),
        ("Psi**2", 1.0),
        ("sin(Psi)", -0.8415),
        ("0.5*Psi + 1", 0.5),
        ("exp(Psi)", 0.3679),
    ]
    for expr, expected_gs in cases:
        op, expl = compile_expression(expr)
        gs, _ = run_vqe(op)
        check(f"VQE compiles + optimises {expr}",
              abs(gs - expected_gs) < 0.05,
              f"op={op.to_list()} gs={gs:.4f} expected={expected_gs}")
    op_pauli, expl_pauli = compile_expression("ZZ + XX")
    gs_pauli, _ = run_vqe(op_pauli)
    check("Pauli-term syntax compiles (ZZ + XX)",
          op_pauli.num_qubits == 2 and abs(gs_pauli - (-2.0)) < 0.2,
          f"op={op_pauli.to_list()} gs={gs_pauli:.4f}")
    try:
        compile_expression("Psi + foo")
        check("bad expression fails cleanly", False, "no error raised")
    except ValueError as exc:
        check("bad expression fails cleanly", True, f"error={exc}")
    name_cos, ham_cos, expl_cos = select_hamiltonian("V(Psi) = cos(Psi)")
    check("cos preset still selectable, labelled preset",
          name_cos == "cos-type" and "PRESET" in expl_cos,
          f"ham={ham_cos.to_list()}")
    gs_cos, _ = run_vqe(ham_cos)
    check("cos-type preset VQE returns a finite energy",
          gs_cos == gs_cos and abs(gs_cos) < 10, f"gs={gs_cos:.4f} Ha")

    print("== 9. Tier gating ==")
    from qce import TIERS

    check("tier1 has no quantum access",
          TIERS["tier1"]["quantum_access"] is False, "")
    check("tier3 has quantum access",
          TIERS["tier3"]["quantum_access"] is True, "")
    gen = swarm.generate_response("V(Psi) = cos(Psi)", [], "tier1")
    t1_text, t1_tel = None, None
    for t1_text, t1_tel, _fig in gen:
        pass
    check("tier1 does NOT trigger the VQE intercept",
          t1_tel["vqe_ground_state"] is None
          and "VQE EXECUTION" not in (t1_text or ""),
          f"vqe={t1_tel['vqe_ground_state']}")
    gen = swarm.generate_response("V(Psi) = cos(Psi)", [], "tier3")
    t3_text, t3_tel = None, None
    for t3_text, t3_tel, _fig in gen:
        pass
    check("tier3 chat intercept compiles cos(Psi) for real",
          t3_tel["vqe_ground_state"] is not None
          and abs(t3_tel["vqe_ground_state"] - 0.5403) < 0.05
          and "Compiled Hamiltonian" in (t3_text or ""),
          f"vqe={t3_tel['vqe_ground_state']}")

    print("== 10. 3D visualizer (real embeddings, no dummy vectors) ==")
    from visualizer import TopologicalVisualizer, real_static_embeddings

    static_vecs = real_static_embeddings(
        engine.embed_model, engine.metadata, max_points=10)
    check("static embeddings are real model outputs",
          len(static_vecs) > 0 and static_vecs[0].shape == (768,),
          f"n={len(static_vecs)} dim={static_vecs[0].shape}")
    v1 = engine.embed_model.encode(["gamma limit GW250114"])
    v2 = engine.embed_model.encode(["gamma limit GW250114"])
    import numpy as _np
    check("embeddings are deterministic (not random)",
          bool(_np.allclose(v1, v2)),
          "same text -> same vector")
    viz = TopologicalVisualizer()
    fig = viz.generate_3d_manifold(static_vecs, [], v1[0])
    n_pts = len(fig.data[0].x) if fig.data else 0
    check("manifold figure has real points",
          n_pts == len(static_vecs) + 1, f"points={n_pts}")

    print("== 11. EEG tracker (synthetic, labelled) ==")
    from eeg import BRAINFLOW_AVAILABLE, EEGBiometricTracker

    eeg = EEGBiometricTracker(mode="synthetic")
    msg = eeg.start_stream()
    time.sleep(5.0)
    val = eeg.current_value
    check("EEG value is a float in [0.1, 1.0]",
          isinstance(val, float) and 0.1 <= val <= 1.0,
          f"value={val:.3f} source={eeg.source} "
          f"brainflow_installed={BRAINFLOW_AVAILABLE}")
    eeg.stop_stream()
    check("EEG stops cleanly", not eeg.is_streaming, "stopped")

    print("== 12. Serve check (127.0.0.1:7862) ==")
    proc = subprocess.Popen(
        [os.path.join(HERE, "venv", "bin", "python"), "app.py"],
        cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    served = False
    try:
        for _ in range(60):
            time.sleep(2)
            try:
                with urllib.request.urlopen(
                        "http://127.0.0.1:7862/", timeout=5) as resp:
                    served = resp.status == 200
                    break
            except Exception:
                continue
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    check("app serves HTTP 200 on 127.0.0.1:7862", served,
          "root reachable" if served else "never came up in 120s")

    print()
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"RESULT: {n_pass}/{len(CHECKS)} checks passed")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
