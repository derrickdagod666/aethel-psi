#!/usr/bin/env python3
"""Headless test for the Syntropic Swarm v2.

Runs, in order:
  1. Index build (build_index.py --force) — chunk count, dim, time.
  2. Retrieval — a query about gamma on GW250114; top L2 distance,
     chunks retrieved, sources of the top chunks.
  3. Simulator quantum job — the 3-qubit Aethel-Psi circuit on Aer,
     1024 shots: counts, total shots, Shannon entropy (must be in [0, 3]
     bits for 3 qubits), and the expected dominant outcomes.
  4. Biometrics — 30 simulated RR intervals: RMSSD in a plausible range
     (5-200 ms), bio score in [0.1, 1.0], temperature = 1 - score.
  5. Ledger — commit/query/persistence round-trip on temp files with the
     ledger enabled and bio_score < 0.4; a disabled ledger must refuse;
     reload from disk and confirm the insight survives.
  6. Demo response — engine.generate_response() in demo mode (LM Studio
     is not expected to be running); reports mode, latency, words
     streamed, and whether the answer cites a source.

Exit 0 only if every check passes. Prints measured numbers, not opinions.
Usage: python test_headless.py
"""
import os
import sys
import tempfile
import time

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
    from quantum import QuantumLink, shannon_entropy

    qlink = QuantumLink(mode="simulator")
    check("quantum mode is simulator", qlink.mode == "simulator",
          f"backend={qlink.backend_name}")
    qc, counts, entropy = qlink.execute_topological_collapse(0.5)
    total = sum(counts.values())
    check("1024 shots measured", total == 1024, f"total={total}")
    check("entropy in [0, 3] bits", 0.0 <= entropy <= 3.0,
          f"entropy={entropy:.4f} bits counts={counts}")
    # Phase 0.5 -> RZ(pi/2) on |111>-branch; dominant outcomes are the
    # GHZ-like pair 000/111 plus phase-spread neighbours. Just require the
    # two GHZ outcomes dominate the distribution.
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

    print("== 6. Demo response ==")
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

    print("== 8. VQE (keyword selection + real optimisation) ==")
    from vqe import run_vqe, select_hamiltonian

    name_cos, ham_cos, expl_cos = select_hamiltonian("V(Psi) = cos(Psi)")
    name_gen, ham_gen, expl_gen = select_hamiltonian("V(Psi) = Psi^2")
    check("cos message selects cos-type Hamiltonian",
          name_cos == "cos-type" and "NOT compiled" in expl_cos,
          f"ham={ham_cos.to_list()}")
    check("other message selects generic Hamiltonian",
          name_gen == "generic-type",
          f"ham={ham_gen.to_list()}")
    gs_gen, _ = None, None
    try:
        gs_gen, _ = run_vqe(ham_gen)
    except Exception as exc:  # noqa: BLE001 - report as a check, not a crash
        pass
    check("generic VQE ground state near -2.0 (ZZ+XX)",
          gs_gen is not None and abs(gs_gen - (-2.0)) < 0.2,
          f"gs={gs_gen:.4f} Ha" if gs_gen is not None else f"VQE raised: {exc}")
    gs_cos = None
    try:
        gs_cos, _ = run_vqe(ham_cos)
    except Exception as exc:  # noqa: BLE001 - report as a check, not a crash
        pass
    check("cos-type VQE returns a finite energy",
          gs_cos is not None and gs_cos == gs_cos and abs(gs_cos) < 10,
          f"gs={gs_cos:.4f} Ha" if gs_cos is not None else f"VQE raised: {exc}")

    print("== 9. 3D visualizer (real embeddings, no dummy vectors) ==")
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

    print("== 10. EEG tracker (synthetic, labelled) ==")
    from eeg import BRAINFLOW_AVAILABLE, EEGBiometricTracker

    eeg = EEGBiometricTracker(mode="synthetic")
    msg = eeg.start_stream()
    time.sleep(5.0)
    val = eeg.current_value
    check("EEG value is a float in [0.1, 1.0]",
          isinstance(val, float) and 0.1 <= val <= 1.0,
          f"value={val:.3f} source={eeg.source} "
          f"brainflow_installed={BRAINFLOW_AVAILABLE}")
    check("EEG loop actually computed (not the 0.95 initial)",
          eeg.updates >= 1,
          f"updates={eeg.updates} value={val:.3f}")
    eeg.stop_stream()
    check("EEG stops cleanly", not eeg.is_streaming, "stopped")

    print()
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"RESULT: {n_pass}/{len(CHECKS)} checks passed")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
