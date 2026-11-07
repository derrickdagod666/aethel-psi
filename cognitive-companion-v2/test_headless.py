#!/usr/bin/env python3
"""Headless test for the Architects of Reality QCE v2.

Runs, in order:
  1. Index build (build_index.py --force) — reports chunk count, dim, time.
  2. Retrieval — a query about gamma on GW250114; reports top L2 distance,
     chunks retrieved, and the sources of the top chunks.
  3. Demo response — generate_response() in demo mode (LM Studio is not
     expected to be running); reports mode, latency, words streamed, and
     whether the answer cites a source.
  4. Billing — get_checkout_url() must return (None, message) with billing
     disabled and must not contact Stripe.

Exit 0 only if every check passes. Prints measured numbers, not opinions.
Usage: python test_headless.py
"""
import os
import sys
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

    index = faiss.read_index(os.path.join(HERE, "syntropic.index"))
    import json

    with open(os.path.join(HERE, "syntropic_metadata.json")) as f:
        meta = json.load(f)
    check("index builds", index.ntotal == len(meta) and index.ntotal > 0,
          f"(ntotal={index.ntotal}, dim={index.d}, build={build_s:.1f}s)")
    sources = sorted({m["source"] for m in meta})
    print(f"       sources in index: {sources}")

    print("== 2. Retrieval ==")
    from qce import QuantumCognitiveEngine

    qce = QuantumCognitiveEngine()
    query = "What is the gamma limit from GW250114?"
    t0 = time.time()
    chunks, dists = qce.retrieve(query, top_k=3)
    ret_s = time.time() - t0
    check("retrieval returns 3 chunks", len(chunks) == 3,
          f"(took {ret_s:.2f}s)")
    check("top L2 distance is a real number", bool(dists) and dists[0] >= 0,
          f"(top L2={dists[0]:.3f})" if dists else "")
    if chunks:
        print(f"       top source: {chunks[0]['source']}")
        print(f"       top text (first 200 chars): {chunks[0]['text'][:200]}")
    check("top chunk mentions gamma or GW250114",
          bool(chunks) and ("gamma" in chunks[0]["text"].lower()
                            or "gw250114" in chunks[0]["text"].lower()))

    print("== 3. Demo response ==")
    t0 = time.time()
    outputs = list(qce.generate_response(query, [], "tier2"))
    demo_s = time.time() - t0
    text, telemetry = outputs[-1]
    print(f"       mode={telemetry['mode']} latency={telemetry['latency_s']}s "
          f"words={telemetry['words_streamed']} "
          f"chunks={telemetry['chunks_retrieved']}")
    check("demo mode when LM Studio is down",
          telemetry["mode"] == "demo" and not qce.llm_available)
    check("demo answer cites a source",
          "new-theory-third-revised.pdf" in text
          or "uaf_paper_improved.pdf" in text)
    check("demo answer says it is a demo",
          "DEMO MODE" in text)
    check("demo answer is non-empty", len(text) > 200,
          f"({len(text)} chars, {demo_s:.2f}s)")

    print("== 4. Billing disabled ==")
    from billing import billing_enabled, get_checkout_url

    check("billing is disabled by default", not billing_enabled())
    url, msg = get_checkout_url("tier3")
    check("checkout returns no URL and no charge", url is None,
          f"(message: {msg[:80]}...)")

    print("== Summary ==")
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"  {n_pass}/{len(CHECKS)} checks passed")
    return 0 if n_pass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
