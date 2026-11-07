# License choice — MIT vs Apache-2.0

**Recommendation: MIT** (included as `LICENSE`).

## Why MIT for this repo

- This is a research data-analysis pipeline (a Bilby waveform plugin,
  validation suites, a Qiskit demo). The goal is maximum reuse: other
  GW groups should be able to run it, fork it, and cite it with the
  least friction. MIT is the shortest, most widely understood
  permissive licence for that.
- There is no patentable invention in this repo. The theory itself
  cannot be patented; the code is analysis tooling, not a device or
  a novel method with patent claims attached.
- MIT keeps the copyright line (Derrick Irons, The Aethelred
  Institute, 2026) and the "AS IS" disclaimer, which matters here:
  the code ships with known FAILs (see README), and the disclaimer
  is the honest legal frame for that.

## Why not Apache-2.0 (now)

- Apache-2.0's main advantage over MIT is its express patent grant
  and patent-retaliation clause. That only matters if the repo
  contains, or will contain, something you intend to patent.
- Apache-2.0 is longer and requires preserving NOTICE files; for a
  research pipeline it adds friction without adding protection
  today.

## When to switch

If/when a patentable application ships in this repo — e.g. a working
Metric Phase-Vocoder device driver, a QCI instrument-control
method, or a Holographic Data Compression codec with patent claims —
re-license that component (or the repo) under **Apache-2.0** before
publishing it, and take a provisional patent filing first. The
provisional patent covers a *device or method*, never the Ψ(x)
equation itself. Until then, MIT is the right call, and switching
later is straightforward while Putt is the sole copyright holder.
