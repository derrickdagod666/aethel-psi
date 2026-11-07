# Proposed repo structure — `aethel-psi` (public, MIT)

Proposed name: **`aethel-psi`** (GitHub search 2026-11-07: 0 results —
no public repo with that name found).
Owner: **derrickdagod666** (verified via GitHub connector).
Visibility recommendation: **public**. The value here is reuse and
citation — labs will not run a private pipeline they cannot see.
Keep it public; the paid tier is support/consulting, not access to
the code (see `PITCH.md`).

Alternatives if `aethel-psi` is taken or unwanted:
`aethel-psi-synthesis`, `uaf-gw-pipeline`, `primordial-waveform`.

```
aethel-psi/
├── README.md                  # honest top-level README (this package's)
├── LICENSE                    # MIT
├── LICENSE-NOTE.md            # MIT vs Apache-2.0 reasoning
├── AUDIT.md                   # asset audit: shippable vs missing
├── PITCH.md                   # paid-tier / consulting outline
├── CHECKLIST.md               # exact steps to create + push
├── .gitignore
├── requirements.txt           # pinned versions from the verified runs
├── gw-pipeline/
│   ├── README.md
│   ├── uaf_bilby_plugin.py    # canonical fixed waveform (copy of gate3_pe.py)
│   ├── gate3_pe.py            # GW250114 PE (gamma, f_merger, width_frac)
│   ├── gate3_common.py        # interferometer build from GWOSC strain
│   ├── gate3_data.py          # GW250114 data prep (downloads GWOSC HDF5)
│   ├── gate3_profile.py       # profile likelihood
│   ├── gate3_validate.py      # validation / timing-offset checks
│   ├── gate3_231123_*.py      # GW231123 equivalents
│   ├── precessing_profile.py  # GW231123 XPHM precessing profile
│   ├── tphm_profile.py        # TPHM cross-check
│   ├── xphm_profile.py        # GW200129 XPHM profile
│   └── tilt_check.py          # GW200129 tilt-robustness check
├── qiskit-stabilizer/
│   ├── README.md
│   └── test_qiskit.py         # 3-qubit stabilizer, Qiskit 2.x, PASS 4/4
├── validation-suites/
│   ├── README.md
│   ├── aethel_psi_suite_v2.py
│   ├── aethel_psi_suite_v3.py
│   ├── aethel_psi_suite_v4.py
│   ├── aethel_psi_suite_v5.py # final: PASS=25 FAIL=18 OPEN=6
│   ├── test_core.py
│   ├── test_bilby_real.py
│   └── test_bilby_qiskit.py
├── act-dr6/
│   ├── README.md
│   ├── a5_real.py             # TT-only real ACT DR6.02
│   ├── a5_joint.py            # TT+TE+EE joint (p=0.105)
│   └── camb_joint_refit.py
├── stacking/
│   ├── README.md
│   ├── stack_profile.py
│   ├── stack_data.py
│   └── fisher_stacking.py
├── drive-code/
│   ├── README.md
│   ├── quantum_cognitive_pennylane.py   # Drive code.py, as-is
│   └── quantum_tests_harmonic_full.py  # fragment, not runnable as-is
├── aethelred-agent/
│   ├── README.md              # sanitization instructions (NO code yet)
│   └── .env.example           # variable names only, no values
└── docs/
    └── aethel-psi-synthesis-third-revised-edition.pdf
```

## What is NOT in the tree (deliberately)

- GWOSC strain (`.hdf5`, ~40 MB/event) and ACT DR6 data files —
  downloaded by the scripts; see component READMEs.
- Python venvs (719 MB + 611 MB locally) — rebuilt from
  `requirements.txt`.
- `ApothegmSovereignEngine` and the VQE drug app — source not
  found in Drive or workspace; add when located (see `AUDIT.md`).
- Aethelred Agent code — blocked until sanitized (hardcoded keys
  in the Drive copy). Ship only the sanitized version.
