"""Corpus fallback for the Architects of Reality QCE (v2).

Used ONLY if PDF text extraction from Putt's papers yields nothing usable.
Every entry is a summary of Putt's own published work (Third Revised Edition
of the Aethel-Psi Synthesis, Nov 2026, and the October 2026 UAF paper).
These are summaries, not the papers themselves — the PDF corpus is preferred
and this file is only a fallback so the index builder never ships empty.
"""
FALLBACK_CHUNKS = [
    {
        "source": "Aethel-Psi Synthesis, Third Revised Edition (Nov 2026) — summary",
        "text": (
            "The Aethel-Psi Synthesis is built around a primordial harmonic waveform "
            "Psi(x) = A cos(Theta(x)). Three readings of one object are given: a "
            "gravitational cosine field, an informational complex scalar whose "
            "modulus squared tracks Quantum Fisher Information density, and the "
            "principal eigenstate of the Fundamental Cosmological Matrix (FCM). "
            "The parameter conflict A = 54 with lambda = 945 against A ~ 0.1 with "
            "omega ~ 470 rad/s is tabled openly, with the RG-flow bridge between "
            "them labelled HYPOTHESIS, NOT A RESULT."
        ),
    },
    {
        "source": "Aethel-Psi Synthesis, Third Revised Edition (Nov 2026) — summary",
        "text": (
            "Gate 3 tests the chirp-blunting prediction on real gravitational-wave "
            "data. On GW250114 the full dynesty parameter estimation gives gamma "
            "median 0.0235 with a 95% upper limit of 0.0813, and a Bayes factor "
            "ln B = -3.06 against the free-gamma UAF model versus GR. The paper "
            "value gamma = 0.4 is excluded for GW250114-like mergers. Prediction 1 "
            "is revised to gamma ≲ 0.08 (95%) for GW250114-like mergers."
        ),
    },
    {
        "source": "Aethel-Psi Synthesis, Third Revised Edition (Nov 2026) — summary",
        "text": (
            "The spectral dimension D(E) flows from 4 at low energy toward 2 near "
            "the Planck scale. The revised edition replaces the faulty typeset "
            "D(E) with a logistic ansatz D = 4 - 2F, F = 1/(1+exp(-k(x-x0))) with "
            "k = 8 and x0 = 0.5, giving D = 3.96 at x = 0, 3.00 at x = 0.5 and "
            "2.04 at x = 1. The ansatz is labelled ANSATZ, not derived."
        ),
    },
    {
        "source": "UAF paper, Quantum and Geometric Harmonic Resonance (Oct 2026) — summary",
        "text": (
            "The Unified Apophthegmatic Framework (UAF) starts from Axiom 1, "
            "Psi(x) = A cos(Theta(x)) with Theta = k_mu x^mu + phi. The action "
            "includes a non-minimal coupling xi R Psi^2, which suppresses the "
            "effective gravitational constant at large Psi. The trace equation is "
            "R(1 + 2 kappa xi Psi^2) = -kappa T + 6 kappa xi Box Psi^2. The "
            "quantum coupling psi = psi_0 exp(i alpha Psi) gives a connection "
            "A = alpha dPsi with quantisation alpha times 2 pi w = 2 pi n."
        ),
    },
    {
        "source": "UAF paper, Quantum and Geometric Harmonic Resonance (Oct 2026) — summary",
        "text": (
            "The paper predicts a topological Shapiro delay for neutrinos from a "
            "core-collapse supernova, a differential TeV-versus-MeV delay in the "
            "microsecond range, targeted at IceCube-Gen2 and Hyper-Kamiokande. "
            "Validation in the paper uses a Bilby waveform with a blunting "
            "parameter gamma_uaf = 0.4 and a Qiskit 3-qubit stabilizer circuit. "
            "The validation suites v2-v5 later found gamma = 0.4 excluded and "
            "several derivations failing; the Third Revised Edition records "
            "v5 as PASS=25 FAIL=18 OPEN=6."
        ),
    },
    {
        "source": "Aethel-Psi Synthesis, Third Revised Edition (Nov 2026) — summary",
        "text": (
            "One derivation that did close: xi = -1/6 is derived under a stated "
            "Dirac-type operator D_Psi with E = -Psi^2, via the Gilkey-de Witt "
            "heat-kernel coefficient a4, whose R Psi^2 cross term is -1/6. "
            "By contrast, beta_A = 1/14 stays FITTED: the Chern-Simons "
            "quadrature gives integer levels 0, -2, -4, never 1/14. The claim "
            "Lambda_eff = 0 was DROPPED, not proven: the heat-kernel a0 term is "
            "strictly positive at dimensions 2, 3 and 4."
        ),
    },
    {
        "source": "Aethel-Psi Synthesis, Third Revised Edition (Nov 2026) — summary",
        "text": (
            "The spin-gamma hypothesis is CLOSED: both branches fail on real LVK "
            "spin samples. The chi_eff branch implies a negative gamma for "
            "GW250114, and the chi_p branch gives a 95% upper limit of 0.177 "
            "against the allowed 0.0813. The ACT DR6 ripple test is a null "
            "result: joint TT+TE+EE global p = 0.105, A95 = 1.33e-2, so a ripple "
            "amplitude of 0.1 is excluded and 0.005 is not."
        ),
    },
]
