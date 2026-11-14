#!/usr/bin/env python3
"""
quinovet_test_campaign.py -- A meticulous, creative test campaign for
the 9-qubit Shor-code module (quinovet_psigate_v4.py) and its two
decoders, built around the code's two-tier "triplet" anatomy: three
GHZ triplets (inner bit-flip codes) nested in an outer phase-flip code.

Every prediction below was registered on 2026-11-13 BEFORE the
campaign was run. Failures are reported as failures.

REGISTERED PREDICTIONS
  PT1 (triplet anatomy): the one-block reduced state of |0_L> has von
     Neumann entropy exactly 0 bits (product of triplets) and of
     |+_L> exactly 1 bit (the macro-GHZ tier carries one ebit across
     every block cut). X-error syndrome bits are block-local for all
     9 qubits (an X in block b lights only block b's two Z-checks).
  PT2 (Berry invariance): the Pancharatnam phase of the ENCODED
     PsiGate loop equals the physical one to < 1e-9 rad -- the
     triplet encoding, being a fixed isometry, must preserve it; this
     cross-checks shor_encode against the loop machinery.
  PT3 (error landscape, exhaustive, noiseless readout): over ALL
     weight-2 (324) and weight-3 (2268) Pauli patterns, the
     favourable decoder corrects 100% of in-family patterns; pure-Z
     weight-2 patterns inside one block: 100% correctable; pure-Z
     weight-2 patterns spread across two blocks: < 25% correctable
     (the proven boundary, now measured exhaustively).
  PT4 (biased noise): with Z-biased (eta=10) and X-biased noise, the
     favourable decoder's total failure stays below the single-error
     decoder's at all 8 points; and its silent-logical SHARE of
     failures is higher under Z bias than X bias at p = 0.02 (the
     Z-spread side is its weak side).
  PT5 (correlated noise): under adjacent-pair bursts and under a
     10x hot qubit, favourable total failure < single-error total
     failure at all 4 points.
  PT6 (readout sweep): at fixed data noise, failure grows with the
     readout flip rate r for every voting depth, and 5 rounds <=
     3 rounds <= 1 round at every r.
  PT7 (coherent errors, exploratory with registered direction):
     small unitary over-rotations hurt at least as much as their
     Pauli twirl at matched average error probability, and the
     Z-only coherent condition (phase accumulation across the
     macro-GHZ tier) is the worst of the three conditions.
  PT8 (method cross-check): a full statevector memory experiment
     (projective syndrome readout from the state itself, noiseless
     readout) reproduces the Pauli-frame logical+residual failure
     rate at p in {0.01, 0.02} (recomputed with p_meas = 0) within
     3 sigma at both points.
  PT9 (sabotage): malformed inputs (even round counts, wrong-length
     syndromes, out-of-family single-error syndromes) raise cleanly;
     nothing silently guesses.

Scope: exact statevector and Pauli-frame simulation of an idealized
code. No hardware claims anywhere in this campaign.
"""

import itertools
import json
import math
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quinovet_psigate_v4 as qv

ZERO_SYN = (0,) * 8
BLOCKS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
RESULTS = {}


def classify(correction, error) -> str:
    if correction is None:
        return "erasure"
    residual = qv.pauli_difference(correction, error)
    if residual in qv.STABILIZER_SPAN:
        return "success"
    if qv.syndrome(residual) != ZERO_SYN:
        return "residual"
    return "logical"


def decode_a(voted):
    try:
        return qv.decode_single_qubit_error(voted)
    except ValueError:
        return None


def decode_b(voted):
    return qv.decode_favourable_family(voted)


def frame_trial(rng, error, p_meas, rounds_n, decoder):
    """One Pauli-frame trial with a given error already sampled."""
    true_syn = qv.syndrome(error)
    rounds = [
        qv.simulate_syndrome_measurement(true_syn, p_meas, rng)
        for _ in range(rounds_n)
    ]
    voted = qv.majority_vote_syndromes(rounds)
    return classify(decoder(voted), error)


def iid_error(rng, p, bias=None):
    """Depolarizing error; bias = ('Z', eta) skews Pauli choice."""
    ops = {}
    for q in range(qv.N_QUBITS):
        if rng.random() < p:
            if bias is None:
                ops[q] = str(rng.choice(["X", "Y", "Z"]))
            else:
                axis, eta = bias
                others = [a for a in ("X", "Y", "Z") if a != axis]
                ops[q] = axis if rng.random() < eta / (eta + 2.0) \
                    else str(rng.choice(others))
    return qv.pauli_from_ops(ops)


def run_frame_condition(name, sampler, trials, p_meas=0.0, rounds_n=3,
                        decoders=("A", "B")):
    """sampler(rng) -> error Pauli. Returns per-decoder outcome counts."""
    rng = np.random.default_rng(seed=zlib.crc32(name.encode()))
    counts = {d: {"success": 0, "erasure": 0, "residual": 0, "logical": 0}
              for d in decoders}
    dec_map = {"A": decode_a, "B": decode_b}
    for _ in range(trials):
        error = sampler(rng)
        for d in decoders:
            out = frame_trial(rng, error, p_meas, rounds_n, dec_map[d])
            counts[d][out] += 1
    summary = {}
    for d, c in counts.items():
        fails = c["erasure"] + c["residual"] + c["logical"]
        summary[d] = {
            **c,
            "total_failure_rate": fails / trials,
            "logical_share_of_failures": (c["logical"] / fails) if fails else 0.0,
        }
    RESULTS[name] = summary
    return summary


# ---------------------------------------------------------------------------
# PT1 -- Triplet anatomy
# ---------------------------------------------------------------------------

def von_neumann_entropy_bits(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-14]
    return float(-np.sum(vals * np.log2(vals)))


def test_pt1_anatomy() -> dict:
    zero_l, one_l = qv.shor_logical_basis()
    plus_l = qv.shor_encode(1.0, 1.0)

    def block_entropy(state: np.ndarray) -> float:
        # State as a tensor over the three GHZ-triplet factors; trace
        # out blocks 2 and 3, keep block 1.
        t = state.reshape(8, 8, 8)
        rho = np.tensordot(t, t.conj(), axes=([1, 2], [1, 2]))
        return von_neumann_entropy_bits(rho)

    s_zero = block_entropy(zero_l)
    s_plus = block_entropy(plus_l)

    locality_ok = True
    for block in BLOCKS:
        own_bits = set()
        for q in block:
            own_bits |= {2 * BLOCKS.index(block), 2 * BLOCKS.index(block) + 1}
        for q in block:
            syn = qv.syndrome(qv.pauli_from_ops({q: "X"}))
            lit = {i for i, b in enumerate(syn[:6]) if b}
            locality_ok &= lit <= own_bits and syn[6:] == (0, 0)

    out = {
        "entropy_block_of_0L_bits": s_zero,
        "entropy_block_of_plusL_bits": s_plus,
        "x_syndrome_block_locality": bool(locality_ok),
        "pass": bool(abs(s_zero) < 1e-9 and abs(s_plus - 1.0) < 1e-9
                     and locality_ok),
    }
    RESULTS["PT1_anatomy"] = out
    return out


# ---------------------------------------------------------------------------
# PT2 -- Berry phase invariance under the triplet encoding
# ---------------------------------------------------------------------------

def test_pt2_berry_invariance(n_per_edge: int = 200) -> dict:
    zero = np.array([1.0, 0.0], dtype=complex)
    pts = qv.psigate_loop_points(n_per_edge)
    phys = [qv.psigate(A, T) @ zero for A, T in pts]
    enc = []
    for A, T in pts:
        psi = qv.psigate(A, T) @ zero
        enc.append(qv.shor_encode(psi[0], psi[1]))

    def pancharatnam(states):
        prod = 1.0 + 0.0j
        n = len(states)
        for k in range(n):
            prod *= np.vdot(states[k], states[(k + 1) % n])
        return float(-np.angle(prod))

    g_phys = pancharatnam(phys)
    g_enc = pancharatnam(enc)
    out = {
        "gamma_physical": g_phys,
        "gamma_encoded": g_enc,
        "abs_difference": abs(g_phys - g_enc),
        "pass": bool(abs(g_phys - g_enc) < 1e-9),
    }
    RESULTS["PT2_berry_invariance"] = out
    return out


# ---------------------------------------------------------------------------
# PT3 -- Exhaustive weight-2 and weight-3 error landscape (noiseless)
# ---------------------------------------------------------------------------

def all_patterns_of_weight(w: int):
    for qubits in itertools.combinations(range(qv.N_QUBITS), w):
        for ops_choice in itertools.product("XYZ", repeat=w):
            yield qv.pauli_from_ops(dict(zip(qubits, ops_choice)))


def in_favourable_family(err) -> bool:
    x, z = err
    for block in BLOCKS:
        if sum(x[q] for q in block) > 1:
            return False
    z_blocks = {bi for bi, block in enumerate(BLOCKS)
                if any(z[q] for q in block)}
    return len(z_blocks) <= 1


def test_pt3_landscape() -> dict:
    report = {}
    for w in (2, 3):
        total = 0
        fam_total = fam_ok = 0
        a_ok = b_ok = 0
        z_same_total = z_same_ok = 0
        z_cross_total = z_cross_ok = 0
        for err in all_patterns_of_weight(w):
            total += 1
            syn = qv.syndrome(err)
            out_b = classify(decode_b(syn), err)
            out_a = classify(decode_a(syn), err)
            b_ok += out_b == "success"
            a_ok += out_a == "success"
            if in_favourable_family(err):
                fam_total += 1
                fam_ok += out_b == "success"
            x, z = err
            if w == 2 and not any(x) and sum(z) == 2:
                zq = [q for q in range(9) if z[q]]
                same = any(zq[0] in b and zq[1] in b for b in BLOCKS)
                if same:
                    z_same_total += 1
                    z_same_ok += out_b == "success"
                else:
                    z_cross_total += 1
                    z_cross_ok += out_b == "success"
        report[f"weight_{w}"] = {
            "total": total,
            "favourable_correct": b_ok,
            "favourable_correct_rate": b_ok / total,
            "single_correct": a_ok,
            "single_correct_rate": a_ok / total,
            "in_family": fam_total,
            "in_family_correct": fam_ok,
            "in_family_rate": (fam_ok / fam_total) if fam_total else None,
        }
        if w == 2:
            report["weight_2"]["pure_z_same_block"] = {
                "total": z_same_total, "correct": z_same_ok,
                "rate": z_same_ok / z_same_total}
            report["weight_2"]["pure_z_cross_block"] = {
                "total": z_cross_total, "correct": z_cross_ok,
                "rate": z_cross_ok / z_cross_total}
    w2 = report["weight_2"]
    w3 = report["weight_3"]
    out = {
        **report,
        "pass": bool(
            w2["in_family_rate"] == 1.0 and w3["in_family_rate"] == 1.0
            and w2["pure_z_same_block"]["rate"] == 1.0
            and w2["pure_z_cross_block"]["rate"] < 0.25
        ),
    }
    RESULTS["PT3_landscape"] = out
    return out


# ---------------------------------------------------------------------------
# PT4 -- Biased noise (Z-heavy and X-heavy), paired decoders
# ---------------------------------------------------------------------------

def test_pt4_bias(trials: int = 20_000) -> dict:
    out = {}
    for label, bias in (("z_bias_eta10", ("Z", 10.0)),
                        ("x_bias_eta10", ("X", 10.0))):
        out[label] = {}
        for p in (0.005, 0.01, 0.02, 0.05):
            name = f"PT4_{label}_p{p}"
            out[label][str(p)] = run_frame_condition(
                name,
                lambda rng, p=p, bias=bias: iid_error(rng, p, bias),
                trials, p_meas=p, rounds_n=3)
    b_below_everywhere = all(
        out[label][str(p)]["B"]["total_failure_rate"]
        < out[label][str(p)]["A"]["total_failure_rate"]
        for label in out for p in (0.005, 0.01, 0.02, 0.05)
    )
    share_z = out["z_bias_eta10"]["0.02"]["B"]["logical_share_of_failures"]
    share_x = out["x_bias_eta10"]["0.02"]["B"]["logical_share_of_failures"]
    out["pass"] = bool(b_below_everywhere and share_z > share_x)
    out["logical_share_B_p002"] = {"z_bias": share_z, "x_bias": share_x}
    RESULTS["PT4_bias"] = out
    return out


# ---------------------------------------------------------------------------
# PT5 -- Correlated noise: adjacent-pair bursts and a hot qubit
# ---------------------------------------------------------------------------

def burst_error(rng, p_burst: float, p_bg: float):
    ops = {}
    for q in range(qv.N_QUBITS):
        if rng.random() < p_bg:
            ops[q] = str(rng.choice(["X", "Y", "Z"]))
    if rng.random() < p_burst:
        start = int(rng.integers(0, 8))
        for q in (start, start + 1):
            ops[q] = str(rng.choice(["X", "Y", "Z"]))
    return qv.pauli_from_ops(ops)


def hot_error(rng, p: float, hot_qubit: int = 4, factor: float = 10.0):
    ops = {}
    for q in range(qv.N_QUBITS):
        rate = p * factor if q == hot_qubit else p
        if rng.random() < rate:
            ops[q] = str(rng.choice(["X", "Y", "Z"]))
    return qv.pauli_from_ops(ops)


def test_pt5_correlated(trials: int = 20_000) -> dict:
    out = {}
    for p_b in (0.01, 0.03):
        out[f"burst_{p_b}"] = run_frame_condition(
            f"PT5_burst_{p_b}",
            lambda rng, p_b=p_b: burst_error(rng, p_b, 0.002),
            trials, p_meas=0.002, rounds_n=3)
    for p in (0.002, 0.005):
        out[f"hot_{p}"] = run_frame_condition(
            f"PT5_hot_{p}",
            lambda rng, p=p: hot_error(rng, p),
            trials, p_meas=p, rounds_n=3)
    out["pass"] = bool(all(
        v["B"]["total_failure_rate"] < v["A"]["total_failure_rate"]
        for v in out.values()))
    RESULTS["PT5_correlated"] = out
    return out


# ---------------------------------------------------------------------------
# PT6 -- Readout-noise sweep x voting depth (favourable decoder)
# ---------------------------------------------------------------------------

def test_pt6_readout(trials: int = 20_000) -> dict:
    out = {}
    for rounds_n in (1, 3, 5):
        out[f"rounds_{rounds_n}"] = {}
        for r in (0.01, 0.05, 0.1, 0.2, 0.3):
            s = run_frame_condition(
                f"PT6_r{rounds_n}_r{r}",
                lambda rng: iid_error(rng, 0.01),
                trials, p_meas=r, rounds_n=rounds_n, decoders=("B",))
            out[f"rounds_{rounds_n}"][str(r)] = s["B"]["total_failure_rate"]
    monotone_r = all(
        out[f"rounds_{n}"]["0.01"] <= out[f"rounds_{n}"]["0.05"]
        <= out[f"rounds_{n}"]["0.1"] <= out[f"rounds_{n}"]["0.2"]
        <= out[f"rounds_{n}"]["0.3"]
        for n in (1, 3, 5))
    depth_order = all(
        out["rounds_5"][str(r)] <= out["rounds_3"][str(r)]
        <= out["rounds_1"][str(r)]
        for r in (0.01, 0.05, 0.1, 0.2, 0.3))
    out["pass"] = bool(monotone_r and depth_order)
    RESULTS["PT6_readout"] = out
    return out


# ---------------------------------------------------------------------------
# PT7 -- Coherent (unitary) errors vs their Pauli twirl, full statevector
# ---------------------------------------------------------------------------

_PAULI_2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

STAB_OPS = [
    {0: "Z", 1: "Z"}, {1: "Z", 2: "Z"}, {3: "Z", 4: "Z"},
    {4: "Z", 5: "Z"}, {6: "Z", 7: "Z"}, {7: "Z", 8: "Z"},
    {q: "X" for q in range(6)}, {q: "X" for q in range(3, 9)},
]


def pauli_matrix_9q(ops: dict) -> np.ndarray:
    """True 512x512 Pauli matrix; qubit 0 is the least significant bit,
    matching quinovet_psigate_v4.apply_pauli's convention."""
    factors = []
    for q in range(8, -1, -1):
        factors.append(_PAULI_2[ops.get(q, "I")])
    mat = factors[0]
    for f in factors[1:]:
        mat = np.kron(mat, f)
    return mat


def symplectic_to_matrix(pauli) -> np.ndarray:
    """Matrix of a symplectic Pauli as Z^z X^x. quinovet's apply_pauli
    implements the same operator up to a global phase (-1)^{z.x},
    which is invisible to every expectation and fidelity used here."""
    x, z = pauli
    ops_x = {q: "X" for q in range(9) if x[q]}
    ops_z = {q: "Z" for q in range(9) if z[q]}
    return pauli_matrix_9q(ops_z) @ pauli_matrix_9q(ops_x)


def test_pt7_coherent(trials: int = 300, frame_trials: int = 20_000) -> dict:
    stab_mats = [pauli_matrix_9q(o) for o in STAB_OPS]
    zero_l, one_l = qv.shor_logical_basis()
    for g in stab_mats:  # the measurement operators must fix the code
        assert np.allclose(g @ zero_l, zero_l, atol=1e-10)
        assert np.allclose(g @ one_l, one_l, atol=1e-10)
    logical_flip = symplectic_to_matrix(qv.LOGICAL_X)    # Z^tensor9
    logical_phase = symplectic_to_matrix(qv.LOGICAL_Z)   # X^tensor9

    psi0 = qv.shor_encode(0.6, 0.8)
    tx = float(np.vdot(psi0, logical_flip @ psi0).real)
    tz = float(np.vdot(psi0, logical_phase @ psi0).real)

    single_mats = {
        (q, a): pauli_matrix_9q({q: a})
        for q in range(9) for a in ("X", "Y", "Z")
    }

    def coherent_trial(rng, theta, q_prob, axes):
        psi = psi0.copy()
        for q in range(9):
            if rng.random() < q_prob:
                axis = str(rng.choice(axes))
                ang = theta * (1 if rng.random() < 0.5 else -1)
                P = single_mats[(q, axis)]
                psi = math.cos(ang) * psi - 1j * math.sin(ang) * (P @ psi)
        syn = []
        for g in stab_mats:
            p_plus = psi + g @ psi
            prob_plus = float(np.vdot(p_plus, p_plus).real) / 4.0
            if rng.random() < prob_plus:
                psi = p_plus / np.linalg.norm(p_plus)
                syn.append(0)
            else:
                p_minus = psi - g @ psi
                psi = p_minus / np.linalg.norm(p_minus)
                syn.append(1)
        corr = qv.decode_favourable_family(tuple(syn))
        psi = symplectic_to_matrix(corr) @ psi
        fx = float(np.vdot(psi, logical_flip @ psi).real)
        fz = float(np.vdot(psi, logical_phase @ psi).real)
        return abs(fx - tx) < 0.5 and abs(fz - tz) < 0.5

    conditions = [
        ("random_axis_t0.20", 0.20, 0.5, ["X", "Y", "Z"]),
        ("random_axis_t0.35", 0.35, 0.5, ["X", "Y", "Z"]),
        ("z_only_t0.20", 0.20, 0.5, ["Z"]),
    ]
    out = {}
    for label, theta, q_prob, axes in conditions:
        rng = np.random.default_rng(seed=zlib.crc32(label.encode()))
        fails = sum(
            not coherent_trial(rng, theta, q_prob, axes)
            for _ in range(trials)
        )
        coh_rate = fails / trials
        coh_sigma = math.sqrt(coh_rate * (1 - coh_rate) / trials)

        p_tw = q_prob * math.sin(theta) ** 2

        def twirl_sampler(rng2, p_tw=p_tw, axes=axes):
            ops = {}
            for q in range(9):
                if rng2.random() < p_tw:
                    ops[q] = str(rng2.choice(axes))
            return qv.pauli_from_ops(ops)

        tw = run_frame_condition(
            f"PT7_twirl_{label}", twirl_sampler, frame_trials,
            p_meas=0.0, rounds_n=1, decoders=("B",))
        out[label] = {
            "theta": theta, "q_prob": q_prob,
            "coherent_failure_rate": coh_rate,
            "coherent_sigma": coh_sigma,
            "twirl_failure_rate": tw["B"]["total_failure_rate"],
            "twirl_p_per_qubit": p_tw,
        }
    ra20 = out["random_axis_t0.20"]
    ra35 = out["random_axis_t0.35"]
    zo = out["z_only_t0.20"]
    out["pass"] = bool(
        ra20["coherent_failure_rate"]
        >= ra20["twirl_failure_rate"] - 2 * ra20["coherent_sigma"]
        and ra35["coherent_failure_rate"]
        >= ra35["twirl_failure_rate"] - 2 * ra35["coherent_sigma"]
        and zo["coherent_failure_rate"] >= ra20["coherent_failure_rate"]
    )
    RESULTS["PT7_coherent"] = out
    return out


# ---------------------------------------------------------------------------
# PT8 -- Statevector memory experiment vs Pauli-frame prediction
# ---------------------------------------------------------------------------

def test_pt8_statevector(trials: int = 1000) -> dict:
    psi0 = qv.shor_encode(0.6, 0.8)
    out = {}
    for p in (0.01, 0.02):
        rng = np.random.default_rng(seed=zlib.crc32(f"PT8_{p}".encode()))
        fails = 0
        for _ in range(trials):
            error = iid_error(rng, p)
            psi = qv.apply_pauli(psi0, error)
            syn = []
            for s in qv.STABILIZERS:
                exp_val = np.vdot(psi, qv.apply_pauli(psi, s)).real
                syn.append(0 if exp_val > 0 else 1)
            corr = qv.decode_favourable_family(tuple(syn))
            fixed = qv.apply_pauli(psi, corr)
            fid = abs(np.vdot(psi0, fixed)) ** 2
            if fid < 0.999:
                fails += 1
        sv_rate = fails / trials
        frame = run_frame_condition(
            f"PT8_frame_p{p}", lambda r, p=p: iid_error(r, p),
            20_000, p_meas=0.0, rounds_n=1, decoders=("B",))
        fr_rate = frame["B"]["total_failure_rate"]
        comb = math.sqrt(sv_rate * (1 - sv_rate) / trials
                         + fr_rate * (1 - fr_rate) / 20_000)
        out[str(p)] = {
            "statevector_failure_rate": sv_rate,
            "frame_failure_rate": fr_rate,
            "combined_sigma": comb,
            "within_3_sigma": bool(abs(sv_rate - fr_rate) < 3 * comb),
        }
    out["pass"] = bool(all(v["within_3_sigma"] for v in out.values()
                           if isinstance(v, dict)))
    RESULTS["PT8_statevector"] = out
    return out


# ---------------------------------------------------------------------------
# PT9 -- Sabotage: malformed inputs must raise, never guess
# ---------------------------------------------------------------------------

def test_pt9_sabotage() -> dict:
    checks = {}

    def raises(fn) -> bool:
        try:
            fn()
            return False
        except (ValueError, RuntimeError):
            return True

    checks["even_rounds_rejected"] = raises(
        lambda: qv.majority_vote_syndromes([(0,) * 8, (0,) * 8]))
    checks["short_round_rejected"] = raises(
        lambda: qv.majority_vote_syndromes([(0,) * 7] * 3))
    multi_syn = qv.syndrome(qv.pauli_from_ops({0: "X", 3: "X", 6: "X"}))
    checks["single_decoder_refuses_multi_error"] = raises(
        lambda: qv.decode_single_qubit_error(multi_syn))
    checks["favourable_decoder_rejects_short_syndrome"] = raises(
        lambda: qv.decode_favourable_family((0,) * 7))
    checks["correction_rejects_wrong_state_size"] = raises(
        lambda: qv.correct_single_pauli_error(
            np.zeros(256, dtype=complex), [(0,) * 8] * 3))
    checks["unknown_syndrome_rejected"] = raises(
        lambda: qv.decode_single_qubit_error(
            qv.syndrome(qv.pauli_from_ops({0: "X", 1: "X", 3: "Z"}))))
    out = {**checks, "pass": bool(all(checks.values()))}
    RESULTS["PT9_sabotage"] = out
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("== PT1 triplet anatomy ==")
    print(json.dumps(test_pt1_anatomy(), indent=2))
    print("== PT2 Berry invariance under encoding ==")
    print(json.dumps(test_pt2_berry_invariance(), indent=2))
    print("== PT3 exhaustive error landscape ==")
    print(json.dumps(test_pt3_landscape(), indent=2))
    print("== PT4 biased noise ==")
    print(json.dumps(test_pt4_bias(), indent=2))
    print("== PT5 correlated noise ==")
    print(json.dumps(test_pt5_correlated(), indent=2))
    print("== PT6 readout sweep ==")
    print(json.dumps(test_pt6_readout(), indent=2))
    print("== PT7 coherent errors ==")
    print(json.dumps(test_pt7_coherent(), indent=2))
    print("== PT8 statevector cross-check ==")
    print(json.dumps(test_pt8_statevector(), indent=2))
    print("== PT9 sabotage ==")
    print(json.dumps(test_pt9_sabotage(), indent=2))

    verdicts = {
        k: v.get("pass") for k, v in RESULTS.items()
        if isinstance(v, dict) and "pass" in v
    }
    print("\n==== CAMPAIGN VERDICTS ====")
    for k, v in verdicts.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "quinovet_test_campaign_results.json")
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"Wrote {path}")
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
