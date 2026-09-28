"""Core tests. Run with:  pytest -q

A small synthetic dataset is used where the true relationships are known,
so every test can check the system against the right answer.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.checks import ks_stat
from sentinel.config import DatasetConfig
from sentinel.diagnose import analyse
from sentinel.evaluate import wilson
from sentinel.profile import build_profile, load_profile
from sentinel.relationships import isolability, learn, signature_matrix
from sentinel.scenarios import Event, apply_fault, make_batch

ROOT = Path(__file__).resolve().parent.parent


def synthetic(n=6000, seed=0) -> pd.DataFrame:
    """a, b, c, d are chained: b follows a, c follows a and b, d follows c. y depends on all."""
    rng = np.random.default_rng(seed)
    a = rng.normal(50, 10, n)
    b = 0.8 * a + rng.normal(0, 2, n)
    c = 0.5 * a + 0.5 * b + rng.normal(0, 2, n)
    d = 1.2 * c + rng.normal(0, 2, n)
    y = 2 * a + b - c + 0.5 * d + rng.normal(0, 3, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "y": y})


@pytest.fixture(scope="module")
def profile(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synthetic")
    synthetic().to_csv(folder / "data.csv", index=False)
    cfg = DatasetConfig(name="synthetic", title="Synthetic", csv="data.csv", target="y",
                        monitored=["a", "b", "c", "d"], batch_size=200, shift_driver="a")
    return build_profile(cfg, folder, log=lambda *_: None)


# ---------- building blocks ----------

def test_ks_is_zero_for_identical_and_one_for_disjoint():
    ref = np.sort(np.arange(100.0))
    assert ks_stat(ref, np.arange(100.0)) == pytest.approx(0.0, abs=1e-9)
    assert ks_stat(ref, np.arange(1000.0, 1100.0)) == pytest.approx(1.0)


def test_wilson_interval_contains_rate():
    lo, hi = wilson(15, 30)
    assert lo < 0.5 < hi
    assert wilson(0, 30)[0] == 0.0


def test_relationships_recover_true_coefficient():
    df = synthetic(seed=1)
    rels = learn(df.iloc[:4000], df.iloc[4000:], ["a", "b", "c", "d"])
    rule_d = next(r for r in rels if r.target == "d")
    assert rule_d.usable and rule_d.r2 > 0.9
    assert "c" in rule_d.parents
    assert rule_d.coefs[rule_d.parents.index("c")] == pytest.approx(1.2, abs=0.1)


def test_signature_signs_follow_the_formula():
    df = synthetic(seed=2)
    rels = [r for r in learn(df.iloc[:4000], df.iloc[4000:], ["a", "b", "c", "d"]) if r.usable]
    S = signature_matrix(rels, ["a", "b", "c", "d"])
    for i, r in enumerate(rels):
        j = ["a", "b", "c", "d"].index(r.target)
        assert S[i, j] > 0                       # target reads high -> residual goes up
        for p, coef in zip(r.parents, r.coefs):
            k = ["a", "b", "c", "d"].index(p)
            assert np.sign(S[i, k]) == -np.sign(coef)  # parent reads high -> prediction goes up


def test_isolability_is_symmetric_with_unit_diagonal():
    M = isolability(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1.0)


def test_faults_follow_their_formulas():
    class P:
        typical = {"a": 50.0}
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
    for fault, amount, expected in [("percent", 10, [11, 22, 33]), ("offset", 2, [12, 22, 32]),
                                    ("unit", 0.1, [1, 2, 3]), ("stuck", None, [10, 10, 10])]:
        batch = base.copy()
        apply_fault(batch, Event("fault", 0, feature="a", fault=fault, amount=amount), rng, P)
        assert batch["a"].tolist() == pytest.approx(expected)


# ---------- the whole system on synthetic data ----------

def test_healthy_batches_are_mostly_normal(profile):
    verdicts = [analyse(profile, make_batch(profile, k, [], seed=3), k, detail=False)["verdict"] for k in range(20)]
    alarms = sum(v in ("broken", "unclear", "quality") for v in verdicts)
    assert alarms <= 2


def test_harmless_shift_is_not_called_a_fault(profile):
    events = [Event("shift", 0, strength=1.0)]
    res = [analyse(profile, make_batch(profile, k, events, seed=4), k, detail=False) for k in range(15)]
    # Most batches, not all: rules that predict the shifted driver from its own effects are only
    # approximately stable when relationships are noisy (see docs/HOW_IT_WORKS.md, limitations).
    assert sum(r["verdict"] == "shift" for r in res) >= 10
    assert sum(r["baseline"]["alarm"] for r in res) >= 12   # the standard check alarms on the same data


def fault_results(profile, feature, n=8, seed=5):
    size = 4 * next(r for r in profile.usable if r.target == feature).sigma
    events = [Event("fault", 0, feature=feature, fault="offset", amount=size)]
    return [analyse(profile, make_batch(profile, k, events, seed=seed), k) for k in range(n)]


def test_offset_fault_on_distinct_feature_is_named(profile):
    res = fault_results(profile, "b")
    assert all(r["verdict"] == "broken" and r["suspect"] == "b" for r in res)


def test_similar_fingerprints_give_unclear_not_a_wrong_answer(profile):
    # c and d leave almost the same fingerprint (d is predicted from c alone),
    # so the system should say "unclear" and list d first, never confidently blame c
    i, j = profile.features.index("c"), profile.features.index("d")
    assert profile.iso[i, j] > 0.85
    res = fault_results(profile, "d")
    assert all(r["verdict"] in ("broken", "unclear") for r in res)
    assert all(r["candidates"][0]["feature"] == "d" for r in res)
    assert not any(r["verdict"] == "broken" and r["suspect"] != "d" for r in res)


def test_unit_bug_is_a_quality_issue_not_drift(profile):
    events = [Event("fault", 0, feature="a", fault="unit", amount=0.1)]
    r = analyse(profile, make_batch(profile, 0, events, seed=6), 0)
    assert r["verdict"] == "quality"
    assert "a" in r["masked"]


def test_every_verdict_has_a_headline_and_explanation(profile):
    cases = [[], [Event("shift", 0)], [Event("fault", 0, feature="b", fault="noise", amount=20)],
             [Event("fault", 0, feature="c", fault="missing", amount=90)]]
    for events in cases:
        r = analyse(profile, make_batch(profile, 0, events, seed=7), 0)
        assert r["headline"] and r["paragraphs"]


def test_same_seed_gives_same_profile(tmp_path):
    synthetic(n=3000).to_csv(tmp_path / "data.csv", index=False)
    cfg = DatasetConfig(name="s", title="S", csv="data.csv", target="y", monitored=["a", "b", "c", "d"],
                        batch_size=100, shift_driver="a")
    p1 = build_profile(cfg, tmp_path, log=lambda *_: None)
    p2 = build_profile(cfg, tmp_path, log=lambda *_: None)
    assert p1.thresholds["z_limit"]["value"] == pytest.approx(p2.thresholds["z_limit"]["value"])


# ---------- the shipped example datasets ----------

@pytest.mark.parametrize("name", ["diamonds", "abalone"])
def test_example_profiles_load_and_run(name):
    p = load_profile(ROOT / "artifacts" / name)
    if p is None:
        pytest.skip("profile not built yet")
    r = analyse(p, make_batch(p, 0, []), 0)
    assert r["verdict"] in ("normal", "shift", "unclear", "broken", "quality", "insufficient")
