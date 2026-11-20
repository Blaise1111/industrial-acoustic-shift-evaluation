import numpy as np
import pytest

from src.evidence_reaudit import paired_class_draws, reviewed_outcome, routing_counts, split_source_proxy


def test_boundaries_and_imperfect_review():
    counts = routing_counts([0,1,2,3, 0,1,2,3], [0]*4+[1]*4, 1,2)
    assert counts.tolist() == [[2,1,1],[2,1,1]]
    r = reviewed_outcome(counts,.1,10,1,.5,.8,.9)
    assert r["model_recall"] == .25
    assert r["system_recall"] == pytest.approx(.45)
    assert r["system_fpr"] == pytest.approx(.275)
    assert r["system_fnr"] == pytest.approx(.55)
    assert r["system_cost"] == pytest.approx(.9225)
    assert r["matched_random_cost"] == pytest.approx(.8725)


@pytest.mark.parametrize("q", [0, 1])
def test_perfect_or_no_review(q):
    counts = [[3*(1-q),3*q,0],[2*(1-q),2*q,0]]
    r = reviewed_outcome(counts,.05,10,1,.5,1,1)
    assert r["weighted_target_review_rate"] == q
    assert r["raw_target_review_rate"] == q
    assert r["system_recall"] == q
    assert r["matched_random_cost"] == pytest.approx(.5)


def test_three_way_and_bootstrap_groups():
    a,b,c=split_source_proxy(20,3,.6,np.random.default_rng(4))
    assert [len(a),len(b),len(c)] == [12,3,5]
    assert len(set(np.r_[a,b,c]))==20
    y=np.array([0,1,0,1,0])
    for label,ids in enumerate(paired_class_draws(y,10,np.random.default_rng(4))):
        assert (y[ids]==label).all()

def test_review_rate_denominators_diverge_at_budget():
    counts = [[80,17,3],[44,37,19]]
    r = reviewed_outcome(counts,.05,10,1,.5,1,1)
    assert r["weighted_target_review_rate"] == pytest.approx(.18)
    assert r["raw_target_review_rate"] == pytest.approx(.27)
    assert r["weighted_target_review_rate"] <= .2 < r["raw_target_review_rate"]
    assert r["model_recall"] == pytest.approx(.19)
    assert r["system_recall"] == pytest.approx(.56)
