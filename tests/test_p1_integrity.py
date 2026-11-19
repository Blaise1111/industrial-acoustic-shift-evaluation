"""Integrity checks against the verified official data and metric definitions."""
from zipfile import ZipFile

import numpy as np
import pytest

from src.cost_and_rules import trivial_random_cost
from src.pipeline import archive_path, config_file
from src.data_manifest import archive_md5, assert_disjoint, build_manifest
from src.features import Frontend
from src.stats_and_calib import fixed_threshold, holm, operating_point, partial_auc

CONFIG = config_file()


@pytest.fixture(scope="module", params=("valve", "fan"))
def real_manifest(request):
    data = config_file(request.param)
    archive = archive_path(data)
    assert archive.is_file(), "Download the official archive before running the integration tests"
    assert archive_md5(archive) == data["dataset"]["md5"]
    with ZipFile(archive) as zipped:
        yield build_manifest(zipped, data["dataset"])


def test_clip_hashes_and_split_counts(real_manifest):
    assert_disjoint(real_manifest)
    for _, group in real_manifest.groupby("section"):
        counts = group.split.value_counts()
        assert counts["source_train"] == int(0.8 * (counts.source_train + counts.source_calib))
        assert counts.target_fewshot == 3
        assert set(group[group.split.str.endswith("test")].label) == {0, 1}


def test_duplicate_hash_is_rejected(real_manifest):
    duplicates = real_manifest.iloc[:2].copy()
    duplicates.iloc[1, duplicates.columns.get_loc("sha256")] = duplicates.sha256.iloc[0]
    with pytest.raises(ValueError, match="Duplicate sha256"):
        assert_disjoint(duplicates)


def test_window_spectrum():
    frontend = Frontend(CONFIG["audio"])
    t = np.arange(10 * frontend.sr) / frontend.sr
    audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    physical, mel, mfcc, images, bands = frontend.transform(audio)
    assert (physical.shape, mel.shape, mfcc.shape, images.shape) == (
        (19, 4), (19, 128), (19, 80), (19, 64, 100))
    assert frontend.frequencies[np.argmax(frontend.power(audio).mean(axis=0))] == 1000
    assert np.argmax(bands) == 1
    assert physical[0, 0] == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_confusion_matrix():
    assert operating_point([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]) == {
        "tp": 2, "fp": 1, "fn": 1, "tn": 1, "fpr": 0.5, "recall": 2 / 3
    }


@pytest.mark.parametrize("scores, expected", [
    ([0, 0, 1, 1], 1), ([1, 1, 0, 0], 0), ([0, 0, 0, 0], 0.05),
    ([0.9, 0.1, 0.8, 0.7], 0),
    ([0.9, 0.1, 1, 0.7], 0.5),
])
def test_normalized_partial_auc(scores, expected):
    assert partial_auc([0, 0, 1, 1], scores, 0.1) == pytest.approx(expected)


def test_discrete_calibration_threshold_and_holm():
    scores = np.arange(100)
    assert np.mean(scores > fixed_threshold(scores, 0.05)) == 0.05
    assert holm([0.04, 0.01, 0.02]).tolist() == pytest.approx([0.04, 0.03, 0.04])


@pytest.mark.parametrize("fn_cost", [10, 20])
def test_matched_random_review_expected_cost(fn_cost):
    config = CONFIG["evaluation"]
    q = 0.25
    always_pass, always_alert, best_random = trivial_random_cost(q, config, fn_cost)
    assert always_pass == pytest.approx(config["prevalence"] * fn_cost)
    assert always_alert == pytest.approx(1 - config["prevalence"])
    assert best_random == pytest.approx((1 - q) * min(always_pass, always_alert) + q * config["review_cost"])
    assert trivial_random_cost(0, config, fn_cost)[2] == pytest.approx(min(always_pass, always_alert))
    assert trivial_random_cost(1, config, fn_cost)[2] == pytest.approx(config["review_cost"])
