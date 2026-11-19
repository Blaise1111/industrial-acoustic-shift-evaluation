"""Rebuild all P1 summary tables from the standalone prediction ledger."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cost_and_rules import audit_rows, policy_rows
from .data_manifest import assert_disjoint
from .pipeline import WINDOWS
from .stats_and_calib import (
    fewshot_rows, fixed_threshold, holm, mcnemar, operating_point, paired_ci, ranking,
)

REPORTS = ("p1_ranking_and_significance", "p1_fewshot_recalibration",
           "p1_selective_review_eddi_cost", "p1_error_audit_and_band_rules")


def ranking_table(ledger, config):
    tests = ledger[ledger.split.str.endswith("_test")]
    rows, significance = [], []
    for (section, domain, model, seed), group in tests.groupby(
            ["section", "domain", "model", "seed"], sort=True):
        for pooling, prefix in (("q90", ""), ("mean", "mean_")):
            y = group.label.to_numpy()
            scores = group[prefix + "score"].to_numpy()
            metrics = ranking(y, scores, config["max_fpr"])
            metrics["pauc_mcclish"] = roc_auc_score(y, scores, max_fpr=config["max_fpr"])
            operating = operating_point(y, group[prefix + "alert"])
            rows.append(dict(kind="model", section=section, domain=domain, model=model,
                             seed=seed, pooling=pooling, threshold=group[prefix + "threshold"].iloc[0],
                             n=len(group), **metrics, **operating))

    frame = pd.DataFrame(rows)
    metrics = ["auc", "pauc", "ap", "fpr", "recall"]
    seeded = frame[(frame.model == "M2") & (frame.seed != "ensemble")]
    for (section, domain, pooling), group in seeded.groupby(["section", "domain", "pooling"]):
        seed_metrics = [*metrics, "pauc_mcclish"]
        stats = {metric: group[metric].mean() for metric in seed_metrics}
        stats.update({metric + "_sd": group[metric].std(ddof=1) for metric in seed_metrics})
        rows.append(dict(kind="seed_summary", section=section, domain=domain, pooling=pooling,
                         model="M2", seed="mean_of_seeds", n_seeds=len(group), **stats))

    for (section, domain), group in tests.groupby(["section", "domain"], sort=True):
        first = group[group.model == "M1-Mel"].set_index("clip_id")
        second = group[(group.model == "M2") & (group.seed == "ensemble")].set_index("clip_id")
        if set(first.index) != set(second.index):
            raise ValueError("Paired models must contain identical clips")
        first = first.loc[second.index]
        y = second.label.to_numpy()
        a, b = first.alert.to_numpy(dtype=bool), second.alert.to_numpy(dtype=bool)
        won_a, won_b, p = mcnemar(y, a, b)
        estimates = [ranking(y, x.score, config["max_fpr"]) | operating_point(y, x.alert)
                     for x in (first, second)]
        ci = paired_ci(y, first.score.to_numpy(), second.score.to_numpy(), config, a, b)
        item = dict(kind="paired_test", section=section, domain=domain, model="M2 vs M1-Mel",
                    seed="ensemble", pooling="q90", n=len(y), m1_only_correct=won_a,
                    m2_only_correct=won_b, mcnemar_p=p)
        for metric, (low, high) in zip(metrics, ci):
            item.update({f"delta_{metric}": estimates[1][metric] - estimates[0][metric],
                         f"delta_{metric}_ci_low": low, f"delta_{metric}_ci_high": high})
        significance.append(item)
        print(f"Paired inference complete: section {section}, {domain}", flush=True)
    for row, adjusted in zip(significance, holm([item["mcnemar_p"] for item in significance])):
        row["mcnemar_p_holm"] = adjusted
    return pd.DataFrame([*rows, *significance])


def qc_table(ledger):
    unique = ledger.drop_duplicates("clip_id")
    return unique.groupby(["section", "split"], sort=True).agg(
        clips=("clip_id", "size"), sample_rate_min=("sample_rate", "min"),
        sample_rate_max=("sample_rate", "max"), seconds_min=("duration_seconds", "min"),
        seconds_max=("duration_seconds", "max"), silent_clips=("silent", "sum"),
        mean_silence_fraction=("silence_fraction", "mean"),
        clips_with_clipping=("clipping_fraction", lambda column: int((column > 0).sum())),
        max_clipping_fraction=("clipping_fraction", "max"),
        max_absolute_dc_offset=("dc_offset", lambda column: column.abs().max())
    ).reset_index()


def validate_ledger(ledger, config):
    identity = ["clip_id", "sha256", "section", "split", "domain", "label"]
    manifest = ledger[identity].drop_duplicates()
    assert_disjoint(manifest)
    expected_models = {("M0", "n/a"), ("M1-Mel", "n/a"), ("M1-MFCC", "n/a"),
                       ("M2", "ensemble")}
    expected_models.update(("M2", str(seed)) for seed in config["model"]["seeds"])
    evaluation = config["evaluation"]

    if set(ledger.section) != set(config["dataset"]["sections"]):
        raise ValueError("Prediction ledger is missing an official section")
    for section, rows in ledger.groupby("section", sort=True):
        models = rows.groupby(["model", "seed"])
        if set(models.groups) != expected_models:
            raise ValueError(f"Section {section} has missing or extra model variants")
        clip_ids = set(manifest.loc[manifest.section == section, "clip_id"])
        for variant, group in models:
            if group.clip_id.duplicated().any() or set(group.clip_id) != clip_ids:
                raise ValueError(f"Incomplete or duplicate predictions: {section}/{variant}")
            windows = group[WINDOWS].to_numpy()
            if not np.isfinite(windows).all():
                raise ValueError(f"Non-finite window scores: {section}/{variant}")
            calib = group.split.eq("source_calib")
            if sum(group.split.eq("target_fewshot")) != evaluation["fewshot_clips"]:
                raise ValueError(f"Wrong target calibration count: {section}/{variant}")
            for score_column, threshold_column, alert_column, recomputed in (
                ("score", "threshold", "alert",
                 np.quantile(windows, evaluation["pooling_quantile"], axis=1)),
                ("mean_score", "mean_threshold", "mean_alert", windows.mean(axis=1)),
            ):
                np.testing.assert_allclose(group[score_column], recomputed, rtol=1e-7, atol=1e-7)
                threshold = fixed_threshold(group.loc[calib, score_column], evaluation["target_fpr"])
                if not group[threshold_column].eq(threshold).all():
                    raise ValueError(f"Threshold disagrees with source calibration: {section}/{variant}")
                if not group[alert_column].eq(group[score_column] > threshold).all():
                    raise ValueError(f"An alert disagrees with the locked threshold: {section}/{variant}")


def build_reports(ledger):
    parameters = ledger.loc[ledger.analysis_config != "", "analysis_config"]
    if len(parameters) != 1:
        raise ValueError("Prediction ledger must contain exactly one configuration record")
    settings = json.loads(parameters.iloc[0])
    validate_ledger(ledger, settings)
    config = settings["evaluation"]
    active = ledger[(ledger.model != "M2") | (ledger.seed == "ensemble")]
    fewshot = []
    for (section, model), group in active.groupby(["section", "model"], sort=True):
        fewshot.extend(dict(section=section, model=model, **row)
                       for row in fewshot_rows(group, config, WINDOWS))
    primary = active[active.model == "M2"]
    policies = []
    for section, group in primary.groupby("section", sort=True):
        policies.extend(dict(section=section, model="M2 ensemble", **row)
                        for row in policy_rows(group, config))
    return {
        REPORTS[0]: ranking_table(ledger, config),
        REPORTS[1]: pd.DataFrame(fewshot),
        REPORTS[2]: pd.DataFrame(policies),
        REPORTS[3]: pd.DataFrame(audit_rows(active, config)),
        "data_qc_summary": qc_table(ledger),
    }
