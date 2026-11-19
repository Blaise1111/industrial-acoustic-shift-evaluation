"""Clip-level metrics, paired inference and three-shot threshold recalibration."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def operating_point(labels, alerts):
    y, a = np.asarray(labels, dtype=bool), np.asarray(alerts, dtype=bool)
    tp, fp, fn, tn = (int(mask.sum()) for mask in (y & a, ~y & a, y & ~a, ~y & ~a))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, fpr=fp / (fp + tn), recall=tp / (tp + fn))


def partial_auc(y, scores, max_fpr):
    """Area under ROC up to max_fpr, divided by max_fpr."""
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    cut = np.searchsorted(fpr, max_fpr, side="right")
    x = np.r_[fpr[:cut], max_fpr]
    z = np.r_[tpr[:cut], np.interp(max_fpr, fpr, tpr)]
    return float(np.trapezoid(z, x) / max_fpr)


def ranking(y, scores, max_fpr):
    return dict(auc=roc_auc_score(y, scores), pauc=partial_auc(y, scores, max_fpr),
                ap=average_precision_score(y, scores))


def fixed_threshold(scores, fpr):
    """With strict '>', exceedances never exceed floor(n * FPR)."""
    ordered = np.sort(scores)
    return float(ordered[len(ordered) - 1 - int(np.floor(len(ordered) * fpr))])


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    corrected = np.empty_like(p, dtype=float)
    corrected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return corrected


def mcnemar(y, first, second):
    a, b = first == y, second == y
    first_only, second_only = int((a & ~b).sum()), int((b & ~a).sum())
    total = first_only + second_only
    return first_only, second_only, binomtest(first_only, total).pvalue if total else 1.0


def paired_ci(y, reference, candidate, config, reference_alert, candidate_alert):
    """Resample within class, preserving model pairs on each clip."""
    rng = np.random.default_rng(config["seed"])
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    draws_neg = rng.choice(neg, size=(config["bootstrap_replicates"], len(neg)))
    draws_pos = rng.choice(pos, size=(config["bootstrap_replicates"], len(pos)))
    indices = np.concatenate((draws_neg, draws_pos), axis=1)
    target = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    differences = []
    for rows in indices:
        a = ranking(target, reference[rows], config["max_fpr"])
        b = ranking(target, candidate[rows], config["max_fpr"])
        differences.append([b[key] - a[key] for key in a])
    delta = candidate_alert.astype(float) - reference_alert
    differences = np.column_stack((differences, delta[draws_neg].mean(axis=1),
                                   delta[draws_pos].mean(axis=1)))
    return np.quantile(differences, [0.025, 0.975], axis=0).T


def recalibrated_thresholds(source_clips, target_windows, fpr, pooling_quantile):
    base = fixed_threshold(source_clips, fpr)
    target_clips = np.quantile(target_windows, pooling_quantile, axis=1)
    source_iqr = np.subtract(*np.quantile(source_clips, [0.75, 0.25]))
    target_iqr = np.subtract(*np.quantile(target_clips, [0.75, 0.25]))
    ratio = target_iqr / source_iqr if min(source_iqr, target_iqr) > 1e-12 else 1.0
    return {
        "Uncalibrated": base,
        "Mean shift": base + target_clips.mean() - source_clips.mean(),
        "Median-IQR": np.median(target_clips) + (base - np.median(source_clips)) * ratio,
        "Window q95": np.quantile(target_windows, 1 - fpr),
    }


def fewshot_rows(group, config, windows):
    calib = group[group.split == "source_calib"]
    few = group[group.split == "target_fewshot"]
    target = group[group.split == "target_test"]
    scores = calib.score.to_numpy()
    arguments = config["target_fpr"], config["pooling_quantile"]
    methods = recalibrated_thresholds(scores, few[windows].to_numpy(), *arguments)
    result = [dict(kind="target", method=name, threshold=t, **operating_point(
        target.label, target.score > t)) for name, t in methods.items()]
    rng = np.random.default_rng(config["seed"])
    for replicate in range(config["simulation_replicates"]):
        holdout = rng.choice(len(calib), config["fewshot_clips"], replace=False)
        mask = ~np.isin(np.arange(len(calib)), holdout)
        for name, t in recalibrated_thresholds(scores[mask], calib.iloc[holdout][windows].to_numpy(),
                                                *arguments).items():
            result.append(dict(kind="simulation", replicate=replicate, method=name,
                               threshold=t, fpr=float(np.mean(scores[mask] > t))))
    replicates = [row for row in result if row["kind"] == "simulation"]
    for name in methods:
        values = [row["fpr"] for row in replicates if row["method"] == name]
        thresholds = [row["threshold"] for row in replicates if row["method"] == name]
        lo, hi = np.quantile(values, [0.025, 0.975])
        result.append(dict(kind="simulation_summary", method=name, fpr=np.mean(values),
                           fpr_sd=np.std(values, ddof=1), fpr_ci_low=lo, fpr_ci_high=hi,
                           threshold=np.mean(thresholds), threshold_sd=np.std(thresholds, ddof=1)))
    return result
