"""Post-hoc clip-ledger audit. The published estimands remain available unchanged."""
import numpy as np

from .stats_and_calib import recalibrated_thresholds


def routing_counts(scores, labels, low, high):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if (scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all()
            or not np.isfinite([low, high]).all() or low > high or set(labels) != {0, 1}):
        raise ValueError("Routing requires finite scores, ordered thresholds and both binary classes")
    route = np.select([scores <= low, scores > high], [0, 2], default=1)
    return np.stack([np.bincount(route[labels == y], minlength=3) for y in (0, 1)])


def reviewed_outcome(counts, pi, fn_cost, fp_cost, review_cost, sensitivity, specificity):
    if (not all(0 <= value <= 1 for value in (pi, sensitivity, specificity))
            or not all(np.isfinite(x) and x >= 0 for x in (fn_cost, fp_cost, review_cost))):
        raise ValueError("Rates must be probabilities and costs finite and nonnegative")
    counts = np.asarray(counts)
    if counts.shape != (2, 3) or not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(axis=1) <= 0).any():
        raise ValueError("Review requires pass/review/alert counts for both classes")
    normal, positive = counts / counts.sum(axis=1, keepdims=True)
    q = (1-pi)*normal[1] + pi*positive[1]
    observed = counts[:, 1].sum()/counts.sum()
    fpr = normal[2] + normal[1]*(1-specificity)
    fnr = positive[0] + positive[1]*(1-sensitivity)
    cost = (1-pi)*fp_cost*fpr + pi*fn_cost*fnr + q*review_cost
    always_pass, always_alert = pi*fn_cost, (1-pi)*fp_cost
    random = ((1-q)*min(always_pass, always_alert)
              + q*(review_cost + (1-pi)*fp_cost*(1-specificity) + pi*fn_cost*(1-sensitivity)))
    return dict(weighted_target_review_rate=q, raw_target_review_rate=observed,
                normal_review_rate=normal[1], anomaly_review_rate=positive[1],
                model_recall=positive[2], system_recall=positive[2]+positive[1]*sensitivity,
                system_fpr=fpr, system_fnr=fnr, system_cost=cost,
                matched_random_cost=random, cost_saving_vs_random=random-cost,
                relative_saving_vs_random=(random-cost)/random if random else np.nan,
                random_unsent_action="pass" if always_pass <= always_alert else "alert")


def split_source_proxy(n, shots, fit_fraction, rng):
    order = rng.permutation(n)
    fit_count = int(n*fit_fraction)
    if not 0 < fit_count < n-shots or shots < 1:
        raise ValueError("Source calibration does not support nonempty fit/proxy/evaluation splits")
    return order[:fit_count], order[fit_count:fit_count+shots], order[fit_count+shots:]


def paired_class_draws(labels, n, rng):
    if set(labels) != {0, 1} or n < 1:
        raise ValueError("Paired bootstrap requires both classes and at least one replicate")
    groups = [np.flatnonzero(labels == y) for y in (0, 1)]
    return [rng.choice(group, size=(n,len(group)), replace=True) for group in groups]


def recalibrate(source, proxy_windows, ledger_config):
    return recalibrated_thresholds(source, proxy_windows, ledger_config["target_fpr"],
                                   ledger_config["pooling_quantile"])
