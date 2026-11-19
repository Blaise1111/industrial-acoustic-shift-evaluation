"""Selection on development scores and descriptive held-out error audit."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from .features import BAND_NAMES
from .stats_and_calib import fixed_threshold


def outcome(scores, labels, low, high, config, fn_cost):
    y = np.asarray(labels, dtype=bool)
    fp = np.mean(scores[~y] > high)
    fn = np.mean(scores[y] <= low)
    review = (scores > low) & (scores <= high)
    pi = config["prevalence"]
    weighted_review = (1 - pi) * review[~y].mean() + pi * review[y].mean()
    cost = (config["false_positive_cost"] * (1 - pi) * fp + fn_cost * pi * fn
            + config["review_cost"] * weighted_review)
    return dict(fpr=fp, fnr=fn, recall=1 - fn, review_rate=weighted_review,
                observed_review_rate=review.mean(), cost=cost)


def trivial_random_cost(review_rate, config, fn_cost):
    """Expected cost under score-independent random routing and perfect review."""
    prevalence = config["prevalence"]
    always_pass = prevalence * fn_cost
    always_alert = (1 - prevalence) * config["false_positive_cost"]
    best_random = ((1 - review_rate) * min(always_pass, always_alert)
                   + review_rate * config["review_cost"])
    return always_pass, always_alert, best_random


def policy_rows(group, config):
    source = group.loc[group.split == "source_calib", "score"].to_numpy()
    target = group[group.split == "target_test"]
    tau = fixed_threshold(source, config["target_fpr"])
    # Plan on an explicit high-score scenario over source-normal quantile ranks.
    # The assumed anomaly score CDF at a normal quantile u is u ** rank_power.
    values = np.r_[np.nextafter(source.min(), -np.inf), np.unique(source)]
    u = np.searchsorted(np.sort(source), values, side="right") / len(source)
    anomaly = u ** config["planning_anomaly_rank_power"]
    pi = config["prevalence"]
    review = (1 - pi) * (u[None, :] - u[:, None]) + pi * (anomaly[None, :] - anomaly[:, None])
    upper_triangle = np.triu(np.ones(review.shape, dtype=bool))
    rows = []
    for fn_cost in config["false_negative_costs"]:
        base = outcome(target.score.to_numpy(), target.label, tau, tau, config, fn_cost)
        common = dict(fn_cost=fn_cost, baseline_cost=base["cost"], baseline_recall=base["recall"])
        rows.append(dict(**common, policy="source_fixed", budget=0, low=tau, high=tau, **base,
                         savings=0.0, recall_gain=0.0))
        objective = ((1 - pi) * config["false_positive_cost"] * (1 - u[None, :])
                     + pi * fn_cost * anomaly[:, None] + config["review_cost"] * review)
        for budget in config["review_budgets"]:
            feasible = upper_triangle & (review <= budget + 1e-12)
            i, j = np.unravel_index(np.argmin(np.where(feasible, objective, np.inf)), objective.shape)
            low, high = values[i], values[j]
            result = outcome(target.score.to_numpy(), target.label, low, high, config, fn_cost)
            rows.append(dict(**common, policy="planned", budget=budget, low=low, high=high,
                             planning_cost=objective[i, j], planning_review_rate=review[i, j],
                             budget_exceeded=result["review_rate"] > budget + 1e-12,
                             savings=(1 - result["cost"] / base["cost"]) if base["cost"] else np.nan,
                             recall_gain=result["recall"] - base["recall"], **result))
    for row in rows:
        always_pass, always_alert, reference = trivial_random_cost(
            row["review_rate"], config, row["fn_cost"])
        row.update(trivial_always_pass_cost=always_pass,
                   trivial_always_alert_cost=always_alert,
                   best_trivial_random_cost=reference,
                   savings_vs_best_trivial_random=1 - row["cost"] / reference if reference else np.nan)
    return rows


def load_bands(frame):
    return frame.loc[:, list(BAND_NAMES)].to_numpy()


def audit_rows(active, config):
    """Fit rules to source-training false alarms and audit them on target normals."""
    result = []
    primary = active[active.model == "M2"]
    for section, group in primary.groupby("section", sort=True):
        source = group[group.split == "source_train"]
        target = group[group.split == "target_test"].copy()
        normal = target[target.label == 0]
        features = load_bands(source)
        tree = DecisionTreeClassifier(max_depth=config["tree_depth"], min_samples_leaf=10,
                                      random_state=config["seed"]).fit(features, source.alert)
        fitted_leaves = tree.apply(features)
        test_leaves = tree.apply(load_bands(normal))
        structure = tree.tree_

        def visit(node, clauses):
            if structure.feature[node] != _tree.TREE_UNDEFINED:
                name = BAND_NAMES[structure.feature[node]]
                threshold = format(structure.threshold[node], ".17g")
                visit(structure.children_left[node], [*clauses, f"{name} <= {threshold}"])
                visit(structure.children_right[node], [*clauses, f"{name} > {threshold}"])
                return
            train_leaf = source[fitted_leaves == node]
            target_leaf = normal[test_leaves == node]
            result.append(dict(kind="rule", section=section, rule=" AND ".join(clauses) or "ALL",
                               train_n=len(train_leaf), train_fp=int(train_leaf.alert.sum()),
                               train_fpr=train_leaf.alert.mean(), target_n=len(target_leaf),
                               target_fp=int(target_leaf.alert.sum()), target_fpr=target_leaf.alert.mean()))
        visit(0, [])
    pool = active[active.split == "target_test"].copy()
    priority = {"M2": 0, "M1-Mel": 1, "M1-MFCC": 2, "M0": 3}
    pool["model_priority"] = pool.model.map(priority)
    pool["margin"] = (pool.score - pool.threshold) / pool.threshold.abs().clip(lower=1e-12)
    positive = pool.label.eq(1)
    alert = pool.alert.astype(bool)
    transient = positive & alert & ~pool.mean_alert.astype(bool)
    used = set()
    selectors = [
        ("Transient TP (q90 only)", transient, "margin"),
        ("TP", positive & alert, "margin"),
        ("Low-frequency FP candidate", ~positive & alert, BAND_NAMES[0]),
        ("High-energy FN candidate", positive & ~alert, BAND_NAMES[3]),
    ]
    for case, mask, sort_key in selectors:
        candidates = pool[mask & ~pool.clip_id.isin(used)].sort_values(
            ["model_priority", sort_key, "clip_id"], ascending=[True, False, True])
        if candidates.empty:
            result.append(dict(kind="case", case=case, status="No eligible distinct clip"))
            continue
        row = candidates.iloc[0]
        used.add(row.clip_id)
        columns = ["section", "clip_id", "member", "sha256", "label", "score", "threshold",
                   "mean_score", "mean_threshold", "model", "seed", *BAND_NAMES]
        result.append(dict(kind="case", case=case, status="Selected", **row[columns].to_dict()))
    return result
