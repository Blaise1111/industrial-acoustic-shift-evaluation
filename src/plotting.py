"""Render the four P1 figures from reconstructed tables and verified case audio."""
import hashlib
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .audio_qc import read_audio
from .cost_and_rules import trivial_random_cost
from .features import Frontend
from .pipeline import WINDOWS
from .reports import REPORTS

PALETTE = ("#285780", "#b75d2e", "#8b3455", "#4f7d60")


def save(figure, directory, filename):
    figure.savefig(directory / filename, dpi=300, facecolor="white")
    plt.close(figure)


def score_shift(primary, output):
    sections = list(primary.groupby("section", sort=True))
    figure, axes = plt.subplots(1, len(sections), figsize=(12.8, 4), layout="constrained")
    populations = (("source_test", 0, "Source normal"), ("target_test", 0, "Target normal"),
                   ("target_test", 1, "Target anomaly"))
    for axis, (section, group) in zip(axes, sections):
        samples = [group.loc[(group.split == split) & (group.label == label), "score"].to_numpy()
                   for split, label, _ in populations]
        support = np.concatenate(samples)
        padding = max(0.2 * np.ptp(support), 1e-10)
        x = np.linspace(support.min() - padding, support.max() + padding, 500)
        for color, (_, _, name), values in zip(PALETTE, populations, samples):
            if np.ptp(values) == 0:
                axis.axvline(values[0], color=color, label=name)
                continue
            axis.plot(x, gaussian_kde(values)(x), color=color, label=name, lw=2)
        axis.axvline(group.threshold.iloc[0], linestyle="--", color="#222222", label="Source threshold")
        axis.set(title=f"Section {section}", xlabel="Clip reconstruction error (q90)")
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Score distributions under domain shift | ConvAE ensemble")
    save(figure, output, "fig1_score_shift_kde.png")


def calibration_plot(table, output):
    table = table[table.model == "M2"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    methods = table.method.unique()
    targets = table[table.kind == "target"]
    simulation = table[table.kind == "simulation"]
    sections = sorted(targets.section.unique())
    positions = np.arange(len(methods))
    for section, color, offset in zip(sections, PALETTE, [-0.24, 0, 0.24]):
        rows = targets[targets.section == section].set_index("method").loc[methods]
        left.bar(positions + offset, rows.fpr, width=0.22, color=color, label=f"Section {section}")
        samples = [simulation.loc[(simulation.section == section) & (simulation.method == method),
                                  "fpr"] for method in methods]
        boxes = right.boxplot(samples, positions=positions + offset, widths=0.2,
                              patch_artist=True, manage_ticks=False, showfliers=False)
        for box in boxes["boxes"]:
            box.set(facecolor=color, alpha=0.65)
    for axis in (left, right):
        axis.set_xticks(positions, methods, rotation=20, ha="right")
        axis.axhline(0.05, linestyle="--", color="#444444", linewidth=1)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        axis.set_xlim(-0.55, len(methods) - 0.45)
    left.set(title="Target test", ylabel="False-positive rate")
    right.set(title="Source calibration resampling", ylabel="False-positive rate")
    left.legend(frameon=False)
    figure.suptitle("Three-shot threshold recalibration | ConvAE ensemble")
    save(figure, output, "fig2_recalibration_and_variance.png")


def cost_plot(table, config, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for col, (fn_cost, data) in enumerate(table.groupby("fn_cost", sort=True)):
        for color, (section, group) in zip(PALETTE, data.groupby("section", sort=True)):
            baseline = group[group.policy == "source_fixed"].iloc[0]
            planned = group[group.policy == "planned"].sort_values("budget")
            for row, metric in enumerate(("cost", "recall")):
                axis = axes[row, col]
                axis.plot(planned.review_rate, planned[metric], marker="o", color=color,
                          label=f"Section {section}")
                axis.scatter(0, baseline[metric], color=color, marker="D", s=28)
                exceeded = planned[planned.budget_exceeded == True]
                axis.scatter(exceeded.review_rate, exceeded[metric], marker="o", s=70,
                             facecolors="none", edgecolors=color)
                axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
                if metric == "recall":
                    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            axes[0, col].set_title(f"Miss to false alarm cost {fn_cost:g}:1")
        review_rates = np.array([0, data.review_rate.max()])
        _, _, reference = trivial_random_cost(review_rates, config, fn_cost)
        axes[0, col].plot(review_rates, reference, color="#666666", linestyle="--",
                          label="Best trivial + random review")
        axes[1, col].set_xlabel("Target review rate at assumed prevalence")
    axes[0, 0].set_ylabel("Expected cost per clip")
    axes[1, 0].set_ylabel("Effective fault recall")
    axes[0, 1].legend(frameon=False)
    figure.suptitle("Selective review on frozen target clips | Rings mark budget overruns")
    save(figure, output, "fig3_selective_review_frontier.png")


def failure_plot(ledger, table, config, archive_path, output):
    figure = plt.figure(figsize=(12, 8), layout="constrained")
    cases = table[table.kind == "case"].set_index("case")
    frontend = Frontend(config["audio"])
    order = ("TP", "Low-frequency FP candidate", "Transient TP (q90 only)", "High-energy FN candidate")
    panels = figure.add_gridspec(2, 3, width_ratios=[1, 1, 0.035])
    powers = {}
    with ZipFile(archive_path) as archive:
        for title, case in cases.iterrows():
            if case.status != "Selected":
                continue
            with archive.open(case.member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != case.sha256:
                raise ValueError(f"Case WAV fails ledger SHA-256: {case.clip_id}")
            audio, _ = read_audio(archive, case.member, config["audio"])
            powers[title] = 10 * np.log10(np.maximum(frontend.power(audio).T, 1e-12))
    limit = (np.percentile(np.concatenate([power.ravel() for power in powers.values()]), 99)
             if powers else 0)
    image = None
    for index, title in enumerate(order):
        case = cases.loc[title]
        panel = panels[index // 2, index % 2].subgridspec(2, 1, height_ratios=[2, 1])
        spectrum = figure.add_subplot(panel[0])
        if case.status != "Selected":
            spectrum.text(0.5, 0.5, case.status, ha="center", va="center")
            spectrum.set(title=title, xticks=[], yticks=[])
            continue
        image = spectrum.imshow(powers[title], origin="lower", aspect="auto", interpolation="nearest",
                                extent=(0, config["audio"]["clip_seconds"], 0, frontend.sr / 2000),
                                cmap="magma", vmin=limit - 65, vmax=limit)
        spectrum.set(title=f"{title} | {case.model}, section {case.section}", ylabel="Frequency (kHz)")
        spectrum.tick_params(labelbottom=False)
        trajectory = figure.add_subplot(panel[1], sharex=spectrum)
        row = ledger[(ledger.clip_id == case.clip_id) & (ledger.model == case.model)
                     & (ledger.seed == case.seed)].iloc[0]
        centers = (np.arange(len(WINDOWS)) * frontend.step + frontend.window / 2) / frontend.sr
        trajectory.plot(centers, row[WINDOWS].astype(float), marker="o", ms=3, color=PALETTE[0])
        trajectory.axhline(case.threshold, color=PALETTE[1], linestyle="--", lw=1)
        trajectory.set(xlabel="Time (s)", ylabel="Window score",
                       xlim=(0, config["audio"]["clip_seconds"]))
    if image is not None:
        colorbar = figure.add_subplot(panels[:, 2])
        figure.colorbar(image, cax=colorbar, label="STFT power (dB, common reference)")
    figure.suptitle("Frozen target cases | Dashed line marks the clip q90 threshold")
    save(figure, output, "fig4_failure_case_spectrograms.png")


def create_figures(ledger, reports, config, archive, output):
    output.mkdir(exist_ok=True, parents=True)
    primary = ledger[(ledger.model == "M2") & (ledger.seed == "ensemble")]
    score_shift(primary, output)
    calibration_plot(reports[REPORTS[1]], output)
    cost_plot(reports[REPORTS[2]], config["evaluation"], output)
    failure_plot(ledger, reports[REPORTS[3]], config, archive, output)
