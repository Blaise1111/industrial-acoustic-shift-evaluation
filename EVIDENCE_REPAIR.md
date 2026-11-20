# Acoustic evidence after the evaluation repair

This is a post-hoc offline analysis of the frozen Fan and Valve M2 ensemble ledgers at public baseline baf1a8f. MIMII DUE v1.01 provides the source and target recordings. The original run fitted the score models on source data, selected cutoffs without target test labels, and stored clip identities and window scores. The new analysis holds those scores and the source-only plans fixed.

## What was repaired

- The [scenario table](audit_outputs/acoustic/review_scenarios.csv) records the source planning rate, prevalence-weighted target rate and raw target sample rate separately. It includes both target budget flags and all six normal/anomaly × routing counts. Here model recall means the fraction of anomalies automatically alerted. System recall also credits anomalies correctly identified in review.
- Scenario parameters come from [the audit configuration](configs/evidence_reaudit.yaml). The random comparator sends the same expected workload at the assumed prevalence to review, with identical review quality and cost. Its unreviewed cases use whichever of always-pass and always-alert has lower expected cost. Review sensitivity and specificity are assumed inputs. All cost differences use random cost minus policy cost, so negative values are unfavorable.
- The [threshold table](audit_outputs/acoustic/threshold_effects.csv) reports per-section operating rates and paired differences from the frozen source threshold. The [macro table](audit_outputs/acoustic/threshold_macro.csv) averages sections equally. Bootstrap draws preserve pairing within each machine, section and class. The pointwise intervals describe target test-sample uncertainty conditional on the fitted cutoffs and three selected target clips. They exclude uncertainty from drawing other target calibration clips.
- [Held-out proxy trials](audit_outputs/acoustic/source_proxy_trials.csv) partition source calibration clips into reference fitting, three proxy clips, and held-out FPR evaluation, with identical membership across methods per repeat. [Membership](audit_outputs/acoustic/source_proxy_partitions.jsonl) is saved. These splits isolate cutoff estimation conditional on the stored score model. The model had already used source calibration for checkpoint selection. The [legacy diagnostic](audit_outputs/acoustic/source_proxy_legacy_fit_in.csv) evaluates on the cutoff fitting subset and is labeled fit-in.
- Window q95 is a window-score heuristic applied to q90 clip scores. The [pAUC table](audit_outputs/acoustic/pauc_definitions.csv) separates area divided by the FPR cap, with random reference equal to half the cap, from the McClish-standardized result, with random reference 0.50.

## What the saved evidence shows

In Fan section 00 at the 20% budget and historical 10:1, 5%-prevalence, perfect-review setting, the source plan used 19.98% review. The same frozen policy routed 18.00% under prevalence weighting and 27.00% in the raw target sample. It met the first budget interpretation and exceeded the second. Automatic anomaly recall was 19.00%, with 37.00% of anomalies routed to review and 56.00% final recall under perfect review. In the post-hoc scenario analysis, changing only the assumed anomaly prevalence from 5% to 1% moved savings against matched random review from +0.1235 to −0.0305 per case for this same frozen policy.

Fan mean shift lowered macro FPR from 28.00% to 2.67%, with a paired difference of −25.33 percentage points (95% interval −29.67 to −20.67). Recall fell from 32.33% to 6.33%. Both outcomes matter for threshold selection. The source proxy exercise is a variability diagnostic. It cannot certify the tail of the target score distribution.

In the [section-level scenario table](audit_outputs/acoustic/review_scenarios.csv), 124 of 810 Fan rows and 129 of 810 Valve rows beat the matched random comparator. Each machine contributes 810 section × scenario rows. Many rows share recordings and policies. The [cost and budget plot](audit_outputs/acoustic/cost_budget_sensitivity.png) shows favorable and unfavorable cells at the 20% budget, including evaluation at ratios different from those used in planning. Thresholds were never chosen on the target step.

## Reproduce and verify

The open benchmark citation and asset-download commands are in [the README](README.md). An existing checkout with the two published ledgers and the locked environment is sufficient for the repair. Raw WAVs are required for the historical spectrum/split integration tests and original audio rendering. Model checkpoints are local to the historical training run and are unnecessary for the audit.

~~~powershell
.venv/Scripts/python.exe scripts/run_all_experiments.py --machine fan --verify-csv
.venv/Scripts/python.exe scripts/run_all_experiments.py --machine valve --verify-csv
.venv/Scripts/python.exe scripts/reaudit_ledger.py --output audit_outputs/acoustic
.venv/Scripts/python.exe -m pytest -q
~~~

The [baseline inventory](audit_outputs/baseline/inventory.json) and [legacy command status](audit_outputs/baseline/commands.json) identify the inputs, environment and checks. The [new analysis manifest](audit_outputs/acoustic/analysis_manifest.json) binds the ledgers to the scenario configuration. The historic summaries still pass offline verification at absolute tolerance 1e-7. The perfect-review overlap of the new grid is tested against historical policy metrics by the new runner.

Local command and test outcomes are listed in the [validation record](audit_outputs/validation_commands.json).

The [prepublication validation record](audit_outputs/prepublish_validation.json) records the follow-up commands, their results and remaining verification limits.

## Status

- **Repaired** · Operating definitions, comparator, budget semantics, source proxy separation and conditional target bootstrap.
- **New measurements** · This phase used saved clip scores.
- **Offline recomputation** · Two machines, six sections, 1,620 frozen-policy scenario rows and 24 target-threshold comparisons.
- **Unverified** · Real human review accuracy and external industrial prevalence. Source assumptions and benchmark test rates cannot establish either.

## Application evidence copy for review

I audited how a source-calibrated acoustic detector transfers across Fan and Valve sections and evaluated a selective-review policy against a matched randomized baseline. I recomputed costs and separated automatic detection, review workload and final system recall using the frozen public clip ledgers. In Fan section 00 the same policy implied 18% review at the stated prevalence but routed 27% of the test sample, so a 20% budget decision depended on its denominator. The result is post-hoc benchmark evidence under assumed review accuracy. The definitions and inputs are recorded in [review_scenarios.csv](audit_outputs/acoustic/review_scenarios.csv). The publication link will be updated after this local branch is reviewed.
