# Industrial Acoustic Anomaly Detection under Operating-Condition Shift

This study compares threshold transfer, three-shot recalibration and cost-sensitive selective review for intermittent valve sounds and continuous fan sounds from the official [MIMII DUE v1.01 development set](https://zenodo.org/records/4740355). The machine-level contrast is descriptive because fault and operating conditions also vary.

## Protocol

Sections 00, 01 and 02 are modeled independently. Source-normal clips are split 80/20 by recording before segmentation, with clip IDs and SHA-256 checked for overlap. The three supplied target-normal clips in each section support three-shot calibration. The official source and target tests remain frozen. Each 10-second clip yields 19 overlapping one-second windows.

M0 uses robust physical features. M1-Mel and M1-MFCC use Ledoit-Wolf shrinkage Mahalanobis scores. M2 uses a convolutional autoencoder with three seeds and an ensemble. Model fitting uses source training. Early stopping and operating thresholds use source calibration. The primary score is the 90th percentile of window scores, with mean pooling as an ablation.

## Results

| Analysis | Valve | Fan | Paired figure |
| --- | --- | --- | --- |
| Ranking and paired tests | [CSV](results/valve/p1_ranking_and_significance.csv) | [CSV](results/fan/p1_ranking_and_significance.csv) | [Figure 1](figs/comparison/fig1_score_shift_kde.png) |
| Three-shot recalibration | [CSV](results/valve/p1_fewshot_recalibration.csv) | [CSV](results/fan/p1_fewshot_recalibration.csv) | [Figure 2](figs/comparison/fig2_recalibration_and_variance.png) |
| Selective review and costs | [CSV](results/valve/p1_selective_review_eddi_cost.csv) | [CSV](results/fan/p1_selective_review_eddi_cost.csv) | [Figure 3](figs/comparison/fig3_selective_review_frontier.png) |
| Error cases and frequency-band rules | [CSV](results/valve/p1_error_audit_and_band_rules.csv) | [CSV](results/fan/p1_error_audit_and_band_rules.csv) | [Figure 4](figs/comparison/fig4_failure_case_spectrograms.png) |

The original normalized partial AUC (`pauc`) divides the ROC area below FPR 0.1 by 0.1 and has a random-ranking reference of 0.05. `pauc_mcclish` uses scikit-learn's standardized partial AUC with `max_fpr=0.1` and a random reference of 0.50. Paired bootstrap intervals and Holm-adjusted exact McNemar tests compare the M2 ensemble with M1-Mel by section and domain.

The selective-review planner fits thresholds on source normals under a declared anomaly-rank scenario and assumes perfect review. The strengthened reference matches the policy's achieved review rate `q` and costs `(1 - q) * min(always_pass, always_alert) + q * review_cost`. Budget overruns are recorded separately. Frequency-band CART rules are fitted on source data and audited on target normals. Case labels describe spectral evidence without establishing the physical cause of an individual error.

## Reproduce

The measured run used Python 3.13.16 and PyTorch 2.11.0 with CUDA 12.8 on a GTX 1650 Ti. The published [valve ledger](results/valve/predictions_clip_level.csv) and [fan ledger](results/fan/predictions_clip_level.csv) support offline CSV verification. Official WAVs are also required to regenerate the case spectrograms.

```powershell
uv venv --python 3.13.16
uv pip install -r requirements.txt
.venv/Scripts/python.exe scripts/download_mimii_valve.py --machine valve --workers 16
.venv/Scripts/python.exe scripts/download_mimii_valve.py --machine fan --workers 24
.venv/Scripts/python.exe -m pytest -v tests/test_p1_integrity.py
.venv/Scripts/python.exe scripts/run_all_experiments.py --machine valve
.venv/Scripts/python.exe scripts/run_all_experiments.py --machine fan
.venv/Scripts/python.exe scripts/run_all_experiments.py --machine valve --verify-csv
.venv/Scripts/python.exe scripts/run_all_experiments.py --machine fan --verify-csv
.venv/Scripts/python.exe scripts/compare_figures.py
```

The Valve command uses the preserved ledger. The Fan command reuses its ledger when available and otherwise fits from the verified archive. Verification recomputes each summary with an absolute tolerance of `1e-7`.
