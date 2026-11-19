"""Score the verified MIMII DUE archive or rebuild reports from saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import ROOT, archive_path, config_file, read_csv, score_models, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--prepare", action="store_true", help="Verify clips and cache features")
    phases.add_argument("--score-only", action="store_true", help="Save the clip ledger before reporting")
    phases.add_argument("--recompute", action="store_true", help="Use only the existing clip ledger")
    phases.add_argument("--verify-csv", action="store_true", help="Compare ledger-only recalculation")
    parser.add_argument("--machine", choices=("valve", "fan"), default="valve")
    parser.add_argument("--cpu", action="store_true", help="Train on CPU")
    args = parser.parse_args()
    results = ROOT / "results" / args.machine
    results.mkdir(exist_ok=True, parents=True)
    predictions = results / "predictions_clip_level.csv"
    if args.prepare:
        from src.pipeline import prepare

        prepare(config_file(args.machine))
        return
    if not predictions.is_file():
        if args.machine == "valve" or args.recompute or args.verify_csv:
            parser.error(f"Missing preserved prediction ledger: {predictions}")
        from src.models import check_device
        import torch

        config = config_file(args.machine)
        device = check_device(args.cpu)
        hardware = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
        print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, device {hardware}", flush=True)
        ledger = score_models(config, device)
        with archive_path(config).open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        ledger["analysis_config"] = ""
        ledger.at[0, "analysis_config"] = json.dumps(
            {**config, "archive_sha256": archive_hash, "torch": torch.__version__, "device": hardware},
            sort_keys=True)
        write_csv(ledger, predictions)
    if args.score_only:
        return

    from src.reports import build_reports

    ledger = read_csv(predictions)
    reports = build_reports(ledger)
    if args.verify_csv:
        from tempfile import TemporaryDirectory
        from pandas.testing import assert_frame_equal

        with TemporaryDirectory() as tmp:
            for name, frame in reports.items():
                path = Path(tmp) / f"{name}.csv"
                write_csv(frame, path)
                assert_frame_equal(read_csv(results / path.name), read_csv(path),
                                   check_exact=False, rtol=0, atol=1e-7)
                print(f"PASS: {path.name}, absolute tolerance 1e-7")
        return

    for name, frame in reports.items():
        write_csv(frame, results / f"{name}.csv")
    if args.recompute:
        print("Finished rebuilding P1 summary CSVs from the ledger")
        return

    from src.plotting import create_figures

    config = json.loads(ledger.loc[ledger.analysis_config != "", "analysis_config"].iloc[0])
    create_figures(ledger, reports, config, archive=archive_path(config), output=ROOT / "figs" / args.machine)
    print("Finished writing P1 results and figures", flush=True)


if __name__ == "__main__":
    main()
