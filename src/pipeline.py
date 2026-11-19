"""Prepare the verified clip ledger and score every clip with each frozen model."""
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from .audio_qc import read_audio
from .data_manifest import archive_md5, assert_disjoint, build_manifest
from .features import BAND_NAMES, Frontend

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [f"window_{number:02d}" for number in range(19)]
TEXT_COLUMNS = {"clip_id": str, "sha256": str, "section": str, "seed": str}


def config_file(machine="valve"):
    return yaml.safe_load((ROOT / "configs" / f"{machine}_experiment.yaml").read_text(encoding="utf-8"))


def archive_path(config):
    filename = config["dataset"]["url"].partition("?")[0].rsplit("/", 1)[-1]
    return ROOT / "data" / filename


def write_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
    temporary.replace(path)


def read_csv(path):
    return pd.read_csv(path, dtype=TEXT_COLUMNS, float_precision="round_trip", keep_default_na=False)


def fingerprint(config, modules):
    digest = hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode())
    for module in modules:
        digest.update((ROOT / "src" / module).read_bytes())
    return digest.hexdigest()[:24]


def prepare(config):
    path = archive_path(config)
    if not path.is_file() or archive_md5(path) != config["dataset"]["md5"]:
        raise ValueError("Run scripts/download_mimii_valve.py to obtain the verified archive")
    key = fingerprint((config["dataset"], config["audio"]),
                      ("data_manifest.py", "audio_qc.py", "features.py"))
    directory = ROOT / ".work" / key
    directory.mkdir(parents=True, exist_ok=True)
    csv = directory / "manifest.csv"
    if csv.is_file():
        manifest = read_csv(csv)
        assert_disjoint(manifest)
        return directory, manifest
    with ZipFile(path) as archive, threadpool_limits(limits=1):
        manifest = build_manifest(archive, config["dataset"])
        frontend = Frontend(config["audio"])
        output_shapes = {"physical": (19, 4), "mel": (19, 128), "mfcc": (19, 80),
                         "images": (19, config["audio"]["n_mels"], 100)}
        qc_by_section = []
        for section, group in manifest.groupby("section", sort=True):
            print(f"Extracting features from {len(group)} verified clips in section {section}", flush=True)
            buffers = {name: np.lib.format.open_memmap(directory / f"{section}_{name}.npy",
                        mode="w+", dtype=np.float32, shape=(len(group), *shape))
                       for name, shape in output_shapes.items()}
            for index, (row_id, row) in enumerate(group.iterrows()):
                audio, qc = read_audio(archive, row.member, config["audio"])
                *features, bands = frontend.transform(audio)
                for array, values in zip(buffers.values(), features):
                    array[index] = values
                qc_by_section.append(dict(row_id=row_id, **qc, **dict(zip(BAND_NAMES, bands))))
            for array in buffers.values():
                array.flush()
        manifest = manifest.join(pd.DataFrame(qc_by_section).set_index("row_id"))
    write_csv(manifest, csv)
    return directory, manifest


def score_models(config, device):
    from .models import fit_autoencoder, mahalanobis_scores, physical_scores

    directory, manifest = prepare(config)
    key = fingerprint(config, ("models.py", "pipeline.py", "stats_and_calib.py"))
    output = []
    model_config = config["model"]
    for section, group in manifest.groupby("section", sort=True):
        print(f"Scoring section {section} on {device}", flush=True)
        train = group.split.eq("source_train").to_numpy()
        calib = group.split.eq("source_calib").to_numpy()
        train_ids = np.flatnonzero(np.repeat(train, len(WINDOWS)))
        calib_ids = np.flatnonzero(np.repeat(calib, len(WINDOWS)))
        with threadpool_limits(limits=model_config["cpu_threads"]):
            for model, feature in (("M0", "physical"), ("M1-Mel", "mel"), ("M1-MFCC", "mfcc")):
                values = np.load(directory / f"{section}_{feature}.npy", mmap_mode="r")
                flat = values.reshape(-1, values.shape[-1])
                scorer = physical_scores if model == "M0" else mahalanobis_scores
                scores = scorer(flat[train_ids], flat)
                output.append(prediction_rows(group, model, "n/a", scores, config))
            images = np.load(directory / f"{section}_images.npy", mmap_mode="r")
            images = images.reshape(-1, *images.shape[-2:])
            seed_scores = []
            for seed in model_config["seeds"]:
                checkpoint = ROOT / ".checkpoints" / key / device.type / f"section_{section}_seed_{seed}.pt"
                scores, history = fit_autoencoder(images, train_ids, calib_ids, seed, model_config,
                                                  device, checkpoint)
                seed_scores.append(scores)
                rows = prediction_rows(group, "M2", str(seed), scores, config)
                rows["epochs_observed"] = len(history)
                rows["selected_epoch"] = int(np.argmin(history)) + 1
                output.append(rows)
            output.append(prediction_rows(group, "M2", "ensemble", np.mean(seed_scores, axis=0), config))
    combined = pd.concat(output, ignore_index=True)
    return combined


def prediction_rows(group, model, seed, scores, config):
    from .stats_and_calib import fixed_threshold

    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite scores in {model}/{seed}")
    scores = scores.reshape(len(group), len(WINDOWS))
    evaluation = config["evaluation"]
    clip = np.quantile(scores, evaluation["pooling_quantile"], axis=1)
    mean = scores.astype(np.float64).mean(axis=1)
    calib = group.split.eq("source_calib").to_numpy()
    tau = fixed_threshold(clip[calib], evaluation["target_fpr"])
    tau_mean = fixed_threshold(mean[calib], evaluation["target_fpr"])
    rows = group.reset_index(drop=True).assign(
        model=model, seed=seed, score=clip, threshold=tau, alert=clip > tau,
        mean_score=mean, mean_threshold=tau_mean, mean_alert=mean > tau_mean)
    return rows.join(pd.DataFrame(scores, columns=WINDOWS))
