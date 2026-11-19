"""Verify the official archive and freeze per-section clip assignments."""
import hashlib
import re
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

NAME = re.compile(
    r"section_(\d{2})_(source|target)_(train|test)_(normal|anomaly)_(\d+).*\.wav",
    re.IGNORECASE,
)


def archive_md5(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def assert_disjoint(manifest):
    for key in ("clip_id", "sha256"):
        if manifest[key].duplicated().any():
            repeats = manifest.loc[manifest[key].duplicated(False), ["clip_id", "split"]]
            raise ValueError(f"Duplicate {key}:\n{repeats.to_string(index=False)}")


def build_manifest(archive, config):
    rows = []
    for member in sorted(archive.namelist()):
        path = PurePosixPath(member)
        if path.suffix.lower() != ".wav":
            continue
        match = NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized WAV name: {member}")
        section, domain, official, condition, _ = match.groups()
        if section not in config["sections"]:
            continue
        if official == "train" and condition != "normal":
            raise ValueError(f"Unexpected abnormal training clip: {member}")
        split = f"{domain}_test" if official == "test" else (
            "source_train" if domain == "source" else "target_fewshot"
        )
        with archive.open(member) as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(dict(clip_id=path.stem, member=member, sha256=digest,
                         section=section, domain=domain, split=split,
                         label=int(condition == "anomaly")))
    manifest = pd.DataFrame(rows)
    rng = np.random.default_rng(config["seed"])
    for section in config["sections"]:
        mask = manifest.section.eq(section)
        indices = manifest.index[mask & manifest.split.eq("source_train")]
        order = rng.permutation(indices)
        cut = int(len(order) * config["train_fraction"])
        if not 0 < cut < len(order):
            raise ValueError(f"Missing calibration or training clips in section {section}")
        manifest.loc[order[cut:], "split"] = "source_calib"
        if sum(mask & manifest.split.eq("target_fewshot")) != 3:
            raise ValueError(f"Section {section} does not contain exactly three target training clips")
        for split in ("source_test", "target_test"):
            if set(manifest.loc[mask & manifest.split.eq(split), "label"]) != {0, 1}:
                raise ValueError(f"Missing test class in {section}/{split}")
    assert_disjoint(manifest)
    return manifest
