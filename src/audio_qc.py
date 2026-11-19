"""Decode the original PCM signal and record clip-level quality measurements."""
import io
import wave

import numpy as np


def read_audio(archive, member, config):
    with wave.open(io.BytesIO(archive.read(member))) as stream:
        sr = stream.getframerate()
        frames = stream.getnframes()
        if (stream.getnchannels(), stream.getsampwidth(), sr, frames) != (
            1, 2, config["sr"], config["sr"] * config["clip_seconds"]
        ):
            raise ValueError(f"Unexpected WAV geometry in {member}: {stream.getparams()}")
        raw = stream.readframes(frames)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if len(audio) != frames:
        raise ValueError(f"Truncated WAV: {member}")
    qc = dict(sample_rate=sr, duration_seconds=frames / sr, silence_fraction=float(
        np.mean(np.abs(audio) <= 1 / 32768)), silent=bool(not np.any(audio)),
        clipping_fraction=float(np.mean(np.abs(audio) > 0.999)), dc_offset=float(audio.mean()))
    return audio, qc
