"""Four physical features, shared log-Mel frontend and MFCC with deltas."""
import numpy as np
from scipy.fft import dct, rfft, rfftfreq
from scipy.signal import get_window
from numpy.lib.stride_tricks import sliding_window_view

BAND_NAMES = ("band_lt500_db", "band_500_2000_db", "band_2000_4000_db", "band_ge4000_db")


class Frontend:
    def __init__(self, config):
        self.sr = config["sr"]
        self.window = int(config["window_seconds"] * self.sr)
        self.step = int(config["step_seconds"] * self.sr)
        self.hop = config["hop_length"]
        self.fft = config["n_fft"]
        self.length = config["win_length"]
        self.hann = get_window("hann", self.length).astype(np.float32)
        self.frequencies = rfftfreq(self.fft, 1 / self.sr)
        mel_edges = np.linspace(*(2595 * np.log10(1 + np.array(
            [config["fmin"], config["fmax"]]) / 700)), config["n_mels"] + 2)
        hz = 700 * np.expm1(mel_edges * np.log(10) / 2595)
        rising = (self.frequencies - hz[:-2, None]) / np.diff(hz)[:-1, None]
        falling = (hz[2:, None] - self.frequencies) / np.diff(hz)[1:, None]
        self.filters = np.maximum(0, np.minimum(rising, falling)).T.astype(np.float32)
        self.bands = []
        edges = config["band_edges"]
        for index, (low, high) in enumerate(zip(edges, edges[1:])):
            upper = self.frequencies <= high if index == len(edges) - 2 else self.frequencies < high
            self.bands.append((self.frequencies >= low) & upper)

    def power(self, x):
        pad = [(0, 0)] * (x.ndim - 1) + [(self.length // 2, self.length // 2 - 1)]
        frames = sliding_window_view(np.pad(x, pad, mode="reflect"), self.length, axis=-1)
        return abs(rfft(frames[..., ::self.hop, :] * self.hann, n=self.fft, axis=-1)) ** 2

    def transform(self, x):
        windows = sliding_window_view(x, self.window)[::self.step]
        power = self.power(windows)
        spec = np.sqrt(power)
        denom = np.maximum(spec.sum(axis=-1), 1e-12)
        centroid = (spec @ self.frequencies) / denom
        rolloff = self.frequencies[(np.cumsum(spec, axis=-1) >= 0.85 * denom[..., None]).argmax(axis=-1)]
        rms = np.sqrt(np.mean(windows**2, axis=-1))
        zcr = np.mean(np.signbit(windows[:, 1:]) != np.signbit(windows[:, :-1]), axis=-1)
        physical = np.column_stack((rms, zcr, centroid.mean(axis=1), rolloff.mean(axis=1)))
        mel = np.log(np.maximum(power @ self.filters, 1e-10))

        def moments(values):
            return np.concatenate((values.mean(axis=1), values.std(axis=1)), axis=1)

        mfcc = dct(mel, type=2, axis=-1, norm="ortho")[..., :20]
        delta = np.gradient(mfcc, axis=1)
        # Use a clip-wide STFT for diagnostic band energies without duplicated overlap.
        whole_clip = self.power(x).mean(axis=0)
        energy = [10 * np.log10(max(float(whole_clip[band].sum()), 1e-12)) for band in self.bands]
        return (physical, moments(mel), moments(np.concatenate((mfcc, delta), axis=-1)),
                np.ascontiguousarray(mel.transpose(0, 2, 1)), energy)
