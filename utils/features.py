"""
Optional helper: same MFE constants as colab_train_flush.ipynb and mfe.cpp.

Not used by Colab or by the board. Run generate_tables.py after changing
hop / FFT / mels so arduino/FlushFan/mfe_tables.h stays in sync, then retrain.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
WINDOW_S = 8.0
FRAME_S = 0.032  # hop
N_FFT = 1024  # 64 ms analysis window at 16 kHz
HOP = 512  # 32 ms, 50% overlap
N_MELS = 32
FMIN = 0.0
FMAX = 8000.0
N_FRAMES = int(WINDOW_S / FRAME_S)  # 250
SAMPLES_PER_CLIP = int(WINDOW_S * SAMPLE_RATE)  # 128_000 (8 s of wav)
STFT_SAMPLES = (N_FRAMES - 1) * HOP + N_FFT  # 128_512 (last hop zero-padded)
EPS = 1e-6
N_FFT_BINS = N_FFT // 2 + 1  # 513


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def hann_window(n: int = N_FFT) -> np.ndarray:
    """Periodic Hann, dumped identically into mfe_tables.h."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)).astype(
        np.float32
    )


def mel_filterbank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sr: int = SAMPLE_RATE,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Slaney-style triangular filterbank, n_mels x n_fft_bins, rows sum to 1."""
    n_bins = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center = min(left + 1, n_bins - 1)
        if right == center:
            right = min(center + 1, n_bins - 1)
        for j in range(left, center):
            fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fb[i, j] = (right - j) / (right - center)
        s = fb[i].sum()
        if s > 0:
            fb[i] /= s
    return fb


WINDOW = hann_window()
FILTERBANK = mel_filterbank()


def mfe_frames(pcm: np.ndarray) -> np.ndarray:
    """pcm: float32 mono at 16 kHz. Returns (n_frames, 32) log-mel, overlapping hops."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n_frames = 1 + (len(x) - N_FFT) // HOP
    windows = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:n_frames]
    frames = windows * WINDOW
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    mel = power @ FILTERBANK.T
    return np.log(mel + EPS).astype(np.float32)


def mfe_clip(pcm: np.ndarray) -> np.ndarray:
    """Fixed 250 x 32 spectrogram. Crops/pads wav to 8 s, then 512 zeros for last hop."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if len(x) < SAMPLES_PER_CLIP:
        x = np.pad(x, (0, SAMPLES_PER_CLIP - len(x)))
    else:
        x = x[:SAMPLES_PER_CLIP]
    if len(x) < STFT_SAMPLES:
        x = np.pad(x, (0, STFT_SAMPLES - len(x)))
    spec = mfe_frames(x)
    if spec.shape[0] < N_FRAMES:
        spec = np.pad(spec, ((0, N_FRAMES - spec.shape[0]), (0, 0)))
    return spec[:N_FRAMES]


def sparse_filterbank(fb: np.ndarray = FILTERBANK) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row start index, length, and packed weights for the Arduino kernel."""
    starts, lengths, weights, weight_offsets = [], [], [], [0]
    for row in fb:
        nz = np.flatnonzero(row > 0)
        if nz.size == 0:
            starts.append(0)
            lengths.append(0)
        else:
            start, end = int(nz[0]), int(nz[-1]) + 1
            starts.append(start)
            lengths.append(end - start)
            weights.extend(row[start:end].tolist())
        weight_offsets.append(len(weights))
    return (
        np.array(starts, dtype=np.int16),
        np.array(lengths, dtype=np.int16),
        np.array(weights, dtype=np.float32),
        np.array(weight_offsets[:-1], dtype=np.int16),
    )
