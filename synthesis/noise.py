"""Diffuse noise with the spatial coherence of an isotropic field, mixed at a given SNR."""
from __future__ import annotations
import os, math, random
import numpy as np
import scipy.signal
import scipy.linalg
import soundfile as sf

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOISE_DIR = os.environ.get('NOISE_DIR',
                            os.path.join(_PKG_ROOT, 'data', 'NoiSig', 'noise92'))
C_SOUND = 343.0


def list_noise_files(noise_dir=NOISE_DIR):
    return sorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
                    if f.endswith(('.wav', '.flac', '.ogg'))])


def gen_diffuse_noise(noise_mono: np.ndarray, T_samples: int, mic_pos: np.ndarray,
                       fs: int = 16000, nfft: int = 256, c: float = C_SOUND
                       ) -> np.ndarray:
    M = mic_pos.shape[0]
    L = T_samples
    if len(noise_mono) < L * M:
        # repeat
        reps = (L * M + len(noise_mono) - 1) // len(noise_mono) + 1
        noise_mono = np.tile(noise_mono, reps)
    noise_M = np.zeros((L, M))
    for m in range(M):
        noise_M[:, m] = noise_mono[m*L:(m+1)*L] - np.mean(noise_mono[m*L:(m+1)*L])

    ww = 2 * math.pi * fs * np.arange(nfft // 2 + 1) / nfft
    DC = np.zeros((M, M, nfft // 2 + 1))
    for p in range(M):
        for q in range(M):
            if p == q:
                DC[p, q, :] = 1.0
            else:
                d = np.linalg.norm(mic_pos[p] - mic_pos[q])
                DC[p, q, :] = np.sinc(ww * d / (c * math.pi))

    # STFT
    noise_pad = np.vstack([np.zeros((nfft // 2, M)), noise_M, np.zeros((nfft // 2, M))])
    f, t, N = scipy.signal.stft(noise_pad.T, window='hann', nperseg=nfft,
                                  noverlap=int(0.75 * nfft), nfft=nfft)
    X = np.zeros(N.shape, dtype=complex)
    for k in range(1, nfft // 2 + 1):
        DC_k = DC[:, :, k] + 1e-9 * np.eye(M)
        try:
            Cchol = scipy.linalg.cholesky(DC_k)
        except np.linalg.LinAlgError:
            Cchol = np.linalg.cholesky(DC_k)
        X[:, k, :] = (np.squeeze(N[:, k, :]).T @ np.conj(Cchol)).T

    _, x = scipy.signal.istft(X, window='hann', nperseg=nfft,
                                noverlap=int(0.75 * nfft), nfft=nfft)
    x = x.T[nfft // 2: -nfft // 2, :]
    return x[:L, :]


def sample_noise(rng: np.random.Generator, T_samples: int, mic_pos: np.ndarray,
                  fs: int = 16000, noise_dir: str = None,
                  noise_file: str = None) -> np.ndarray:
    if noise_dir is None:
        noise_dir = NOISE_DIR
    M = mic_pos.shape[0]
    files = list_noise_files(noise_dir)
    if noise_file is None:
        path = files[rng.integers(0, len(files))]
    else:
        path = noise_file
    noise, fs_in = sf.read(path)
    if noise.ndim > 1:
        noise = noise[:, 0]
    if fs_in != fs:
        noise = scipy.signal.resample_poly(noise, up=fs, down=fs_in)
    nsample_desired = int(T_samples * M)
    while noise.shape[0] < nsample_desired:
        noise = np.concatenate([noise, noise])
    st = rng.integers(0, noise.shape[0] - nsample_desired)
    noise = noise[st: st + nsample_desired]

    return gen_diffuse_noise(noise, T_samples, mic_pos, fs=fs)


def acoustic_power(s: np.ndarray) -> float:
    w, o = 512, 256
    s = np.ascontiguousarray(s)
    if len(s) < w:
        return float(np.mean(s ** 2) + 1e-12)
    sh = (s.size - w + 1, w)
    st = s.strides * 2
    S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]
    window_power = np.mean(S ** 2, axis=-1)
    th = 0.01 * window_power.max()
    active = window_power[window_power > th]
    return float(np.mean(active)) if len(active) > 0 else 1e-12


def add_noise_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float,
                      direct_sig: np.ndarray = None) -> np.ndarray:
    sig_for_snr = direct_sig if direct_sig is not None else x
    nch = sig_for_snr.shape[1]
    sig_pow = np.mean([acoustic_power(sig_for_snr[:, c]) for c in range(nch)])
    noise_pow = np.mean([acoustic_power(noise[:, c]) for c in range(nch)])
    scale = np.sqrt(sig_pow / 10**(snr_db/10) / (noise_pow + 1e-12))
    return (x + scale * noise).astype(np.float32)
