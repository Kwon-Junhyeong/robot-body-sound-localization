"""Room impulse responses with the body: every image source gets the BEM response of its arrival direction."""
from __future__ import annotations
import os, sys, time
import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from bem_table import BEMTableLazy

C_SOUND = 343.0


def build_rir_per_image(images_pos, damping, body_center, panels, cached_bem,
                          bem_meta, fs=16000, n_fft_bem=512, c_sound=C_SOUND,
                          rir_len=None):
    r = images_pos.T - body_center[None, :]
    R = np.linalg.norm(r, axis=-1)
    valid = (R > 1e-3) & (np.abs(damping) > 1e-9)
    r = r[valid]; damping = damping[valid]; R = R[valid]
    if R.size == 0:
        return np.zeros((len(panels), rir_len or n_fft_bem), dtype=np.float32)

    u = r / R[:, None]
    amp = damping / np.maximum(R, 0.05)
    tau = R / c_sound
    delay_samples = np.round(tau * fs).astype(np.int64)

    az = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    az = np.where(az < 0, az + 360, az)
    el = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))

    n_az = bem_meta['n_az']
    n_el = bem_meta['n_el']
    az_step = bem_meta['az_step']
    el_step = bem_meta['el_step']

    az_f = az / az_step
    el_f = (el + 90.0) / el_step
    el_f = np.clip(el_f, 0.0, n_el - 1.0)

    az_lo = (np.floor(az_f).astype(np.int64)) % n_az
    az_hi = (az_lo + 1) % n_az
    el_lo = np.clip(np.floor(el_f).astype(np.int64), 0, n_el - 1)
    el_hi = np.clip(el_lo + 1, 0, n_el - 1)

    az_w = az_f - np.floor(az_f)
    el_w = el_f - np.floor(el_f)
    el_w = np.where(el_lo == el_hi, 0.0, el_w)

    q00 = az_lo * n_el + el_lo
    q01 = az_lo * n_el + el_hi
    q10 = az_hi * n_el + el_lo
    q11 = az_hi * n_el + el_hi

    w00 = (1 - az_w) * (1 - el_w)
    w01 = (1 - az_w) * el_w
    w10 = az_w * (1 - el_w)
    w11 = az_w * el_w

    pre = n_fft_bem // 2
    if rir_len is None:
        rir_len = int(delay_samples.max()) + n_fft_bem + pre + 64

    all_corners = np.concatenate([q00, q01, q10, q11])
    unique_doa, inverse = np.unique(all_corners, return_inverse=True)
    inv00, inv01, inv10, inv11 = np.split(inverse, 4)

    n_panel = len(panels)
    bem_unique = cached_bem[:, unique_doa, :]                  # (N, n_unique, n_freq)
    bem_filters_raw = np.fft.irfft(bem_unique, n=n_fft_bem, axis=-1)
    # fftshift on time axis: index 0 → N/2, index N-1 → N/2-1
    bem_filters = np.fft.fftshift(bem_filters_raw, axes=-1).astype(np.float32)

    rir = np.zeros((n_panel, rir_len), dtype=np.float32)
    for k in range(R.size):
        dly = int(delay_samples[k])
        start = dly - pre
        end = start + n_fft_bem
        # bounds
        if end <= 0 or start >= rir_len:
            continue
        filt = (w00[k] * bem_filters[:, inv00[k], :] +
                w01[k] * bem_filters[:, inv01[k], :] +
                w10[k] * bem_filters[:, inv10[k], :] +
                w11[k] * bem_filters[:, inv11[k], :])
        # clamp
        f_lo, f_hi = 0, n_fft_bem
        if start < 0:
            f_lo = -start
            start = 0
        if end > rir_len:
            f_hi = n_fft_bem - (end - rir_len)
            end = rir_len
        rir[:, start:end] += amp[k] * filt[:, f_lo:f_hi]
    return rir


def gen_diffuse_noise_simple(noise_mono, T_samples, mic_pos, fs=16000, c=C_SOUND):
    import math, scipy.signal, scipy.linalg
    M = mic_pos.shape[0]
    L = T_samples
    nfft = 256
    noise_mono = noise_mono - np.mean(noise_mono)
    noise_M = np.zeros((L, M))
    for m in range(M):
        noise_M[:, m] = noise_mono[m*L:(m+1)*L]
    ww = 2 * math.pi * fs * np.arange(nfft//2 + 1) / nfft
    DC = np.zeros((M, M, nfft//2 + 1))
    for p in range(M):
        for q in range(M):
            if p == q:
                DC[p, q, :] = 1.0
            else:
                d = np.linalg.norm(mic_pos[p] - mic_pos[q])
                DC[p, q, :] = np.sinc(ww * d / (c * math.pi))
    noise_pad = np.vstack([np.zeros((nfft//2, M)), noise_M, np.zeros((nfft//2, M))])
    f, t, N_stft = scipy.signal.stft(noise_pad.T, window='hann',
                                      nperseg=nfft, noverlap=int(0.75*nfft), nfft=nfft)
    X = np.zeros(N_stft.shape, dtype=complex)
    for k in range(1, nfft//2 + 1):
        C_chol = scipy.linalg.cholesky(DC[:, :, k])
        X[:, k, :] = (np.squeeze(N_stft[:, k, :]).T @ np.conj(C_chol)).T
    _, x = scipy.signal.istft(X, window='hann', nperseg=nfft,
                                noverlap=int(0.75*nfft), nfft=nfft)
    x = x.T[nfft//2:-nfft//2, :]
    return x[:L, :]


def synth_scene(source_mono: np.ndarray,
                  panels: np.ndarray,
                  cached_bem: np.ndarray,
                  bem_meta: dict,
                  room_sz=(8.0, 6.0, 3.0),
                  rt60=0.5,
                  max_order=10,
                  body_center=None,
                  source_doa_az_el_deg: tuple = None,
                  source_distance: float = 1.5,
                  fs=16000, n_fft=512,
                  snr_db: float = None,
                  noise_mono: np.ndarray = None,
                  rng: np.random.Generator = None,
                  ) -> dict:
    rng = rng or np.random.default_rng(0)
    body_center = np.array(body_center if body_center is not None else
                            [room_sz[0]/2, room_sz[1]/2, 1.5])

    if source_doa_az_el_deg is None:
        az = float(rng.uniform(0, 360))
        el = float(rng.uniform(-30, 30))
    else:
        az, el = source_doa_az_el_deg
    az_r = np.deg2rad(az); el_r = np.deg2rad(el)
    src_offset = source_distance * np.array([
        np.cos(el_r)*np.cos(az_r),
        np.cos(el_r)*np.sin(az_r),
        np.sin(el_r),
    ])
    src_pos = body_center + src_offset
    src_pos = np.clip(src_pos, 0.2, np.array(room_sz) - 0.2)
    true_doa_unit = src_offset / np.linalg.norm(src_offset)

    # PRA ISM
    e_abs, _ = pra.inverse_sabine(rt60, list(room_sz))
    materials = pra.Material(e_abs)
    room = pra.ShoeBox(list(room_sz), fs=fs, materials=materials,
                        max_order=max_order, air_absorption=False)
    room.add_source(src_pos.tolist())
    room.add_microphone_array(body_center.reshape(3, 1))
    room.image_source_model()

    rir = build_rir_per_image(
        room.sources[0].images, room.sources[0].damping[0],
        body_center, panels, cached_bem, bem_meta,
        fs=fs, n_fft_bem=n_fft,
    )  # (N, rir_len)

    # source convolve with rir per panel
    N = len(panels)
    T_src = len(source_mono)
    rir_len = rir.shape[1]
    x = np.zeros((T_src, N), dtype=np.float32)
    for p in range(N):
        x[:, p] = fftconvolve(source_mono, rir[p], mode='full')[:T_src]

    if snr_db is not None and noise_mono is not None:
        sig_pow = np.mean(np.abs(x)**2)
        # noise generate
        mic_pos = body_center + np.zeros((N, 3))
        n_per_mic = noise_mono[:T_src*N].reshape(N, T_src).T
        noise_pow = np.mean(np.abs(n_per_mic)**2)
        scale = np.sqrt(sig_pow / 10**(snr_db/10) / (noise_pow + 1e-12))
        x = x + scale * n_per_mic.astype(np.float32)

    return {
        'x': x.astype(np.float32),
        'rir': rir,
        'true_doa_unit': true_doa_unit.astype(np.float32),
        'true_doa_az_el': (az, el),
        'src_pos': src_pos,
        'room_sz': room_sz,
        'rt60': rt60,
        'snr_db': snr_db if snr_db is not None else float('inf'),
    }
