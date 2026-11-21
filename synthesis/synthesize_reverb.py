"""One reverberant scene: image sources of the room, each passed through the body's response for its direction."""
from __future__ import annotations
import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from reverb_synth import build_rir_per_image


C_SOUND = 343.0


def synthesize_reverb(
    source: np.ndarray,
    source_pos: np.ndarray,
    body_center: np.ndarray,
    panel_indices: np.ndarray,
    room_dim: tuple,
    rt60: float,
    bem_table,
    fs: int = 16000,
    n_fft: int = 512,
    hop: int = 256,
    max_order: int = 10,
    c_sound: float = C_SOUND,
) -> np.ndarray:
    panel_indices = np.asarray(panel_indices, dtype=np.int64)
    body_center = np.asarray(body_center, dtype=np.float64).reshape(3)
    source_pos = np.asarray(source_pos, dtype=np.float64).reshape(3)
    room_dim = tuple(float(x) for x in room_dim)

    # 1) PRA Image Source Method
    e_abs, _ = pra.inverse_sabine(rt60, list(room_dim))
    materials = pra.Material(e_abs)
    room = pra.ShoeBox(
        list(room_dim), fs=fs, materials=materials,
        max_order=max_order, air_absorption=False,
    )
    room.add_source(source_pos.tolist())
    room.add_microphone_array(body_center.reshape(3, 1))
    room.image_source_model()

    images_pos = room.sources[0].images           # (3, n_img)
    damping = room.sources[0].damping[0]          # (n_img,)

    cached_bem = bem_table.get_panels(panel_indices)       # (N, n_doa, n_freq)
    bem_meta = {
        'n_az':    bem_table.n_az,
        'n_el':    bem_table.n_el,
        'az_step': bem_table.az_step_deg,
        'el_step': bem_table.el_step_deg,
    }

    rir = build_rir_per_image(
        images_pos, damping, body_center,
        panel_indices, cached_bem, bem_meta,
        fs=fs, n_fft_bem=n_fft, c_sound=c_sound,
    )   # (N, rir_len) float32

    # 4) Source convolve
    N = len(panel_indices)
    T_src = len(source)
    x = np.zeros((T_src, N), dtype=np.float32)
    for p in range(N):
        x[:, p] = fftconvolve(source, rir[p], mode='full')[:T_src]

    return x
