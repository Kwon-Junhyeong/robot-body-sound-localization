"""Training audio: speech in a random room, heard through the body. Writes 4-channel wav files and their labels."""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bem_table import BEMTableLazy
from synthesize_reverb import synthesize_reverb
from noise import sample_noise, add_noise_at_snr


FS = 16000
N_FFT = 512
HOP = 256
C_SOUND = 343.0

# FN-SSL: room 6-10 × 6-8 × 2.5-6 m, distance 0.5-2 m
ROOM_SZ_MIN = np.array([6.0, 6.0, 2.5])
ROOM_SZ_MAX = np.array([10.0, 8.0, 6.0])
ARRAY_POS_MIN = np.array([0.30, 0.30, 0.30])
ARRAY_POS_MAX = np.array([0.70, 0.70, 0.70])
T60_MIN, T60_MAX = 0.2, 1.3
SRC_POS_RATIO_MIN = 0.05
SRC_POS_RATIO_MAX = 0.95
SRC_MAX_RESAMPLE = 200
SOURCE_DIST_MIN, SOURCE_DIST_MAX = 0.5, 2.5
EL_MIN, EL_MAX = -90.0, 90.0
SNR_MIN_DEFAULT, SNR_MAX_DEFAULT = -5.0, 15.0
PRA_MAX_ORDER = 10
N_PANELS_PICK = 4

FIXED_PANEL_BODIES = {
    "box_24x20x16":                       np.array([0, 3, 5, 6], dtype=np.int64),    # arr_a
    "box_24x20x16__arr_b":                np.array([0, 1, 2, 3], dtype=np.int64),    # planar
    "box_24x20x16__arr_c":                np.array([11, 16, 19, 21], dtype=np.int64),
    "free_field_box_24x20x16":            np.array([0, 3, 5, 6], dtype=np.int64),
    "free_field_box_24x20x16__arr_b":     np.array([0, 1, 2, 3], dtype=np.int64),    # paired
    "free_field_box_24x20x16__arr_c":     np.array([11, 16, 19, 21], dtype=np.int64),
}


_W_BEM = None
_W_AUDIO_FILES = None


def init_worker(bem_h5_path: str, libri_root: str, noise_dir: str):
    try:
        import noise as _n
        _n.NOISE_DIR = noise_dir
        global _W_BEM, _W_AUDIO_FILES
        _W_BEM = BEMTableLazy(bem_h5_path, eager_load=False, apply_conjugate=True)
        _W_AUDIO_FILES = _list_audio_files(libri_root)
        if not _W_AUDIO_FILES:
            raise RuntimeError(f"no audio files in {libri_root}")
        if not os.path.isdir(noise_dir):
            raise RuntimeError(f"no noise directory: {noise_dir}")
    except Exception as e:
        sys.stderr.write(
            f"[init_worker FAIL pid={os.getpid()}] {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}\n"
        )
        sys.stderr.flush()
        raise


def _list_audio_files(root: str):
    files = []
    for r, _, fs_ in os.walk(root):
        for f in fs_:
            if f.endswith('.flac') or f.endswith('.wav'):
                files.append(os.path.join(r, f))
    return sorted(files)


def az_el_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)],
                     dtype=np.float64)


def load_audio_segment(rng: np.random.Generator, audio_files,
                        T_samples: int) -> np.ndarray:
    for _ in range(50):
        path = audio_files[rng.integers(0, len(audio_files))]
        try:
            s, sr = sf.read(path)
        except Exception:
            continue
        if s.ndim > 1:
            s = s[:, 0]
        if sr != FS:
            continue
        if len(s) >= T_samples:
            start = rng.integers(0, len(s) - T_samples + 1)
            seg = s[start: start + T_samples].astype(np.float32)
        else:
            seg = np.pad(s, (0, T_samples - len(s))).astype(np.float32)
        seg = seg - seg.mean()
        if np.std(seg) > 1e-4:
            return seg
    raise RuntimeError("no valid speech segment found")


def synthesize_one(seed: int, body_id: str, body_centroids: np.ndarray,
                    n_total_panels: int, t_samples: int,
                    snr_min: float, snr_max: float,
                    clean: bool = False) -> dict:
    import pyroomacoustics as _pra
    rng = np.random.default_rng(seed)

    if body_id in FIXED_PANEL_BODIES:
        panel_indices = FIXED_PANEL_BODIES[body_id].copy()
    else:
        panel_indices = rng.choice(n_total_panels, size=N_PANELS_PICK,
                                     replace=False).astype(np.int64)
    panel_pos = body_centroids[panel_indices].astype(np.float64)

    room_sz = None; t60 = None
    for _try in range(50):
        rs = tuple(rng.uniform(ROOM_SZ_MIN, ROOM_SZ_MAX).tolist())
        tt = float(rng.uniform(T60_MIN, T60_MAX))
        try:
            _pra.inverse_sabine(tt, list(rs))
            room_sz, t60 = rs, tt
            break
        except Exception:
            continue
    if room_sz is None:
        room_sz = tuple(ROOM_SZ_MIN.tolist())
        t60 = float(T60_MAX)

    # 3) SNR
    snr_db = float(rng.uniform(snr_min, snr_max))

    array_pos_ratio = rng.uniform(ARRAY_POS_MIN, ARRAY_POS_MAX)
    body_center = array_pos_ratio * np.array(room_sz)

    room_arr = np.array(room_sz)
    lo_bound = SRC_POS_RATIO_MIN * room_arr
    hi_bound = SRC_POS_RATIO_MAX * room_arr
    src_pos = None
    az = el = distance = 0.0
    d_unit = np.array([1.0, 0.0, 0.0])
    for _try in range(SRC_MAX_RESAMPLE):
        az = float(rng.uniform(0, 360))
        el = float(rng.uniform(EL_MIN, EL_MAX))
        distance = float(rng.uniform(SOURCE_DIST_MIN, SOURCE_DIST_MAX))
        d_unit = az_el_to_unit(az, el)
        src_pos = body_center + distance * d_unit
        if np.all(src_pos >= lo_bound) and np.all(src_pos <= hi_bound):
            break
    else:
        src_pos = np.clip(src_pos, lo_bound, hi_bound)
    actual_dir = src_pos - body_center
    norm = np.linalg.norm(actual_dir)
    if norm < 1e-6:
        d_unit_actual = d_unit
    else:
        d_unit_actual = actual_dir / norm
    az_actual = float(np.rad2deg(np.arctan2(d_unit_actual[1], d_unit_actual[0])) % 360)
    el_actual = float(np.rad2deg(np.arcsin(np.clip(d_unit_actual[2], -1.0, 1.0))))
    distance_actual = float(np.linalg.norm(actual_dir))

    # 6) source signal
    src_mono = load_audio_segment(rng, _W_AUDIO_FILES, t_samples)

    if clean:
        x_direct = synthesize_reverb(
            source=src_mono, source_pos=src_pos, body_center=body_center,
            panel_indices=panel_indices, room_dim=tuple(room_sz),
            rt60=t60, bem_table=_W_BEM, fs=FS, n_fft=N_FFT, hop=HOP,
            max_order=0,
        )
        x = x_direct.astype(np.float32)
        snr_db = float('inf')
    else:
        x_clean = synthesize_reverb(
            source=src_mono, source_pos=src_pos, body_center=body_center,
            panel_indices=panel_indices, room_dim=tuple(room_sz),
            rt60=t60, bem_table=_W_BEM, fs=FS, n_fft=N_FFT, hop=HOP,
            max_order=PRA_MAX_ORDER,
        )
        noise_per_mic = sample_noise(rng, T_samples=t_samples,
                                       mic_pos=panel_pos, fs=FS)
        x = add_noise_at_snr(x_clean, noise_per_mic, snr_db).astype(np.float32)

    peak = float(np.max(np.abs(x)) + 1e-9)
    scale = 0.95 / peak
    x_int16 = np.clip(x * scale * 32767, -32768, 32767).astype(np.int16)

    return {
        'wav': x_int16,                          # (T, 4) int16
        'wav_scale': float(scale),
        'meta': {
            'd_unit':        d_unit_actual.astype(np.float32),
            'az_deg':        np.float32(az_actual),
            'el_deg':        np.float32(el_actual),
            'distance':      np.float32(distance_actual),
            'src_pos':       src_pos.astype(np.float32),
            'body_center':   body_center.astype(np.float32),
            'room_sz':       np.array(room_sz, dtype=np.float32),
            't60':           np.float32(t60),
            'snr':           np.float32(snr_db) if not np.isinf(snr_db)
                              else np.float32(999.0),
            'body_id':       np.array(body_id),
            'panel_indices': panel_indices.astype(np.int64),
            'panel_pos':     panel_pos.astype(np.float32),
            'fs':            np.int32(FS),
            't_sec':         np.float32(t_samples / FS),
            'wav_scale':     np.float32(scale),
        },
    }


def worker(args: tuple) -> bool:
    sample_idx, seed, body_id, body_centroids, n_total_panels, \
        t_samples, snr_min, snr_max, clean, out_dir = args
    try:
        sample = synthesize_one(
            seed, body_id, body_centroids, n_total_panels,
            t_samples, snr_min, snr_max, clean=clean,
        )
        wav_path = Path(out_dir) / f"{sample_idx:06d}.wav"
        npz_path = Path(out_dir) / f"{sample_idx:06d}.npz"
        sf.write(str(wav_path), sample['wav'], FS, subtype='PCM_16')
        np.savez(str(npz_path), **sample['meta'])
        return True
    except Exception as e:
        sys.stderr.write(
            f"[worker FAIL pid={os.getpid()} sample={sample_idx}] "
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}\n"
        )
        sys.stderr.flush()
        return False


def process_split(bem_h5_path: Path, body_id: str, split: str,
                   n_samples: int, base_seed: int,
                   out_split_dir: Path, libri_root: str, noise_dir: str,
                   t_samples: int, snr_min: float, snr_max: float,
                   n_workers: int, clean: bool = False):
    out_split_dir.mkdir(parents=True, exist_ok=True)

    import h5py
    with h5py.File(bem_h5_path, 'r') as f:
        body_centroids = f['centroids'][:].astype(np.float64)
        n_total_panels = int(f.attrs['n_panels'])

    jobs = [
        (i, base_seed * 1_000_000 + i, body_id, body_centroids, n_total_panels,
         t_samples, snr_min, snr_max, clean, str(out_split_dir))
        for i in range(n_samples)
    ]

    print(f"  [{body_id}/{split}] {n_samples} samples, {n_workers} workers (spawn)")
    t0 = time.time()
    success = 0
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=spawn_ctx,
        initializer=init_worker,
        initargs=(str(bem_h5_path), libri_root, noise_dir),
    ) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for f in tqdm(as_completed(futures), total=len(futures),
                       desc=f"{body_id}/{split}"):
            try:
                if f.result():
                    success += 1
            except Exception as e:
                sys.stderr.write(f"[future FAIL] {type(e).__name__}: {e}\n")
                sys.stderr.flush()
    elapsed = time.time() - t0
    print(f"  [{body_id}/{split}] done {success}/{n_samples} in "
          f"{elapsed:.1f}s ({elapsed/n_samples*1000:.0f} ms/sample)")


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bem_dir", required=True,
                    help="directory of BEM tables (p_total_*_24panel.h5)")
    ap.add_argument("--libri_root", required=True,
                    help="LibriSpeech root")
    ap.add_argument("--noise_dir", required=True,
                    help="noise directory (NOISEX-92)")
    ap.add_argument("--out_dir", required=True,
                    help="output directory: <out_dir>/<body_id>/<split>/")
    ap.add_argument("--bodies", default="all",
                    help="body ids separated by commas, or 'all'")
    ap.add_argument("--t_sec", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_dev",   type=int, default=4000)
    ap.add_argument("--n_test",  type=int, default=4000)
    ap.add_argument("--snr_min", type=float, default=SNR_MIN_DEFAULT)
    ap.add_argument("--snr_max", type=float, default=SNR_MAX_DEFAULT)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--clean", action="store_true",
                    help="direct path only, no reverberation or noise")
    ap.add_argument("--splits", default="train,dev,test")
    args = ap.parse_args()

    bem_dir = Path(args.bem_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t_samples = int(args.t_sec * FS)

    all_bems = sorted(bem_dir.glob("p_total_*_24panel.h5"))
    if not all_bems:
        raise SystemExit(f"no BEM table: {bem_dir}/p_total_*_24panel.h5")
    body_map = {}
    for p in all_bems:
        #         p_total_free_field_<ref>_az360el181_24panel.h5
        name = p.stem.replace("p_total_", "")
        body_id = name.split("_az360")[0]
        body_map[body_id] = p
    print("BEM tables found:")
    for bid, p in body_map.items():
        print(f"  {bid:35s} → {p.name}")

    if args.bodies == "all":
        targets = list(body_map.keys())
    else:
        targets = [b.strip() for b in args.bodies.split(",")]
        for b in targets:
            if b not in body_map:
                base = b.split("__")[0]
                if base in body_map:
                    body_map[b] = body_map[base]
                else:
                    raise SystemExit(f"unknown body {b!r}. choices={list(body_map.keys())}")

    splits = args.splits.split(",")
    n_map = {"train": args.n_train, "dev": args.n_dev, "test": args.n_test}
    seed_map = {"train": 100, "dev": 102, "test": 101}

    print(f"\n[Plan]")
    print(f"  bodies     : {targets}")
    print(f"  splits     : {splits}")
    print(f"  per body   : train={args.n_train}, dev={args.n_dev}, test={args.n_test}")
    print(f"  total      : {len(targets) * sum(n_map[s] for s in splits)} samples")
    print(f"  t_sec      : {args.t_sec} ({t_samples} samples)")
    print(f"  SNR        : [{args.snr_min}, {args.snr_max}]")
    print(f"  workers    : {args.n_workers}")
    print(f"  clean mode : {args.clean}")
    print(f"  out_dir    : {out_dir}")
    print("  panels     :")
    for b in targets:
        if b in FIXED_PANEL_BODIES:
            idx = FIXED_PANEL_BODIES[b].tolist()
            print(f"    {b:35s} → fixed panels {idx}")
        else:
            print(f"    {b:35s} → 4 of the 24 at random")

    t_total = time.time()
    for body_id in targets:
        bem_path = body_map[body_id]
        print(f"\n{'='*70}\n[BODY] {body_id} ({bem_path.name})\n{'='*70}")
        for split in splits:
            out_split = out_dir / body_id / split
            process_split(
                bem_h5_path=bem_path, body_id=body_id, split=split,
                n_samples=n_map[split], base_seed=seed_map[split],
                out_split_dir=out_split,
                libri_root=args.libri_root, noise_dir=args.noise_dir,
                t_samples=t_samples, snr_min=args.snr_min, snr_max=args.snr_max,
                n_workers=args.n_workers, clean=args.clean,
            )

    elapsed = time.time() - t_total
    print(f"\n{'='*70}\ndone ({elapsed:.0f}s = {elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
