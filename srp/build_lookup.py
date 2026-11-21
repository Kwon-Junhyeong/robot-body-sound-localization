"""Per-direction templates for the matcher, from a BEM table: unit cross-spectra of the microphone pairs and their log magnitude ratios."""
import os, sys, time, argparse
import numpy as np
import h5py


def precompute_mag(bem_path, panel_indices, res_the=181, res_phi=360,
                   freq_lo=300.0, freq_hi=7500.0, out_path=None):
    t0 = time.time()
    print(f'[mag-lookup] panel={panel_indices}, grid=({res_the},{res_phi}), freq=[{freq_lo},{freq_hi}]')

    with h5py.File(bem_path, 'r') as f:
        bem_doa = f['doa_grid'][...].astype(np.float32)
        bem_freqs = f['freqs'][...].astype(np.float32)
        p_full = f['p']
        p_sel = np.empty((len(panel_indices), bem_doa.shape[0], bem_freqs.size), dtype=np.complex64)
        for k, pi in enumerate(panel_indices):
            p_sel[k] = p_full[pi, :, :]
    print(f'  BEM: doa {bem_doa.shape}, p_sel {p_sel.shape}')

    theta = np.linspace(0, np.pi, res_the).astype(np.float32)
    phi = np.linspace(-np.pi, np.pi, res_phi, endpoint=False).astype(np.float32)
    THE, PHI = np.meshgrid(theta, phi, indexing='ij')
    grid_xyz = np.stack([np.sin(THE)*np.cos(PHI), np.sin(THE)*np.sin(PHI), np.cos(THE)], axis=-1).astype(np.float32)
    G = res_the * res_phi
    grid_xyz_flat = grid_xyz.reshape(G, 3)

    # nearest BEM idx
    idx = np.empty(G, dtype=np.int64)
    chunk = 512
    for i in range(0, G, chunk):
        cos = grid_xyz_flat[i:i+chunk] @ bem_doa.T
        idx[i:i+chunk] = cos.argmax(axis=1)

    freq_mask = (bem_freqs >= freq_lo) & (bem_freqs <= freq_hi)
    freq_idx = np.where(freq_mask)[0]
    F = len(freq_idx)

    p_grid = p_sel[:, idx, :][:, :, freq_idx]  # (M, G, F)
    p_aligned = np.conj(p_grid)  # exp(+jωt) → exp(-jωt)

    M = len(panel_indices)
    PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    P = len(PAIRS)

    a_pair_unit = np.empty((G, P, F), dtype=np.complex64)
    log_mag_ratio = np.empty((G, P, F), dtype=np.float32)

    for p_idx, (i, j) in enumerate(PAIRS):
        cross = p_aligned[i] * np.conj(p_aligned[j])
        mag_cross = np.abs(cross) + 1e-12
        a_pair_unit[:, p_idx, :] = (cross / mag_cross).astype(np.complex64)
        mag_i = np.abs(p_aligned[i]) + 1e-12
        mag_j = np.abs(p_aligned[j]) + 1e-12
        log_mag_ratio[:, p_idx, :] = np.log(mag_i / mag_j).astype(np.float32)

    log_mag_single = np.log(np.abs(p_aligned) + 1e-12).astype(np.float32)  # (M, G, F)
    log_mag_single = log_mag_single.transpose(1, 0, 2)  # (G, M, F)
    log_mag_single -= log_mag_single.mean(axis=(1, 2), keepdims=True)

    if out_path is None:
        out_path = f'./data_cache/bem_lookup_mag_grid{res_the}x{res_phi}_f{int(freq_lo)}to{int(freq_hi)}.npz'

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path,
        a_pair_unit=a_pair_unit,
        log_mag_ratio=log_mag_ratio,
        log_mag_single=log_mag_single,
        theta=theta, phi=phi,
        grid_xyz=grid_xyz,
        freq_hz=bem_freqs[freq_idx],
        panel_indices=np.array(panel_indices, dtype=np.int32),
        pairs=np.array(PAIRS, dtype=np.int32),
        res_the=np.int32(res_the),
        res_phi=np.int32(res_phi))
    sz = os.path.getsize(out_path) / 1024 / 1024
    print(f'  log_mag_ratio std/mean: {log_mag_ratio.std():.3f} / {log_mag_ratio.mean():.3f} ')
    print(f'  saved {out_path}  ({sz:.1f} MB, dt={time.time()-t0:.1f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bem', help='BEM table (HDF5: p, doa_grid, freqs)')
    ap.add_argument('out', help='templates file to write (.npz)')
    ap.add_argument('--panel', nargs='+', type=int, default=[0, 3, 5, 6])
    ap.add_argument('--res_the', type=int, default=181)
    ap.add_argument('--res_phi', type=int, default=360)
    ap.add_argument('--freq_lo', type=float, default=300.0)
    ap.add_argument('--freq_hi', type=float, default=7500.0)
    args = ap.parse_args()
    precompute_mag(args.bem, args.panel, args.res_the, args.res_phi,
                   args.freq_lo, args.freq_hi, args.out)


if __name__ == '__main__':
    main()
