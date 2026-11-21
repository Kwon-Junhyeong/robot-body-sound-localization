"""Direction finding on a robot body: phase-only cross-spectra of four microphones matched to a BEM table."""
import h5py
import numpy as np
import torch

FS = 16000
NFFT = 512
HOP = 256


def load_table(path, mics, device):
    """BEM table to unit cross-spectra per direction.

    .npz from bem/make_table.py (H: direction x freq x mic), or .h5 (p: mic x direction x freq).
    """
    mics = np.asarray(mics)
    if str(path).endswith(".npz"):
        d = np.load(path)
        p = np.transpose(d["H"][:, :, mics], (2, 0, 1)).astype(np.complex64)
        grid = d["doa_grid"].astype(np.float32)
    else:
        order = np.argsort(mics)
        with h5py.File(path, "r") as f:
            p = f["p"][mics[order].tolist()].astype(np.complex64)[np.argsort(order)]
            grid = f["doa_grid"][:].astype(np.float32)
    n = len(mics)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # the table uses the e^{+jwt} convention and the STFT e^{-jwt}, hence conj(p_i) p_j
    flat = np.concatenate([np.conj(p[i]) * p[j] for i, j in pairs], axis=1)
    flat /= np.abs(flat) + 1e-12
    return {"flat": torch.from_numpy(flat), "grid": torch.from_numpy(grid).to(device), "pairs": pairs}


def cross_spectra(wav, pairs):
    """wav (batch, mic, samples) to time-summed cross-spectra (batch, pair, freq)."""
    b, c, n = wav.shape
    win = torch.hann_window(NFFT, device=wav.device)
    x = torch.stft(wav.reshape(b * c, n), n_fft=NFFT, hop_length=HOP, win_length=NFFT, window=win,
                   center=False, return_complex=True)
    x = x.reshape(b, c, x.shape[-2], x.shape[-1])
    return torch.stack([(x[:, i] * x[:, j].conj()).sum(-1) for i, j in pairs], dim=1)


def match(obs, table, chunk=8192):
    """Score of every direction in the table: phase agreement summed over pairs and frequencies."""
    u = (obs / (obs.abs() + 1e-12)).reshape(obs.shape[0], -1)
    flat = table["flat"]
    scores = torch.empty(u.shape[0], flat.shape[0], device=u.device)
    for s in range(0, flat.shape[0], chunk):
        scores[:, s:s + chunk] = (u @ flat[s:s + chunk].to(u.device).T).real
    return scores


@torch.no_grad()
def locate(wav, table):
    """Direction (unit vector) and score map for each recording in the batch."""
    obs = cross_spectra(wav, table["pairs"])
    scores = match(obs, table)
    return table["grid"][scores.argmax(-1)], scores


def azimuth_elevation(d):
    d = np.asarray(d, float)
    return np.degrees(np.arctan2(d[..., 1], d[..., 0])), np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
