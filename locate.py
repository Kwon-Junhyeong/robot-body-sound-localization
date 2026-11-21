"""Direction of a speaker in a 4-channel recording, one estimate per one-second window.

usage: python locate.py TEMPLATES.npz RECORDING.wav
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srp'))
from srp_mag_aware import MagAwareSRP, topk_argmax_subpixel, stft_cross_with_logmag

FS = 16000


def make_stft_freq_idx(bem_freqs_hz, fs=FS, n_fft=512):
    stft_freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    idx = np.empty(len(bem_freqs_hz), dtype=np.int64)
    for k, f in enumerate(bem_freqs_hz):
        idx[k] = int(np.argmin(np.abs(stft_freqs - f)))
    return idx


def locate(windows, srp, device, batch=4):
    """windows: (N, samples, 4) float32. Returns unit vectors (N, 3)."""
    sti = torch.from_numpy(make_stft_freq_idx(srp.freq_hz.cpu().numpy())).to(device)
    out = []
    with torch.no_grad():
        for b in range(0, len(windows), batch):
            w = torch.from_numpy(windows[b:b + batch]).to(device)
            cross, logmr = stft_cross_with_logmag(w, n_fft=512, hop=64, freq_idx=sti, alpha=0.25)
            cross = torch.conj(cross)      # a recording has the opposite time convention to the templates
            out.append(topk_argmax_subpixel(srp(cross, logmr, w_p=1.0, w_m=0.0), topk=48).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('templates')
    ap.add_argument('wav')
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    srp = MagAwareSRP(a.templates).to(device).eval()
    x, fs = sf.read(a.wav, dtype='float32')
    assert fs == FS and x.ndim == 2 and x.shape[1] >= 4, '16 kHz, 4 channels'
    n = len(x) // FS
    windows = np.ascontiguousarray(x[:n * FS, :4].reshape(n, FS, 4))
    d = locate(windows, srp, device)
    for k, v in enumerate(d):
        print('%4d s  azimuth %7.1f deg  elevation %6.1f deg' % (k, np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(v[2]))))


if __name__ == '__main__':
    main()
