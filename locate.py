"""Direction of a speaker in one 4-channel recording.

usage: python locate.py TABLE.h5 RECORDING.wav [--mics 0 1 2 3]
"""
import argparse

import numpy as np
import torch
from scipy.io import wavfile

from doa import matcher

ap = argparse.ArgumentParser()
ap.add_argument("table")
ap.add_argument("wav")
ap.add_argument("--mics", type=int, nargs=4, default=[0, 1, 2, 3])
a = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
table = matcher.load_table(a.table, a.mics, device)
fs, x = wavfile.read(a.wav)
assert fs == matcher.FS and x.ndim == 2 and x.shape[1] >= 4, "16 kHz, 4 channels"
x = x[:, :4].T.astype(np.float32)[None]
direction, _, answer = matcher.locate(torch.from_numpy(x).to(device), table)
az, el = matcher.azimuth_elevation(direction.cpu().numpy()[0])
print(f"azimuth {az:.1f} deg, elevation {el:.1f} deg, {'answer' if answer[0] else 'abstain'}")
