"""SRP-PHAT with the body's responses in place of free-field steering, and an optional magnitude score."""
import numpy as np
import torch
import torch.nn as nn


class MagAwareSRP(nn.Module):
    def __init__(self, lookup_npz):
        super().__init__()
        c = np.load(lookup_npz)
        self.G = c['a_pair_unit'].shape[0]
        self.P = c['a_pair_unit'].shape[1]
        self.F = c['a_pair_unit'].shape[2]
        self.res_the = int(c['res_the'])
        self.res_phi = int(c['res_phi'])
        self.register_buffer('a_pair_unit', torch.from_numpy(c['a_pair_unit']))
        self.register_buffer('log_mag_ratio', torch.from_numpy(c['log_mag_ratio']))  # (G, P, F)
        self.register_buffer('grid_xyz', torch.from_numpy(c['grid_xyz']))
        self.register_buffer('freq_hz', torch.from_numpy(c['freq_hz']))
        self.panel_indices = list(c['panel_indices'])
        self.pairs = c['pairs']

    def forward(self, cross_unit, obs_log_mag_ratio, w_p=1.0, w_m=0.0):
        """
        cross_unit: (B, T, P, F) complex: PHAT-normalized cross
        obs_log_mag_ratio: (B, T, P, F) float: log(|X_i|/|X_j|) per pair, freq
        return power_map (B, T, R, C)
        """
        B, T, P, F = cross_unit.shape
        # phase score
        score_phase = torch.empty(B, T, self.G, device=cross_unit.device, dtype=torch.float32)
        a_conj = torch.conj(self.a_pair_unit)
        chunk = 4096
        for g0 in range(0, self.G, chunk):
            g1 = min(g0 + chunk, self.G)
            s = torch.einsum('btpf,gpf->btg', cross_unit, a_conj[g0:g1])
            score_phase[:, :, g0:g1] = s.real / (P * F)

        if w_m != 0:
            obs_mean = obs_log_mag_ratio.mean(dim=1)  # (B, P, F)
            # per-sample normalize
            obs_n = obs_mean - obs_mean.mean(dim=(-1, -2), keepdim=True)
            obs_n = obs_n / (obs_n.std(dim=(-1, -2), keepdim=True) + 1e-6)
            lmr = self.log_mag_ratio  # (G, P, F)
            lkp_n = lmr - lmr.mean(dim=(-1, -2), keepdim=True)
            lkp_std = lkp_n.std(dim=(-1, -2), keepdim=True) + 1e-6
            lkp_n = lkp_n / lkp_std
            # correlation = mean(obs_n * lkp_n) per (P, F)
            corr_t = torch.einsum('bpf,gpf->bg', obs_n, lkp_n) / (P * F)  # (B, G)
            score_mag = corr_t[:, None, :].expand(B, T, self.G)  # (B, T, G)
            total = w_p * score_phase + w_m * score_mag
        else:
            total = w_p * score_phase
        return total.view(B, T, self.res_the, self.res_phi)


def topk_argmax_subpixel(power_map, topk=48):
    B, T, R, C = power_map.shape
    peak_per_t = power_map.view(B, T, -1).max(dim=-1).values
    k = min(topk, T)
    idx = peak_per_t.topk(k, dim=1).indices
    gather_idx = idx[:, :, None, None].expand(-1, -1, R, C)
    s = torch.gather(power_map, 1, gather_idx).mean(dim=1)  # (B, R, C)

    flat = s.view(B, R * C)
    ix = flat.argmax(dim=1)
    r = (ix // C).long(); c = (ix % C).long()
    rm1 = (r - 1).clamp(0, R - 1); rp1 = (r + 1).clamp(0, R - 1)
    cm1 = (c - 1) % C; cp1 = (c + 1) % C
    b_idx = torch.arange(B, device=s.device)
    s00 = s[b_idx, r, c]
    dr_num = s[b_idx, rm1, c] - s[b_idx, rp1, c]
    dr_den = s[b_idx, rm1, c] - 2*s00 + s[b_idx, rp1, c]
    dr_den = torch.where(dr_den.abs() < 1e-9, torch.full_like(dr_den, 1e-9), dr_den)
    dr = (0.5 * dr_num / dr_den).clamp(-0.5, 0.5)
    dc_num = s[b_idx, r, cm1] - s[b_idx, r, cp1]
    dc_den = s[b_idx, r, cm1] - 2*s00 + s[b_idx, r, cp1]
    dc_den = torch.where(dc_den.abs() < 1e-9, torch.full_like(dc_den, 1e-9), dc_den)
    dc = (0.5 * dc_num / dc_den).clamp(-0.5, 0.5)

    rf = r.float() + dr; cf = c.float() + dc
    theta = rf * (np.pi / (R - 1))
    phi = -np.pi + cf * (2 * np.pi / C)
    st = torch.sin(theta)
    xyz = torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=1)
    return xyz / xyz.norm(dim=1, keepdim=True).clamp_min(1e-8)


def stft_cross_with_logmag(mic_sig, n_fft=512, hop=64, freq_idx=None, alpha=0.25):
    """
    return:
      cross_phat (B, T, 6, F) complex
      obs_log_mag_ratio (B, T, 6, F) float: log(|X_i|/|X_j|)
    """
    import torch
    B, Ts, M = mic_sig.shape
    sig = mic_sig.transpose(1, 2).reshape(B * M, Ts)
    window = torch.hann_window(n_fft, device=mic_sig.device)
    spec = torch.stft(sig, n_fft=n_fft, hop_length=hop, window=window, return_complex=True, center=True)
    Fall, Tfr = spec.shape[1], spec.shape[2]
    spec = spec.view(B, M, Fall, Tfr).permute(0, 3, 1, 2).contiguous()
    if freq_idx is not None:
        spec = spec[..., freq_idx]
    eps = 1e-8
    mag = spec.abs() + eps
    spec_phat = spec / mag.pow(1.0 - alpha)
    log_mag = torch.log(mag)  # (B, T, M, F)
    PAIRS = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    cross_list = []
    logmr_list = []
    for (i, j) in PAIRS:
        cross_list.append(spec_phat[..., i, :] * torch.conj(spec_phat[..., j, :]))
        logmr_list.append(log_mag[..., i, :] - log_mag[..., j, :])
    return torch.stack(cross_list, dim=2), torch.stack(logmr_list, dim=2)
