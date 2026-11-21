"""Lazy loader of a BEM table (HDF5), with interpolation between directions."""
import numpy as np
import h5py


class BEMTableLazy:

    def __init__(self, h5_path, eager_load=False, apply_conjugate=True,
                 rdcc_nbytes=2 * 1024**3):
        self.h5_path = str(h5_path)
        self.apply_conjugate = bool(apply_conjugate)
        self.eager_load = bool(eager_load)

        self.f = h5py.File(self.h5_path, 'r', rdcc_nbytes=rdcc_nbytes)

        self.doa_grid = np.asarray(self.f['doa_grid'])           # (n_doa, 3)
        self.freqs = np.asarray(self.f['freqs'])                 # (n_freq,)
        self.centroids = np.asarray(self.f['centroids'])         # (n_panels, 3)
        self.normals = np.asarray(self.f['normals'])             # (n_panels, 3)
        self.areas = np.asarray(self.f['areas'])                 # (n_panels,)

        # Shape
        self.n_panels, self.n_doa, self.n_freq = self.f['p'].shape

        self.n_az = int(self.f.attrs.get('n_az', 72))
        self.n_el = int(self.f.attrs.get('n_el', 37))
        if self.n_az * self.n_el != self.n_doa:
            raise ValueError(f"n_az ({self.n_az}) × n_el ({self.n_el}) "
                             f"≠ n_doa ({self.n_doa})")
        # Az / El step
        self.az_step_deg = 360.0 / self.n_az
        self.el_step_deg = 180.0 / (self.n_el - 1)

        # Convention
        self.use_chief = bool(self.f.attrs.get('use_chief', False))
        self.convention_str = str(self.f.attrs.get('convention', 'unknown'))
        self.c_sound = float(self.f.attrs.get('c_sound', 343.0))

        self._p_eager = None
        if eager_load:
            shape = self.f['p'].shape
            data = np.empty(shape, dtype=np.complex64)
            self.f['p'].read_direct(data)
            if self.apply_conjugate:
                data.imag *= -1                    # in-place conjugate
            self._p_eager = data

    # Properties

    @property
    def shape(self):
        return (self.n_panels, self.n_doa, self.n_freq)

    def __repr__(self):
        return (f"BEMTableLazy(panels={self.n_panels}, doa={self.n_doa} "
                f"({self.n_az}×{self.n_el}), freqs={self.n_freq}, "
                f"eager={self.eager_load}, conj={self.apply_conjugate})")

    def close(self):
        if hasattr(self, 'f') and self.f:
            self.f.close()

    def __del__(self):
        self.close()


    def get_panels(self, panel_indices):
        if self._p_eager is not None:
            data = self._p_eager[panel_indices]
        else:
            key = tuple(np.atleast_1d(panel_indices).tolist())
            if not hasattr(self, '_pcache'):
                self._pcache = {}
            if key in self._pcache:
                return self._pcache[key]
            idx_arr = np.atleast_1d(panel_indices)
            sort_perm = np.argsort(idx_arr)
            inv_perm = np.argsort(sort_perm)
            data_sorted = self.f['p'][idx_arr[sort_perm]]
            data = data_sorted[inv_perm]
            if np.isscalar(panel_indices):
                data = data[0]
            if self.apply_conjugate:
                data = np.conj(data)
            self._pcache[key] = data
        return data

    def get_at_doa_idx(self, panel_indices, doa_idx):
        if self._p_eager is not None:
            return self._p_eager[panel_indices, doa_idx, :]
        idx_arr = np.atleast_1d(panel_indices)
        sort_perm = np.argsort(idx_arr)
        inv_perm = np.argsort(sort_perm)
        data_sorted = self.f['p'][idx_arr[sort_perm], doa_idx, :]
        data = data_sorted[inv_perm]
        if np.isscalar(panel_indices):
            data = data[0]
        if self.apply_conjugate:
            data = np.conj(data)
        return data


    def az_el_to_grid_idx(self, az_deg, el_deg):
        az_norm = az_deg % 360.0
        el_clamp = np.clip(el_deg, -90.0, 90.0)
        az_idx = int(round(az_norm / self.az_step_deg)) % self.n_az
        el_idx = int(round((el_clamp + 90.0) / self.el_step_deg))
        el_idx = min(max(el_idx, 0), self.n_el - 1)
        return az_idx * self.n_el + el_idx

    def az_el_to_bilinear_corners(self, az_deg, el_deg):
        az_norm = az_deg % 360.0
        el_clamp = np.clip(el_deg, -90.0, 90.0)

        az_f = az_norm / self.az_step_deg
        el_f = (el_clamp + 90.0) / self.el_step_deg

        az_lo = int(np.floor(az_f)) % self.n_az
        az_hi = (az_lo + 1) % self.n_az
        el_lo = int(np.floor(el_f))
        # el clamp at top edge
        if el_lo >= self.n_el - 1:
            el_lo = self.n_el - 1
            el_hi = self.n_el - 1
        else:
            el_hi = el_lo + 1

        az_w = az_f - np.floor(az_f)
        el_w = el_f - np.floor(el_f)
        if el_hi == el_lo:
            el_w = 0.0

        q00 = az_lo * self.n_el + el_lo
        q01 = az_lo * self.n_el + el_hi
        q10 = az_hi * self.n_el + el_lo
        q11 = az_hi * self.n_el + el_hi

        w00 = (1 - az_w) * (1 - el_w)
        w01 = (1 - az_w) * el_w
        w10 = az_w * (1 - el_w)
        w11 = az_w * el_w

        return (q00, q01, q10, q11), (w00, w01, w10, w11)


    def get_at_doa_continuous(self, panel_indices, az_deg, el_deg):
        corners, weights = self.az_el_to_bilinear_corners(az_deg, el_deg)
        H_list, w_list = [], []
        for q, w in zip(corners, weights):
            if w == 0:
                continue
            H_list.append(self.get_at_doa_idx(panel_indices, q))
            w_list.append(w)

        if len(H_list) == 1:
            return H_list[0].astype(np.complex64)

        H_stack = np.stack(H_list, axis=0)                     # (k, ..., n_freq)
        w_arr = np.asarray(w_list, dtype=np.float64)            # (k,)
        w_arr = w_arr / w_arr.sum()                              # renormalize

        mag = np.abs(H_stack)                                    # (k, ..., n_freq)
        phase = np.angle(H_stack)                                # (k, ..., n_freq) ∈ [-π, π]

        delta = phase - phase[0:1]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        phase_aligned = phase[0:1] + delta                       # unwrap relative to corner 0

        ws = w_arr.reshape((-1,) + (1,) * (mag.ndim - 1))
        mag_avg = (mag * ws).sum(axis=0)
        phase_avg = (phase_aligned * ws).sum(axis=0)
        result = mag_avg * np.exp(1j * phase_avg)
        return result.astype(np.complex64)

    def _legacy_complex_bilinear(self, panel_indices, az_deg, el_deg):
        corners, weights = self.az_el_to_bilinear_corners(az_deg, el_deg)
        result = None
        for q, w in zip(corners, weights):
            if w == 0:
                continue
            p_at_q = self.get_at_doa_idx(panel_indices, q)
            result = (w * p_at_q) if result is None else (result + w * p_at_q)
        return result.astype(np.complex64)

    # Diagnostic

    def info(self):
        print(f"BEM table: {self.h5_path}")
        print(f"  shape: panels={self.n_panels}, doa={self.n_doa} "
              f"(az={self.n_az} × el={self.n_el}), freqs={self.n_freq}")
        print(f"  freq range: {self.freqs.min():.1f} ~ {self.freqs.max():.1f} Hz")
        print(f"  use_chief: {self.use_chief}")
        print(f"  convention: {self.convention_str}")
        print(f"  apply_conjugate: {self.apply_conjugate}")
        print(f"  eager_load: {self.eager_load}")
