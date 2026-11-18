# BEM table of a rigid body with microphones on it (GPU).
# For every direction and frequency: H = p_total / p_inc at each microphone.
# Requires CuPy for your CUDA version, e.g. pip install cupy-cuda12x

import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpx_la

C_SOUND = 343.0
DOA_CHUNK = 4096                 # directions solved at once (N_elem x 4096 complex128 is about 800 MB)

# body
MESH_TYPE = "box"                # "box" or "stl"
LX, LY, LZ = 0.24, 0.20, 0.16    # box size, m
N_EDGE = 32                      # divisions per edge; elements smaller than a sixth of the wavelength at 8 kHz
STL_PATH = "body.stl"            # used when MESH_TYPE = "stl", in mm

# microphones on the surface, m
MIC_POS = np.array([
    [0.12, 0.0686, 0.0580],
    [0.12, -0.0686, -0.0580],
    [-0.12, 0.0686, -0.0549],
    [-0.12, -0.0686, 0.0549],
])

N_AZ, N_EL = 180, 91             # 2 degree grid
FS, NFFT = 16000, 512            # 257 bins up to 8 kHz


def box_mesh(lx, ly, lz, n_per_edge=6):
    """Triangle mesh of a box: centroids, outward normals, areas, vertices, triangles."""
    half = np.array([lx, ly, lz]) / 2.0
    verts, tris = [], []
    for u_ax, v_ax, n_ax in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        for sign in (+1, -1):
            base = len(verts)
            us = np.linspace(-half[u_ax], half[u_ax], n_per_edge + 1)
            vs = np.linspace(-half[v_ax], half[v_ax], n_per_edge + 1)
            for u in us:
                for v in vs:
                    pt = [0.0, 0.0, 0.0]
                    pt[u_ax], pt[v_ax], pt[n_ax] = u, v, sign * half[n_ax]
                    verts.append(pt)
            nv = n_per_edge + 1
            for i in range(n_per_edge):
                for j in range(n_per_edge):
                    v00, v01 = base + i * nv + j, base + i * nv + j + 1
                    v10, v11 = base + (i + 1) * nv + j, base + (i + 1) * nv + j + 1
                    if sign > 0:
                        tris += [[v00, v01, v11], [v00, v11, v10]]
                    else:
                        tris += [[v00, v11, v01], [v00, v10, v11]]
    verts = np.asarray(verts)
    tris = np.asarray(tris, dtype=np.int64)
    centroids, normals, areas = _geometry(verts, tris)
    flip = (centroids * normals).sum(axis=1) < 0          # make every normal point outward
    if flip.any():
        tris[flip] = tris[flip][:, ::-1]
        centroids, normals, areas = _geometry(verts, tris)
    return centroids, normals, areas, verts, tris


def _geometry(verts, tris):
    v = verts[tris]
    cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    nrm = np.linalg.norm(cross, axis=1, keepdims=True).clip(min=1e-15)
    return v.mean(axis=1), cross / nrm, 0.5 * nrm.ravel()


def stl_mesh(path):
    """Triangle mesh from an STL file in mm. Normals are taken from the file (outward in CAD)."""
    import stl
    mesh = stl.mesh.Mesh.from_file(path)
    tri_verts = mesh.vectors
    verts, inverse = np.unique(tri_verts.reshape(-1, 3), axis=0, return_inverse=True)
    tris = inverse.reshape(len(tri_verts), 3)
    verts = verts / 1000.0
    v = verts[tris]
    centroids = v.mean(axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    n = np.linalg.norm(mesh.normals, axis=1)
    if (n < 1e-15).any():
        raise ValueError("triangles without normals; export the STL with normals")
    normals = mesh.normals / n[:, None]
    valid = areas > 1e-8                                   # degenerate triangles make the system ill-conditioned
    return centroids[valid], normals[valid], areas[valid], verts, tris[valid]


def assemble_M(k, centroids, normals, areas, tile=1024):
    """Double-layer operator: M_ij = G(r_ij) (-jk - 1/r_ij) (R_ij . n_j) / r_ij A_j, M_ii = 0, built in row tiles."""
    n = len(centroids)
    dtype = cp.complex64 if centroids.dtype == cp.float32 else cp.complex128
    j = dtype(1j)
    M = cp.empty((n, n), dtype=dtype)
    for i0 in range(0, n, tile):
        i1 = min(i0 + tile, n)
        R = centroids[i0:i1, None, :] - centroids[None, :, :]
        d = cp.linalg.norm(R, axis=2)
        d = cp.where(d < 1e-15, 1.0, d)
        G = cp.exp(-j * k * d) / (4.0 * cp.pi * d)
        M[i0:i1] = G * (-j * k - 1.0 / d) * cp.einsum("ijk,jk->ij", R, normals) / d * areas[None, :]
        del R, d, G
    cp.fill_diagonal(M, 0.0)
    return M


def spherical_doa_grid(n_az=72, n_el=37):
    """Unit vectors toward the source: azimuth 0 to 360, elevation -90 to 90 degrees."""
    az = np.linspace(0, 360, n_az, endpoint=False)
    el = np.linspace(-90, 90, n_el, endpoint=True)
    a, e = np.meshgrid(np.deg2rad(az), np.deg2rad(el), indexing="ij")
    a, e = a.ravel(), e.ravel()
    return np.column_stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]), az, el


def find_nearest_elements(mic_pos, centroids, max_dist=10e-3):
    d = np.linalg.norm(np.atleast_2d(mic_pos)[:, None, :] - centroids[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    for i, di in enumerate(d[np.arange(len(idx)), idx]):
        if di > max_dist:
            raise ValueError(f"mic {i} is {di * 1000:.1f} mm off the surface")
    return idx


def compute_h_table(freqs, doa_grid, centroids, normals, areas, mic_pos, c=C_SOUND, doa_chunk=DOA_CHUNK):
    """(0.5 I + M) p_total = p_inc for every direction; one LU factorization per frequency. Returns (doa, freq, mic)."""
    from tqdm import tqdm
    n, n_doa, n_mic = len(centroids), len(doa_grid), len(mic_pos)
    mic_idx = find_nearest_elements(mic_pos, centroids)
    cen, nor, are = cp.asarray(centroids), cp.asarray(normals), cp.asarray(areas)
    grid, mics = cp.asarray(doa_grid), cp.asarray(mic_pos)
    H = np.zeros((n_doa, len(freqs), n_mic), dtype=np.complex128)
    pool = cp.get_default_memory_pool()
    for i, f in tqdm(enumerate(freqs), total=len(freqs), desc="BEM"):
        if f < 1e-10:
            H[:, i, :] = 1.0
            continue
        k = 2.0 * cp.pi * f / c
        A = 0.5 * cp.eye(n, dtype=cp.complex128) + assemble_M(k, cen, nor, are)
        lu, piv = cpx_la.lu_factor(A, overwrite_a=True)
        del A
        p_inc_mics = cp.exp(1j * k * (grid @ mics.T)).T
        for d0 in range(0, n_doa, doa_chunk):
            d1 = min(d0 + doa_chunk, n_doa)
            p_inc = cp.exp(1j * k * (grid[d0:d1] @ cen.T)).T
            p_total = cpx_la.lu_solve((lu, piv), p_inc, overwrite_b=True)
            H[d0:d1, i, :] = cp.asnumpy((p_total[mic_idx, :] / p_inc_mics[:, d0:d1]).T)
            del p_inc, p_total
        del lu, piv, p_inc_mics
        pool.free_all_blocks()
    return H


if __name__ == "__main__":
    if MESH_TYPE == "box":
        centroids, normals, areas, verts, tris = box_mesh(LX, LY, LZ, N_EDGE)
    else:
        centroids, normals, areas, verts, tris = stl_mesh(STL_PATH)
    doa_grid, _, _ = spherical_doa_grid(N_AZ, N_EL)
    freqs = np.linspace(0, FS / 2, NFFT // 2 + 1)
    print(f"{len(centroids)} elements, {len(doa_grid)} directions, {len(freqs)} frequencies")
    H = compute_h_table(freqs, doa_grid, centroids, normals, areas, MIC_POS)
    np.savez_compressed("h_table.npz", H=H, doa_grid=doa_grid.astype(np.float32), freqs=freqs.astype(np.float32),
                        mic_pos=MIC_POS.astype(np.float32))
    print("saved h_table.npz", H.shape)
