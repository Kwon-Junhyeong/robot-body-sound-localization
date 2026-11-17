# BEM H-table generator for rigid scattering (GPU version)

# 임의 강체(box or STL) + 임의 마이크 배열
# 산란 전달함수 H = p_total / p_inc 테이블 계산
# GPU 가속: assemble_M, solve_surface_pressure → cupy
# 필요: pip install cupy-cuda12x (CUDA 버전에 맞게)

import numpy as np
import cupy as cp # GPU 연산 모듈 (numpy 대체)
import cupyx.scipy.linalg as cpx_la # LU factor / solve (주파수당 1회 분해 + 청크별 재사용)

C_SOUND = 343.0

# DOA 청크 크기 (n_doa 이 커서 p_inc/p_total 전체가 GPU에 올라가지 않을 때 사용)
# 4096 권장: (N_elem, 4096) complex128 ≈ 800 MB @ N_elem=12288
DOA_CHUNK = 4096

# 0. 설정
MESH_TYPE = "box"       # "box" or "stl"

# box 설정, MESH_TYPE = "box"일 때만 사용
LX, LY, LZ = 0.24, 0.20, 0.16  # 24 x 20 x 16 cm
N_EDGE = 32 # 각 모서리의 분할 개수, 8000hz 20cm 기준 28 권장, element size < lambda/6

# stl 설정, MESH_TYPE = "stl"일 때만 사용
STL_PATH = "torso.stl" # STL 파일 경로

# 마이크 배열

# 직사각형 24 x 20 x 16 cm 4ch 기준
MIC_POS = np.array([
    [ -0.10,  0.10,  0.06],   # +y face
    [ 0.10,  0.10,  0.06],   # +y face
    [ 0.12,  -0.08,  -0.06],   # +x face
    [ -0.12,  -0.08,  -0.06],   # -x face
])

# MIC_POS = np.array([
#     ( 0.0600,  0.0470, -0.0200),
#     (-0.0600,  0.0620, -0.0600),
#     (-0.0901, -0.0630,  0.0000),
#     ( 0.0901, -0.0780, -0.0650),
# ])


# H-table 해상도
# N_AZ, N_EL = 2, 1 # 180도 간격
# N_AZ, N_EL = 6, 3 # 60도 간격, 18개
# N_AZ, N_EL = 18, 10 # 20도 간격, 180개
# N_AZ, N_EL = 36, 19 # 10도 간격, 684개
# N_AZ, N_EL = 72, 37 # 5도 간격, 2664개
N_AZ, N_EL = 180, 91 # 2도 간격, 16380개
# N_AZ, N_EL = 360, 181 # 1도 간격, 65160개

# 주파수
FS, NFFT = 16000, 512 # FS/2=8000Hz NFFT//2+1=257bin




# 1. 메쉬 생성 

# 직육면체 메쉬 생성 함수
# lx, ly, lz: 직육면체의 크기
# n_per_edge: 각 모서리당 정점 개수
# 반환값: centroids, normals, areas, verts, tris
def box_mesh(lx, ly, lz, n_per_edge=6):
    
    half = np.array([lx, ly, lz]) / 2.0 # 직육면체의 절반 크기
    verts, tris = [], [] # 정점과 삼각형 리스트 초기화
    
    # 6개 면에 대해 반복
    for u_ax, v_ax, n_ax in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]: # (x,y,z) 순서로 6개 면
        for sign in (+1, -1): # 각 면의 양면
            base = len(verts) # 현재 정점 개수
            us = np.linspace(-half[u_ax], half[u_ax], n_per_edge + 1) # u축 방향 정점 생성
            vs = np.linspace(-half[v_ax], half[v_ax], n_per_edge + 1) # v축 방향 정점 생성
            
            for u in us: # u축 방향으로 반복
                for v in vs: # v축 방향으로 반복
                    pt = [0.0, 0.0, 0.0] # 정점 초기화
                    pt[u_ax], pt[v_ax], pt[n_ax] = u, v, sign * half[n_ax] # 정점 좌표 설정
                    verts.append(pt) # 정점 추가
                    
            nv = n_per_edge + 1 # 각 면의 정점 개수
            for i in range(n_per_edge): # u축 방향으로 반복
                for j in range(n_per_edge): # v축 방향으로 반복
                    v00 = base + i * nv + j # 정점 인덱스 계산
                    v01 = base + i * nv + (j + 1)
                    v10 = base + (i + 1) * nv + j
                    v11 = base + (i + 1) * nv + (j + 1)
                    if sign > 0: # 양면
                        tris.append([v00, v01, v11]) # 삼각형 인덱스 추가
                        tris.append([v00, v11, v10])
                    else: # 음면
                        tris.append([v00, v11, v01]) # 삼각형 인덱스 추가
                        tris.append([v00, v10, v11])
                    
    verts = np.asarray(verts) # 정점 배열 변환
    tris = np.asarray(tris, dtype=np.int64) # 삼각형 배열 변환
    
    # centroid / normal / area
    v_tri = verts[tris] # 삼각형 정점 좌표
    centroids = v_tri.mean(axis=1) # 삼각형 중심 계산
    cross = np.cross(v_tri[:, 1] - v_tri[:, 0], v_tri[:, 2] - v_tri[:, 0]) # 삼각형 법선 계산
    nrm = np.linalg.norm(cross, axis=1, keepdims=True).clip(min=1e-15) # 법선 벡터 정규화
    normals = cross / nrm # 법선 벡터 정규화
    areas = 0.5 * nrm.ravel() # 삼각형 면적 계산
    
    # 외향 법선 보정 (원점 중심 → centroid dot normal > 0)
    flip = (centroids * normals).sum(axis=1) < 0 # 법선 방향이 안쪽이면 뒤집기
    if flip.any(): # 뒤집힌 면이 있으면
        tris[flip] = tris[flip][:, ::-1] # 삼각형 인덱스 뒤집기
        v_tri = verts[tris] # 삼각형 정점 좌표
        centroids = v_tri.mean(axis=1) # 삼각형 중심 계산
        cross = np.cross(v_tri[:, 1] - v_tri[:, 0], v_tri[:, 2] - v_tri[:, 0]) # 삼각형 법선 계산
        nrm = np.linalg.norm(cross, axis=1, keepdims=True).clip(min=1e-15) # 법선 벡터 정규화
        normals = cross / nrm # 법선 벡터 정규화
        areas = 0.5 * nrm.ravel() # 삼각형 면적 계산
        
    return centroids, normals, areas, verts, tris # 삼각형 중심, 법선, 면적, 정점, 삼각형 반환


# STL 파일에서 삼각형 메쉬 로드
# path: STL 파일 경로
# 반환값: centroids, normals, areas, verts, tris
def stl_mesh(path):
    import stl  # numpy-stl 패키지
    mesh = stl.mesh.Mesh.from_file(path)

    verts_all = mesh.vectors  # (N_tri, 3, 3)
    n_tri = len(verts_all)

    # 중복 정점 제거하여 verts/tris 구성
    verts_flat = verts_all.reshape(-1, 3)  # (N_tri*3, 3)
    verts, inverse = np.unique(verts_flat, axis=0, return_inverse=True)
    tris = inverse.reshape(n_tri, 3)

    # mm → m 변환
    verts = verts / 1000.0

    # centroid / area 계산
    v_tri = verts[tris]  # (N_tri, 3, 3)
    centroids = v_tri.mean(axis=1)
    cross = np.cross(v_tri[:, 1] - v_tri[:, 0], v_tri[:, 2] - v_tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1) # 삼각형 면적 계산

    # 법선은 STL 원본 사용 (CAD 외향 법선 보장, cross product 재계산 시 오목 형상에서 오류)
    # STL 법선이 (0,0,0)인 경우 에러 (CAD에서 다시 export 필요)
    stl_normals = mesh.normals.copy()  # (N_tri, 3)
    stl_nrm = np.linalg.norm(stl_normals, axis=1) # 법선 벡터 정규화
    zero_mask = stl_nrm < 1e-15 # 법선이 (0,0,0)인 경우
    if zero_mask.any():
        raise ValueError(f"[STL] 법선이 (0,0,0)인 삼각형 {zero_mask.sum()}개 발견. CAD에서 법선 포함하여 다시 export하세요.")
    normals = stl_normals / stl_nrm[:, None]

    # 극소 면적 삼각형 제거 (퇴화 삼각형 → ill-conditioned 행렬 방지)
    valid = areas > 1e-8
    if (~valid).sum() > 0:
        print(f"[STL] 극소 삼각형 {(~valid).sum()}개 제거 ({n_tri} → {valid.sum()})")
    centroids = centroids[valid]
    normals = normals[valid]
    areas = areas[valid]
    tris = tris[valid]

    return centroids, normals, areas, verts, tris


# 2. BEM core

# Double-layer boundary operator M 행렬 조립.
# k: 파수
# centroids: 삼각형 중심
# normals: 삼각형 법선
# areas: 삼각형 면적
# 반환값: M
# M_ij = G(r_ij) * (-jk - 1/r_ij) * (R_ij . n_j) / r_ij * A_j (i: observation, j: source)
# G = exp(-jkr) / (4*pi*r) (Green function)
# Self-term M_ii = 0 (smooth surface) 
# R_ij = r_i - r_j  
def assemble_M(k, centroids, normals, areas, tile=1024):
    """입력: cupy 배열. GPU에서 연산. 행 블록 단위로 타일링하여 피크 메모리 축소."""
    N = len(centroids) # 삼각형 개수
    # dtype 추론: 입력 centroids 가 float32 면 complex64, 아니면 complex128
    out_dtype = cp.complex64 if centroids.dtype == cp.float32 else cp.complex128
    j_unit = cp.complex64(1j) if out_dtype == cp.complex64 else cp.complex128(1j)
    M = cp.empty((N, N), dtype=out_dtype) # 결과 버퍼만 미리 확보
    for i0 in range(0, N, tile):
        i1 = min(i0 + tile, N)
        R = centroids[i0:i1, None, :] - centroids[None, :, :] # (t, N, 3) 블록
        dist = cp.linalg.norm(R, axis=2) # (t, N)
        dist_safe = cp.where(dist < 1e-15, 1.0, dist) # 대각선/중복 centroid 나눗셈 방지
        G = cp.exp(-j_unit * k * dist_safe) / (4.0 * cp.pi * dist_safe) # Green function
        R_dot_nj = cp.einsum('ijk,jk->ij', R, normals) # (t, N) 내적
        M[i0:i1] = G * (-j_unit * k - 1.0 / dist_safe) * R_dot_nj / dist_safe * areas[None, :]
        del R, dist, dist_safe, G, R_dot_nj # 다음 타일 전에 중간 텐서 해제
    cp.fill_diagonal(M, 0.0) # 대각선 0으로 설정
    return M # M 행렬 반환



# 단일 주파수에서 모든 DOA에 대한 표면 음압 계산.
# (0.5*I + M) p_total = p_inc (강체 경계조건)
# p_inc = exp(+jk * doa . r)   (Convention B) 
# 반환 p_total: (N_elem, n_doa) complex128 

def solve_surface_pressure(freq, doa_grid, centroids, normals, areas, c=C_SOUND):
    """입력: cupy 배열. GPU에서 연산."""
    N = len(centroids) # 삼각형 개수
    if freq < 1e-10: # 주파수가 0에 가까우면
        return cp.ones((N, len(doa_grid)), dtype=cp.complex128) # 1 반환

    k = 2.0 * cp.pi * freq / c # 파수 계산
    M = assemble_M(k, centroids, normals, areas) # 타일링된 M 조립
    A = 0.5 * cp.eye(N, dtype=cp.complex128) + M # A 행렬 계산
    del M # solve 전에 M 해제 (피크 메모리 축소)
    p_inc = cp.exp(1j * k * (doa_grid @ centroids.T)).T # 입사파 계산
    p_total = cp.linalg.solve(A, p_inc) # GPU solve (LU 분해 + solve 한번에)
    del A, p_inc # LU/RHS 버퍼 해제
    return p_total # 표면 음압 반환

# 3. DOA grid

# 균등 구면 DOA grid 생성. (physics convention: el=0 → +z)
# n_az: 방위각 개수, n_el: 고도각 개수
# 반환값: doa_grid (n_az*n_el, 3), az_deg (n_az,), el_deg (n_el,)
def spherical_doa_grid(n_az=72, n_el=37):
    # az: 0~360(정면: 90도), el: -90~90(수평: 0도)
    az = np.linspace(0, 360, n_az, endpoint=False) # 방위각 범위
    el = np.linspace(-90, 90, n_el, endpoint=True) # 고도각 범위
    az_r, el_r = np.meshgrid(np.deg2rad(az), np.deg2rad(el), indexing='ij') # 라디안 변환
    az_f, el_f = az_r.ravel(), el_r.ravel() # 평탄화
    doa = np.column_stack([
        np.cos(el_f) * np.cos(az_f), # x 좌표
        np.cos(el_f) * np.sin(az_f), # y 좌표
        np.sin(el_f),                 # z 좌표
    ]) # DOA 벡터 생성
    return doa, az, el # DOA 벡터, 방위각, 고도각 반환

# 4. H-table computation

# 각 마이크에 가장 가까운 mesh element 인덱스 반환.
# mic_pos: (n_mic, 3) 마이크 위치
# centroids: (N_elem, 3) 삼각형 중심
# max_dist: 표면~마이크 허용 거리 [m] (초과 시 에러)
# 반환값: (n_mic,) 인덱스 배열
def find_nearest_elements(mic_pos, centroids, max_dist=10e-3):
    mic_pos = np.atleast_2d(mic_pos) # 마이크 위치 배열 변환
    dists = np.linalg.norm(mic_pos[:, None, :] - centroids[None, :, :], axis=2) # (n_mic, N_elem) 거리 행렬
    idx = np.argmin(dists, axis=1) # 각 마이크에 가장 가까운 삼각형 인덱스
    for i, d in enumerate(dists[np.arange(len(idx)), idx]): # 표면 거리 검증
        if d > max_dist:
            raise ValueError(f"mic[{i}]: {d*1000:.1f}mm off surface (threshold: {max_dist*1000:.1f}mm)")
    return idx

# H-table 계산: H = p_total(at mic) / p_inc(at mic).
# freqs: (n_freq,) 주파수 배열 [Hz]
# doa_grid: (n_doa, 3) DOA 단위 벡터
# centroids: (N_elem, 3) 삼각형 중심
# normals: (N_elem, 3) 삼각형 법선
# areas: (N_elem,) 삼각형 면적
# mic_pos: (n_mic, 3) 마이크 위치
# c: 음속
# verbose: 진행 상황 출력 여부
# 반환값: H_table (n_doa, n_freq, n_mic) complex128
def compute_h_table(freqs, doa_grid, centroids, normals, areas, mic_pos,
                    c=C_SOUND, doa_chunk=DOA_CHUNK):
    from tqdm import tqdm

    N = len(centroids) # 삼각형 개수
    n_doa = len(doa_grid)
    n_mic = len(mic_pos)
    print(f"[BEM] {len(freqs)} freqs x {n_doa} DOAs x {N} elements (doa_chunk={doa_chunk})")

    mic_idx = find_nearest_elements(mic_pos, centroids) # 각 마이크에 가장 가까운 삼각형 인덱스 (CPU)

    # GPU로 데이터 전송 (1회)
    centroids_g = cp.asarray(centroids)
    normals_g = cp.asarray(normals)
    areas_g = cp.asarray(areas)
    doa_grid_g = cp.asarray(doa_grid)
    mic_pos_g = cp.asarray(mic_pos)

    H_table = np.zeros((n_doa, len(freqs), n_mic), dtype=np.complex128) # H-table 초기화 (n_doa, n_freq, n_mic)

    mempool = cp.get_default_memory_pool() # 매 freq 이후 블록 반환용

    for i, freq in tqdm(enumerate(freqs), total=len(freqs), desc="[BEM-GPU] H-table"): # 주파수별 반복
        if freq < 1e-10: # DC: p_total=p_inc=1 → H=1
            H_table[:, i, :] = 1.0
            continue

        k = 2.0 * cp.pi * freq / c # 파수 계산

        # 1) 행렬 조립 & LU 분해 (주파수당 1회)
        M = assemble_M(k, centroids_g, normals_g, areas_g)
        A = 0.5 * cp.eye(N, dtype=cp.complex128) + M
        del M
        lu, piv = cpx_la.lu_factor(A, overwrite_a=True) # A는 이후 불필요
        del A

        # 2) 모든 DOA에 대한 p_inc(at mic) 미리 계산 (작음: n_mic x n_doa)
        p_inc_at_mics_all = cp.exp(1j * k * (doa_grid_g @ mic_pos_g.T)).T # (n_mic, n_doa)

        # 3) DOA 청크별로 solve → mic 값만 추출 → CPU 전송
        for d0 in range(0, n_doa, doa_chunk):
            d1 = min(d0 + doa_chunk, n_doa)
            doa_chk = doa_grid_g[d0:d1] # (c, 3)
            p_inc_chk = cp.exp(1j * k * (doa_chk @ centroids_g.T)).T # (N, c)
            p_total_chk = cpx_la.lu_solve((lu, piv), p_inc_chk, overwrite_b=True) # (N, c)
            p_at_mics = p_total_chk[mic_idx, :] # (n_mic, c)
            H_chk = (p_at_mics / p_inc_at_mics_all[:, d0:d1]).T # (c, n_mic)
            H_table[d0:d1, i, :] = cp.asnumpy(H_chk)
            del doa_chk, p_inc_chk, p_total_chk, p_at_mics, H_chk

        del lu, piv, p_inc_at_mics_all
        mempool.free_all_blocks() # 다음 freq 전에 GPU 메모리 풀 반환
    return H_table # H-table 반환


# 5. 시각화

# H-table 히트맵 저장 (dB / phase 각각 png 1장)
# H: (n_doa, n_freq, n_mic), freqs: (n_freq,), n_az/n_el: DOA grid 크기
# plot_freqs: 시각화할 주파수 리스트 [Hz]
def plot_h_table(H, freqs, n_az, n_el, n_mic, mic_pos=None, plot_freqs=[500, 1000, 2000, 4000, 6000, 8000]):
    import matplotlib.pyplot as plt

    az_deg = np.linspace(0, 360, n_az, endpoint=False)
    el_deg = np.linspace(-90, 90, n_el, endpoint=True)

    for plot_type in ["dB", "phase"]:
        fig, axes = plt.subplots(n_mic, len(plot_freqs), figsize=(5 * len(plot_freqs), 4 * n_mic))
        if n_mic == 1:
            axes = axes[np.newaxis, :]
        if len(plot_freqs) == 1:
            axes = axes[:, np.newaxis]

        # dB: 전체 subplot 공통 vmax 계산, outlier 영향 감소를 위한 percentile 사용
        if plot_type == "dB":
            all_vals = []
            for pf in plot_freqs:
                fi = np.argmin(np.abs(freqs - pf))
                for row in range(n_mic):
                    val = 20 * np.log10(np.abs(H[:, fi, row]).clip(min=1e-10))
                    all_vals.append(val)
            all_vals = np.concatenate(all_vals)
            global_vmax = max(3.0, np.percentile(all_vals, 99))

        for col, pf in enumerate(plot_freqs):
            fi = np.argmin(np.abs(freqs - pf))
            actual_f = freqs[fi]

            for row in range(n_mic):
                ax = axes[row, col]
                h_slice = H[:, fi, row]
                h_map = h_slice.reshape(n_az, n_el)

                if plot_type == "dB":
                    val = 20 * np.log10(np.abs(h_map).clip(min=1e-10))
                    im = ax.pcolormesh(az_deg, el_deg, val.T, cmap='RdBu_r',
                                       vmin=-global_vmax, vmax=global_vmax, shading='auto')
                else:
                    val = np.abs(np.rad2deg(np.angle(h_map)))  # 0°=변화없음, 180°=최대변화
                    im = ax.pcolormesh(az_deg, el_deg, val.T, cmap='Blues',
                                       vmin=0, vmax=180, shading='auto')

                # 마이크 위치를 az/el로 변환하여 마커 표시
                if mic_pos is not None:
                    p = mic_pos[row]
                    mic_az = np.rad2deg(np.arctan2(p[1], p[0])) % 360
                    mic_el = np.rad2deg(np.arctan2(p[2], np.sqrt(p[0]**2 + p[1]**2)))
                    ax.plot(mic_az, mic_el, 'k*', markersize=12)

                if row == 0:
                    ax.set_title(f"{actual_f:.0f} Hz")
                if col == 0:
                    if mic_pos is not None:
                        pos = mic_pos[row]
                        ax.set_ylabel(f"M{row} ({pos[0]*100:.0f},{pos[1]*100:.0f},{pos[2]*100:.0f})cm")
                    else:
                        ax.set_ylabel(f"Mic {row}")
                if row == n_mic - 1:
                    ax.set_xlabel("Azimuth (°)")
                plt.colorbar(im, ax=ax)

        fig.suptitle(f"H-table {'|H| (dB)' if plot_type == 'dB' else 'Phase (deg)'}", fontsize=14)
        fig.tight_layout()
        fname = f"h_table_{plot_type}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")


# 강체 형상 + 마이크 배열 3D 시각화 (6 view)
# centroids: (N_elem, 3), mic_pos: (n_mic, 3)
def plot_geometry(centroids, mic_pos, verts=None, tris=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    views = [
        ("Top (XY)",    90,  -90),
        ("Bottom (XY)", -90, -90),
        ("Front (XZ)",  0,   -90),
        ("Right (YZ)",  0,     0),
        ("Iso 1",       30,  -60),
        ("Iso 2",       30,   120),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13),
                             subplot_kw={'projection': '3d'})
    axes = axes.ravel()

    mic_pos = np.atleast_2d(mic_pos)
    all_pts = np.vstack([centroids, mic_pos])
    margin = 0.02

    # box_mesh인 경우 모서리 좌표 계산
    box_edges = None
    if verts is not None:
        mn, mx = verts.min(axis=0), verts.max(axis=0)
        corners = np.array([[x, y, z]
                             for x in [mn[0], mx[0]]
                             for y in [mn[1], mx[1]]
                             for z in [mn[2], mx[2]]])
        # 12개 모서리: 한 축만 다른 꼭짓점 쌍
        box_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                diff = np.abs(corners[i] - corners[j])
                if np.sum(diff > 1e-10) == 1:
                    box_edges.append((corners[i], corners[j]))

    for ax, (name, elev, azim) in zip(axes, views):
        # 메쉬 면 그리기
        if verts is not None and tris is not None:
            polys = verts[tris]
            pc = Poly3DCollection(polys, alpha=0.4, facecolor='orange',
                                  edgecolor='darkorange', linewidth=0.3)
            ax.add_collection3d(pc)

        # 직육면체 모서리 강조
        if box_edges is not None:
            for p1, p2 in box_edges:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color='black', linewidth=1.5, zorder=5)

        # 마이크 표시 (plot으로 그려서 depthshade 영향 제거)
        for i in range(len(mic_pos)):
            ax.plot([mic_pos[i, 0]], [mic_pos[i, 1]], [mic_pos[i, 2]],
                    'ko', markersize=8, zorder=10)
            ax.text(mic_pos[i, 0], mic_pos[i, 1], mic_pos[i, 2],
                    f" M{i}", fontsize=9, fontweight='bold', color='black')

        # 축 범위
        for setter, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            setter(all_pts[:, idx].min() - margin, all_pts[:, idx].max() + margin)

        # 축 라벨/틱 제거
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_title(name)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle("Geometry + Mic Array", fontsize=14)
    fig.tight_layout()
    fname = "geometry.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# 6. 실행

if __name__ == "__main__":
    import os

    # 1. Mesh generation
    if MESH_TYPE == "box":
        centroids, normals, areas, verts, tris = box_mesh(LX, LY, LZ, N_EDGE)
    elif MESH_TYPE == "stl":
        centroids, normals, areas, verts, tris = stl_mesh(STL_PATH)
    else:
        raise ValueError("Invalid MESH_TYPE")
    print(f"Mesh: {len(centroids)} elements (MESH_TYPE = {MESH_TYPE})")

    # 2. DOA grid
    doa_grid, _, _ = spherical_doa_grid(N_AZ, N_EL)
    print(f"DOA : {len(doa_grid)} directions ({N_AZ} az x {N_EL} el)")

    # 3. 주파수
    freqs = np.linspace(0, FS / 2, NFFT // 2 + 1)
    print(f"Freq: {len(freqs)} bins (0~{FS/2:.0f} Hz)")

    # 4. H-table 계산
    H = compute_h_table(freqs, doa_grid, centroids, normals, areas, MIC_POS)

    # 5. 저장
    out = "h_table.npz"
    mesh_bbox_min = verts.min(axis=0) # 메쉬 바운딩 박스 최소 (원점 기준)
    mesh_bbox_max = verts.max(axis=0) # 메쉬 바운딩 박스 최대 (원점 기준)
    np.savez_compressed(
        out,
        H=H,
        doa_grid=doa_grid.astype(np.float32),
        freqs=freqs.astype(np.float32),
        mic_pos=MIC_POS.astype(np.float32),
        mesh_bbox_min=mesh_bbox_min.astype(np.float32),
        mesh_bbox_max=mesh_bbox_max.astype(np.float32),
        n_az=N_AZ,
        n_el=N_EL,
        c_sound=C_SOUND,
    )
    print(f"\nSaved: {out} ({os.path.getsize(out)/1024:.1f} KB)")
    print(f"  H shape: {H.shape}  dtype: {H.dtype}")

    # 6. 시각화
    plot_geometry(centroids, MIC_POS, verts, tris)
    plot_h_table(H, freqs, N_AZ, N_EL, MIC_POS.shape[0], mic_pos=MIC_POS)