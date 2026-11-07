"""
CPU (corrcal) vs GPU (gpu_corrcal) calibration benchmark on simulated CHORD-like data.

For each array size, one simulated problem is built (same setup as
calibration_simulation_chord_layout.ipynb) and both backends calibrate the
identical data from the identical initial guess with scipy CG.

Reported per size and backend
-----------------------------
nll, grad   : median wall time of one objective / gradient call at the initial
              guess, after warm-up. Includes the host<->device copies scipy forces.
              For gpu_fused, nll is the one fused (nll, grad) call and grad is blank.
point       : cost of one scipy trial point: nll + grad, or the fused call.
backends    : cpu (CorrCal), gpu (separate nll / grad_nll), gpu_fused (nll_and_grad,
              passed to scipy with jac=True: one call per trial point).
it, t       : iterations and wall seconds to reach the accuracy target (below).
speedup     : CPU time-to-target / GPU time-to-target.
gap         : rms of (GPU - CPU) solution at their target iterates, degenerate modes
              removed. Should be far below the noise-limited error.

Accuracy target (the fair stopping rule)
----------------------------------------
Both backends run scipy CG unchanged; the callback only records a timestamp and a
copy of the gains after every iteration (microseconds). After the solves, every
iterate is compared with the true gains with the degenerate modes removed (overall
amplitude, overall phase, phase gradient). The noise floor is the smallest error any
iterate of either backend reaches; the target is (1 + TARGET_TOL) x floor. A backend's
time-to-target is the timestamp of its first iterate below the target. Both backends
are therefore timed to the same calibration accuracy, independent of where scipy's
own stopping test fires (float32 vs float64 makes that differ).

Usage
-----
python benchmark_calibration.py
python benchmark_calibration.py --sizes grid:8x8,chord --cpu-maxiter 60 --snr 1e4
"""
import argparse
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

from threadpoolctl import threadpool_info, threadpool_limits

import numpy as np
import healpy as hp
from scipy.optimize import minimize
from scipy.special import j1

from pyuvdata import AiryBeam
from corrcal import models as cc_models
from corrcal import utils as cc_utils

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cupy as cp
    HAVE_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp, HAVE_CUPY = None, False

C_LIGHT, K_B = 299792458.0, 1.380649e-23

# CHORD geometry (Ryan, CHORD Localization v0.2)
CHORD_DX = np.array([6.300000, -0.000358, -0.045129])
CHORD_DY = np.array([0.000484, 8.500000, 0.033065])
UTM_MARKER = np.array([309551.114, 5466417.524, 545.454])
UTM_CONVERGENCE_DEG = -1.98814
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "elevation_axis_alignment_points_9Jul2025.csv"


# ---------------------------------------------------------------------------------------------
# Array layout and redundant groups (same as the notebook)
# ---------------------------------------------------------------------------------------------
def load_chord_layout(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = np.array(data["DishID"])
    utm = np.column_stack([0.5 * (data["P_Ex"] + data["P_Wx"]),
                           0.5 * (data["P_Ey"] + data["P_Wy"]),
                           data["Pz"]])
    g = np.deg2rad(UTM_CONVERGENCE_DEG)
    R = np.array([[np.cos(g), np.sin(g), 0.0], [-np.sin(g), np.cos(g), 0.0], [0.0, 0.0, 1.0]])
    topo = (utm - UTM_MARKER) @ R.T
    grid_ij = np.column_stack([[int(n[1:]) - 1 for n in names], [ord(n[0]) - ord("A") for n in names]])
    return topo, grid_ij


def make_grid(nx, ny):
    grid_ij = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), -1).reshape(-1, 2)
    return (grid_ij[:, :1] * CHORD_DX + grid_ij[:, 1:] * CHORD_DY).astype(float), grid_ij


def make_groups(antpos, grid_ij, min_len, min_size):
    a, b = np.triu_indices(len(antpos), 1)
    off = (grid_ij[b] - grid_ij[a]).astype(np.int64)
    flip = (off[:, 1] < 0) | ((off[:, 1] == 0) & (off[:, 0] < 0))
    off[flip] *= -1
    a, b = np.where(flip, b, a), np.where(flip, a, b)
    keys, inv, counts = np.unique(off, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    ubl = np.zeros((len(keys), 3))
    np.add.at(ubl, inv, antpos[b] - antpos[a])
    ubl /= counts[:, None]
    length = np.linalg.norm(ubl, axis=1)
    kept = np.flatnonzero((length >= min_len) & (counts >= min_size))
    kept = kept[np.argsort(length[kept], kind="stable")]
    rank = np.full(len(keys), -1)
    rank[kept] = np.arange(len(kept))
    sel = rank[inv] >= 0
    order = np.argsort(rank[inv][sel], kind="stable")
    ant_1, ant_2 = a[sel][order], b[sel][order]
    grp_of_bl = rank[inv][sel][order]
    edges = 2 * np.concatenate([[0], np.cumsum(counts[kept])])
    ubl = ubl[kept]
    spread = np.linalg.norm(antpos[ant_2] - antpos[ant_1] - ubl[grp_of_bl], axis=1).max()
    return ant_1, ant_2, edges, ubl, grp_of_bl, spread


def split_re_im(z):
    out = np.empty(2 * z.size)
    out[0::2], out[1::2] = z.real, z.imag
    return out


# ---------------------------------------------------------------------------------------------
# Problem setup: sky, visibilities, model covariance, gains, data
# ---------------------------------------------------------------------------------------------
def build_problem(layout, cfg, rng):
    t_setup = time.perf_counter()
    lam = C_LIGHT / cfg.freq
    if layout == "chord":
        antpos, grid_ij = load_chord_layout(cfg.chord_csv)
    else:
        nx, ny = (int(v) for v in layout.split(":")[1].split("x"))
        antpos, grid_ij = make_grid(nx, ny)
    antpos = antpos - antpos[np.lexsort((grid_ij[:, 1], grid_ij[:, 0]))[0]]

    ant_1, ant_2, edges, ubl, grp_of_bl, spread = make_groups(
        antpos, grid_ij, np.sqrt(2) * cfg.dish_d, cfg.min_group_size)
    if spread > 1e-3 * lam:
        raise ValueError(f"{layout}: layout not exactly redundant (spread {spread:.2e} m)")
    n_ant, n_bl, n_grp = len(antpos), len(ant_1), len(ubl)

    # Sky: GRF diffuse map + power-law point sources in a zenith cap.
    b_max = np.linalg.norm(ubl, axis=1).max()
    nside_need = 1 << int(np.ceil(np.log2(np.sqrt(4 * np.pi / 12) / (lam / (2 * b_max)))))
    nside = min(nside_need, cfg.nside_max)
    lmax_map = 3 * nside - 1
    jy_per_k = 2 * K_B / lam**2 * 1e26

    def airy_power(za):
        x = np.pi * cfg.dish_d * np.sin(za) / lam
        safe = np.where(x > 1e-12, x, 1.0)
        return np.where(x > 1e-12, (2 * j1(safe) / safe) ** 2, 1.0)

    cap = hp.query_disc(nside, [0.0, 0.0, 1.0], np.deg2rad(cfg.max_za_deg))
    za, az = hp.pix2ang(nside, cap)
    pix_nhat = np.array([np.sin(za) * np.cos(az), np.sin(za) * np.sin(az), np.cos(za)])
    pix_beam = airy_power(za)
    ells = np.arange(lmax_map + 1.0)
    cl_shape = 1.0 / (1.0 + ells) ** 2
    cl_shape[0] = 0.0
    cl_K = cl_shape * cfg.t_rms**2 / np.sum((2 * ells + 1) / (4 * np.pi) * cl_shape)
    np.random.seed(cfg.seed)                                   # healpy.synfast uses the global RNG
    T_map = hp.synfast(cl_K, nside)

    cos_za = rng.uniform(np.cos(np.deg2rad(cfg.max_za_deg)), 1.0, cfg.n_src_total)
    s_za, s_az = np.arccos(cos_za), rng.uniform(0, 2 * np.pi, cfg.n_src_total)
    src_nhat = np.array([np.sin(s_za) * np.cos(s_az), np.sin(s_za) * np.sin(s_az), np.cos(s_za)])
    src_app = airy_power(s_za) * cfg.s_min * rng.uniform(size=cfg.n_src_total) ** (-1.0 / (cfg.src_slope - 1.0))
    cal_idx = np.argsort(src_app)[::-1][:cfg.n_src_cal]

    def direct_sum(bl, nhat, w, chunk_elems=2 ** 23):
        xp = cp if HAVE_CUPY else np
        chunk = max(256, chunk_elems // max(len(bl), 1))
        bl_l = xp.asarray(bl / lam)
        V = xp.zeros(len(bl), dtype=complex)
        for s in range(0, w.size, chunk):
            V += xp.exp(-2j * np.pi * (bl_l @ xp.asarray(nhat[:, s:s + chunk]))) @ xp.asarray(w[s:s + chunk])
        return cp.asnumpy(V) if HAVE_CUPY else V

    pix_area = hp.nside2pixarea(nside)
    V_sky = direct_sum(ubl, pix_nhat, pix_beam * T_map[cap] * jy_per_k * pix_area)
    V_sky = (V_sky + direct_sum(ubl, src_nhat, src_app))[grp_of_bl]
    V_cal = np.stack([direct_sum(ubl, src_nhat[:, [j]], src_app[[j]])[grp_of_bl] for j in cal_idx], axis=1)
    V_auto = np.sum(pix_beam * (T_map[cap] + cfg.t_mean)) * jy_per_k * pix_area + src_app.sum()

    # Model covariance: Delta (one representative baseline per exactly redundant group), Sigma, N = I.
    lmax_cov = int(np.ceil(2 * np.pi * b_max / lam * 1.3)) + 50
    cl_jy = np.zeros(max(lmax_cov, lmax_map) + 1)
    cl_jy[:lmax_map + 1] = cl_K * jy_per_k**2
    tail = np.arange(lmax_map + 1, cl_jy.size)
    cl_jy[tail] = cl_jy[lmax_map] * ((1.0 + lmax_map) / (1.0 + tail)) ** 2
    reps = edges[:-1] // 2
    diff_one = cc_models._compute_diffuse_matrix_from_flat_sky(
        cl_jy, 64, AiryBeam(diameter=cfg.dish_d), cfg.freq, antpos, ant_1[reps], ant_2[reps],
        2 * np.arange(n_grp + 1), n_eig=1, bl_convention="j-i")
    rows = np.repeat(np.arange(n_grp), np.diff(edges) // 2)
    diff_mat = np.empty((edges[-1], diff_one.shape[1]))
    diff_mat[0::2], diff_mat[1::2] = diff_one[2 * rows], diff_one[2 * rows + 1]
    src_mat = np.stack([split_re_im(V_cal[:, j]) for j in range(len(cal_idx))], axis=1)

    sigma_row = (V_auto / cfg.snr) / np.sqrt(2)
    diff_mat /= sigma_row
    src_mat /= sigma_row
    noise = np.ones(edges[-1])

    # Gains and data.
    amp = rng.normal(1.0, cfg.amp_jitter, n_ant)
    phs = rng.uniform(0, 2 * np.pi, n_ant)
    true_gains = amp * np.exp(1j * phs)
    init_gains = amp * rng.normal(1.0, cfg.init_amp_err, n_ant) * \
        np.exp(1j * (phs + rng.normal(0, cfg.init_phs_err, n_ant)))
    sky_c = V_sky / sigma_row
    data = split_re_im(true_gains[ant_1] * np.conj(true_gains[ant_2]) * sky_c) + rng.standard_normal(2 * n_bl)

    return dict(layout=layout, n_ant=n_ant, n_bl=n_bl, n_grp=n_grp, n_row=int(edges[-1]),
                nside=nside, nside_need=nside_need, antpos=antpos, ant_1=ant_1, ant_2=ant_2,
                edges=edges, noise=noise, diff_mat=diff_mat, src_mat=src_mat, data=data,
                true_gains=true_gains, init_gains=init_gains,
                setup_s=time.perf_counter() - t_setup)


# ---------------------------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------------------------
def build_objective(backend, P, cfg):
    if backend == "gpu":
        from gpu_corrcal.sparse import GpuSparseCov
        from gpu_corrcal.optimize import nll, grad_nll
        from gpu_corrcal.linalg import GradWorkspace
        cov = GpuSparseCov(P["noise"], P["src_mat"], P["diff_mat"], P["edges"], n_eig=P["diff_mat"].shape[1])
        d = cp.ascontiguousarray(cp.asarray(P["data"], dtype=cp.float32))
        a1 = cp.asarray(P["ant_1"], dtype=cp.int32)
        a2 = cp.asarray(P["ant_2"], dtype=cp.int32)
        gws = GradWorkspace(a1, a2, P["n_ant"], cov.n_grp, cov.n_eig, cov.n_src)
        to_host = lambda x: cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)
        fun = lambda g: float(nll(g, cov, d, a1, a2, 1.0, cfg.phs_norm))
        jac = lambda g: to_host(grad_nll(g, cov, d, a1, a2, 1.0, cfg.phs_norm, gws)).astype(np.float64)
    elif backend == "gpu_fused":
        from gpu_corrcal.sparse import GpuSparseCov
        from gpu_corrcal.optimize import nll_and_grad
        from gpu_corrcal.linalg import GradWorkspace, NllWorkspace
        cov = GpuSparseCov(P["noise"], P["src_mat"], P["diff_mat"], P["edges"], n_eig=P["diff_mat"].shape[1])
        d = cp.ascontiguousarray(cp.asarray(P["data"], dtype=cp.float32))
        a1 = cp.asarray(P["ant_1"], dtype=cp.int32)
        a2 = cp.asarray(P["ant_2"], dtype=cp.int32)
        gws = GradWorkspace(a1, a2, P["n_ant"], cov.n_grp, cov.n_eig, cov.n_src)
        nws = NllWorkspace()
        fun = lambda g: nll_and_grad(g, cov, d, a1, a2, 1.0, cfg.phs_norm, gws, nws)
        jac = True     # scipy: fun returns (f, grad)
    elif backend == "cpu":
        from corrcal.sparse import SparseCov
        from corrcal.optimize import nll, grad_nll
        cov = SparseCov(noise=P["noise"], src_mat=P["src_mat"], diff_mat=P["diff_mat"], edges=P["edges"],
                        n_eig=P["diff_mat"].shape[1], isinv=False)
        a1, a2 = P["ant_1"].astype(np.int64), P["ant_2"].astype(np.int64)
        fun = lambda g: nll(g, cov, P["data"], a1, a2, 1.0, cfg.phs_norm)
        jac = lambda g: grad_nll(g, cov, P["data"], a1, a2, 1.0, cfg.phs_norm)
    else:
        raise ValueError(backend)
    return fun, jac


def time_call(f, x, repeats, warmup_s=0.5):
    # Warm-up: CuPy kernel compilation, memory pool, and GPU clocks (a laptop GPU idles at a low
    # clock during the CPU-side setup; timing it cold inflates the first function timed).
    f(x)
    t_end = time.perf_counter() + warmup_s
    while time.perf_counter() < t_end:
        f(x)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def solve_traced(fun, jac, x0, maxiter):
    """scipy CG with a timestamp + copy of the gains after every iteration (iterate 0 = x0 at t = 0)."""
    xs, ts = [x0.copy()], [0.0]
    t0 = time.perf_counter()
    res = minimize(fun, x0, jac=jac, method="CG", options={"maxiter": maxiter},
                   callback=lambda xk: (xs.append(xk.copy()), ts.append(time.perf_counter() - t0)))
    return np.array(xs), np.array(ts), res, time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------
# Accuracy against truth, degenerate modes removed
# ---------------------------------------------------------------------------------------------
def nondegenerate_error(x, ref_gains, antpos):
    """rms complex fractional gain error after removing overall amplitude, overall phase and phase gradient."""
    g = x[0::2] + 1j * x[1::2]
    r = cc_utils.rephase_to_ant(g, 0) / cc_utils.rephase_to_ant(ref_gains, 0)
    la, ph = np.log(np.abs(r)), np.angle(r)
    A = np.column_stack([np.ones(len(r)), antpos[:, 0], antpos[:, 1]])
    coef, *_ = np.linalg.lstsq(A, ph, rcond=None)
    amp_res = la - la.mean()
    phs_res = ph - A @ coef
    return np.sqrt(np.mean(amp_res ** 2 + phs_res ** 2))


def run_size(layout, cfg, backends):
    rng = np.random.default_rng(cfg.seed)
    P = build_problem(layout, cfg, rng)
    print(f"\n=== {layout}: {P['n_ant']} antennas, {P['n_bl']:,} baselines, {P['n_grp']} groups, "
          f"{P['n_row']:,} rows, nside {P['nside']}"
          + (f" (capped from {P['nside_need']})" if P["nside"] < P["nside_need"] else "")
          + f", setup {P['setup_s']:.1f} s")
    x0 = split_re_im(P["init_gains"])
    out = {}
    for b in backends:
        fun, jac = build_objective(b, P, cfg)
        # Thread cap for the CPU backend: CorrCal's OpenMP loops and NumPy/SciPy BLAS.
        # Applied after build_objective so the CorrCal extension is already loaded.
        limit = threadpool_limits(limits=cfg.cpu_threads) if (b == "cpu" and cfg.cpu_threads > 0) else nullcontext()
        with limit:
            if b == "cpu":
                pools = ", ".join(f"{p['internal_api']}={p['num_threads']}" for p in threadpool_info())
                print(f"  [cpu] thread pools: {pools}")
            fused = jac is True
            t_nll = time_call(fun, x0, cfg.repeats)
            t_grad = np.nan if fused else time_call(jac, x0, cfg.repeats)
            t_point = t_nll if fused else t_nll + t_grad
            maxiter = cfg.cpu_maxiter if b == "cpu" else cfg.gpu_maxiter
            xs, ts, res, wall = solve_traced(fun, jac, x0, maxiter)
        err = np.array([nondegenerate_error(x, P["true_gains"], P["antpos"]) for x in xs])
        # With jac=True scipy's njev counts gradients it read from the fused
        # call's cache, not separate calls: the objective ran nfev times.
        n_calls = res.nfev if fused else res.nfev + res.njev
        out[b] = dict(t_nll=t_nll, t_grad=t_grad, t_point=t_point, xs=xs, ts=ts, err=err, nit=res.nit,
                      nfev=res.nfev, njev=res.njev, n_calls=n_calls, wall=wall, msg=res.message)
        times = (f"nll+grad {1e3 * t_nll:9.3f} ms" if fused else
                 f"nll {1e3 * t_nll:9.3f} ms   grad {1e3 * t_grad:9.3f} ms")
        calls = (f"{res.nfev} fused calls" if fused else
                 f"{res.nfev} nll + {res.njev} grad calls")
        print(f"  [{b}] {times}   point {1e3 * t_point:9.3f} ms   "
              f"full solve: {res.nit} it, {calls}, {wall:.2f} s "
              f"({1e3 * wall / n_calls:.3f} ms/call avg) ({res.message})")

    floor = min(o["err"].min() for o in out.values())
    target = (1 + cfg.target_tol) * floor
    for b, o in out.items():
        hit = np.flatnonzero(o["err"] <= target)
        o["it_target"] = int(hit[0]) if hit.size else -1
        o["t_target"] = float(o["ts"][hit[0]]) if hit.size else np.nan
    row = dict(layout=layout, n_ant=P["n_ant"], n_row=P["n_row"], floor=floor, target=target,
               **{f"{b}_{k}": out[b][k] for b in out for k in ("t_nll", "t_grad", "t_point", "it_target",
                                                                  "t_target", "nit", "wall")})
    # gap_<a>_<b>: rms(a - b) at their target iterates, degenerate modes removed
    for a_, b_ in (("gpu", "cpu"), ("gpu_fused", "cpu"), ("gpu_fused", "gpu")):
        if a_ in out and b_ in out and out[a_]["it_target"] >= 0 and out[b_]["it_target"] >= 0:
            x_a = out[a_]["xs"][out[a_]["it_target"]]
            x_b = out[b_]["xs"][out[b_]["it_target"]]
            row[f"gap_{a_}_{b_}"] = nondegenerate_error(x_a, x_b[0::2] + 1j * x_b[1::2], P["antpos"])
    print(f"  noise floor {floor:.3e}, target {target:.3e} (+{100 * cfg.target_tol:.0f}%)")
    for b, o in out.items():
        print(f"  [{b}] reaches target at iteration {o['it_target']}, {o['t_target']:.3f} s")
    for k in [k for k in row if k.startswith("gap_")]:
        a_, b_ = k[4:].replace("gpu_fused", "gpu-fused").split("_")
        print(f"  {a_} vs {b_} at target: {row[k]:.2e}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="grid:8x8,grid:16x16,chord,grid:24x32",
                    help="comma list of grid:NXxNY or chord")
    ap.add_argument("--backends", default=None,
                    help="comma list of gpu, gpu_fused, cpu; default all three (cpu only without a GPU)")
    ap.add_argument("--chord-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--snr", type=float, default=1e4)
    ap.add_argument("--n-src-cal", type=int, default=5)
    ap.add_argument("--cpu-maxiter", type=int, default=60,
                    help="the non-degenerate error converges in <~10 iterations; later CPU iterations only move "
                         "degenerate modes, so this cap only saves time")
    ap.add_argument("--gpu-maxiter", type=int, default=2000)
    ap.add_argument("--cpu-threads", type=int, default=1,
                    help="threads for the CPU backend (OpenMP + BLAS) during timing and solve; 0 = no limit")
    ap.add_argument("--repeats", type=int, default=20, help="timed repeats per call (median reported)")
    ap.add_argument("--target-tol", type=float, default=0.05)
    ap.add_argument("--nside-max", type=int, default=1024,
                    help="caps the sky resolution for large arrays (affects data realism only, not timing)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("benchmark_calibration.npz"))
    cfg = ap.parse_args()
    # fixed simulation settings (as in the notebook)
    cfg.freq, cfg.dish_d, cfg.min_group_size, cfg.max_za_deg = 600e6, 6.0, 5, 40.0
    cfg.t_mean, cfg.t_rms, cfg.n_src_total, cfg.s_min, cfg.src_slope = 30.0, 5.0, 2000, 1.0, 2.5
    cfg.amp_jitter, cfg.init_amp_err, cfg.init_phs_err, cfg.phs_norm = 0.1, 0.1, 0.1, 0.1

    backends = (cfg.backends.split(",") if cfg.backends
                else (["gpu", "gpu_fused", "cpu"] if HAVE_CUPY else ["cpu"]))
    rows = [run_size(s.strip(), cfg, backends) for s in cfg.sizes.split(",")]

    print("\n" + "=" * 110)
    gpu_bs = [b for b in backends if b != "cpu"] if "cpu" in backends else []
    hdr = f"{'layout':>12} {'n_ant':>6} {'rows':>9} |"
    for b in backends:
        hdr += f" {b + ' point[ms]':>19} {b + ' it':>12} {b + ' t[s]':>14} |"
    for b in gpu_bs:
        hdr += f" {'speedup ' + b:>17} {'gap ' + b:>13}"
    print(hdr)
    for r in rows:
        line = f"{r['layout']:>12} {r['n_ant']:>6} {r['n_row']:>9,} |"
        for b in backends:
            line += (f" {1e3 * r[b + '_t_point']:>19.3f} {r[b + '_it_target']:>12} "
                     f"{r[b + '_t_target']:>14.3f} |")
        for b in gpu_bs:
            sp = r["cpu_t_target"] / r[b + "_t_target"]
            line += f" {sp:>17.1f} {r.get(f'gap_{b}_cpu', np.nan):>13.2e}"
        print(line)
    print("point = one scipy trial point (nll + grad, or one fused call); "
          "speedup = CPU time-to-target / GPU time-to-target; "
          "gap = rms(GPU - CPU) at target, degenerate modes removed")

    np.savez(cfg.out, rows=np.array(rows, dtype=object), backends=np.array(backends),
             config=np.array(vars(cfg), dtype=object))
    print(f"saved {cfg.out}")


if __name__ == "__main__":
    main()