# This kills any multithreaded activity:
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# All of the rest of the code
import numpy as np
import cupy as cp
import ctypes
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from gpu_corrcal.inverse_covariance import *
from gpu_corrcal.utils.tools import *
from gpu_corrcal.cupy_reference.cupy_inverse_covariance import *
from cupyx.profiler import benchmark
from datetime import datetime
cudart = ctypes.CDLL("libcudart.so")

from corrcal.sparse import *

# ============================================================
# Timing test
# ============================================================

def timing_test(n_eig, n_src, rc_tuple, threads_per_block, seed):
    """Time the GPU kernel."""
    # Generate the test data

    # --------------------------------------------------------
    # Generating test data

    test_data = make_test_data(n_eig, n_src, rc_tuple, seed)
    diffuse = test_data["diffuse"]
    noise = test_data["noise"]
    edges = test_data["edges"]
    source = test_data["source"]
    # print(len(diffuse))

    # CPU data for cpu corrcal
    cpu_diffuse = cp.asnumpy(diffuse).astype(np.float64)
    cpu_noise = cp.asnumpy(noise).astype(np.float64)
    cpu_edges = cp.asnumpy(edges).astype(np.float64)
    cpu_source = cp.asnumpy(source).astype(np.float64)
    # --------------------------------------------------------

    n_rep = 400

    # Benchmark gen_eig kernel ++++++++++++++++++++++++++++++++++++++++++++++
    ws = InvCovWorkspace(diffuse, source, edges)
    times = benchmark(inv_cov,
                      (noise, diffuse, source, edges, ws), 
                      n_repeat=n_rep)
    
    avg_gpu = float(cp.mean(times.gpu_times)) # if want in  microseconds: *1e6
    avg_cpu = float(cp.mean(times.cpu_times))    

    # Benchmark [Custom GPU Routine] -> CuPy ref right now ++++++++++++++++++
    zp_inv_noise, lb, nb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
    zp_diffuse, lb, nb = zeroPad(diffuse, edges, return_inv=False, dtype=cp.float32)
    zp_source, lb, nb = zeroPad(source, edges, return_inv=False, dtype=cp.float32)

    ref_times = benchmark(inverse_covariance, 
                          (zp_inv_noise, zp_diffuse, zp_source, cp, False, True), 
                          n_repeat=n_rep)

    ref_avg_gpu = float(cp.mean(ref_times.gpu_times)) 
    ref_avg_cpu = float(cp.mean(ref_times.cpu_times))

    #CPU ref ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    cpu_sparse_cov = SparseCov(cpu_noise, cpu_source, cpu_diffuse, cpu_edges, n_eig, False)
    
    # Leave in if only want single core --------
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {0})
    # ------------------------------------------

    ref_cpu_times = benchmark(cpu_sparse_cov.inv, 
                          (), 
                          n_repeat=n_rep)

    # Leave in if only want single core --------
    os.sched_setaffinity(0, original_affinity)
    # ------------------------------------------

    # print(70*'-')
    # print(f"printing a single output for more granular cpu times:")
    # print(f"{ref_cpu_times}")

    ref_cpu_avg_gpu = float(cp.mean(ref_cpu_times.gpu_times))
    ref_cpu_avg_cpu = float(cp.mean(ref_cpu_times.cpu_times))

    # n_sym = n_eig * (n_eig + 1) // 2
    # Print out the results
    # ------------------IF USING MU SECONDS-----------------------------------------
    # print(f"  [n_eig={n_eig:>2}  n_src={n_src:>2}]   "
    #         f"Custom Algo: gpu={avg_gpu:>5.1f} us  cpu={avg_cpu:>5.1f} us | "
    #         f"CUPY: gpu={ref_avg_gpu:>7.1f} us  cpu={ref_avg_cpu:>5.1f} us | "
    #         f"CPU: gpu={ref_cpu_avg_gpu:>7.1f} us  cpu={ref_cpu_avg_cpu:>5.1f} us")

    #--------------------IF USING NORMAL SECONDS------------------------------------
    print(f"  [n_eig={n_eig:>2}  n_src={n_src:>2}]   "
            f"Custom Algo: gpu={avg_gpu*1e6:>5.1f} us  cpu={avg_cpu*1e6:>5.1f} us | "
            f"CUPY: gpu={ref_avg_gpu*1e6:>7.1f} us  cpu={ref_avg_cpu*1e6:>7.1f} us | "
            f"CPU: gpu={ref_cpu_avg_gpu*1e6:>7.1f} us  cpu={ref_cpu_avg_cpu*1e6:>7.1f} us  &  cpu_std={cp.std(ref_cpu_times.cpu_times)}")

    # return avg_gpu
    return avg_gpu, ref_avg_gpu, ref_cpu_avg_cpu
# =========================================================================================================


# =========================================================================================================
def time_multiple(rc_tuple, tpb, eig_range, src_range,  random_seed, time_eigs=False, time_srcs=False):
    eig_start, eig_stop = eig_range[0], eig_range[1]
    src_start, src_stop = src_range[0], src_range[1]

    eig, src = 3, 3

    print()
    print("=" * 65)
    print(f"TIMING TESTS -> n_ant = {rc_tuple[0]} * {rc_tuple[1]} = {rc_tuple[0]*rc_tuple[1]}")
    print("=" * 65)

    if time_eigs:
        for n_eig in range(eig_start, eig_stop):
            timing_test(n_eig, src, rc_tuple, tpb, random_seed)
    elif time_srcs:
        for n_src in range(src_start, src_stop):
            timing_test(eig, src, rc_tuple, tpb, random_seed)

    print("=" * 65)
# =========================================================================================================


# =========================================================================================================
#for generating combos of rows and columns for full benchmarking        
def pop_row_col_input(end_iter_num):
    row_col_list = []
    j = 0
    for i in range(2, end_iter_num):
        j += i
        row_col = [j, j]
        row_col_list.append(row_col)
    return row_col_list

# This function generates the same nant inputs as Bobby's paper plot
def pop_row_col_input_tens(iter_num):
    row_col_list = []
    coeff = [1, 2, 3, 5]
    for n in range(iter_num+1):
        for c in coeff:
            ant_val = c*10**(n+1)
            for i in range(int(math.isqrt(ant_val)), 0, -1):
                if ant_val % i == 0:
                    rc_list_pair = [i, ant_val // i]
                    row_col_list.append(rc_list_pair)
                    break

    row_col_list = row_col_list[:]
    return row_col_list

# =========================================================================================================


# =========================================================================================================
def timing_plot_nant_varies(
        n_eig, n_src, n_trials, tpb, random_seed, save_plot=True, cluster=False
        ):

    n_iter = n_trials + 2
    # row_col_inputs = pop_row_col_input(n_iter)
    row_col_inputs = pop_row_col_input_tens(n_trials)
    n_ants = np.zeros(len(row_col_inputs))
    for i, rc in enumerate(row_col_inputs):
        n_ant = rc[0]*rc[1]
        n_ants[i] = n_ant
    print(f"row-col combos used are: {row_col_inputs}")

    test_times = np.zeros(len(row_col_inputs))
    reference_times = np.zeros(len(row_col_inputs))
    cpu_reference_times = np.zeros(len(row_col_inputs))

    for i, rc in enumerate(row_col_inputs):
        print(f"on iteration {i}")
        # test_data = make_test_data(n_eig, rc, random_seed)
        gpu_t, ref_gpu_t, ref_cpu_t = timing_test(n_eig, n_src, rc, tpb, random_seed)
        # gpu_t = timing_test(n_eig, n_src, rc, tpb, random_seed)
        
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        cpu_reference_times[i] = ref_cpu_t

    # Save raw data for later replotting
    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('benchmark_data', exist_ok=True)
    data_file = f'benchmark_data/nant_neig{n_eig}_nsrc{n_src}_{tstamp}.npz'
    np.savez(data_file,
             n_ants=n_ants,
             test_times=test_times,
             reference_times=reference_times,
             cpu_reference_times=cpu_reference_times,
             n_eig=n_eig,
             n_src=n_src)
    print(f"Saved benchmark data to {data_file}")

    # ---------------------------------------------------------------------
    # Returning the correct file name and title (cluster vs Device)
    title_label, file_label = get_machine_label()
    # dir_name = 'full_inv_cov_tests'
    dir_name = 'single_core_test_plots'
    tstamp = datetime.now().strftime("%Y%m%d_%H%M")
    if "5070" in file_label:
        file_name = f'SC_device_var_nant: n_trials={n_iter-2}_device={file_label}_{tstamp}'
    elif "A40" in file_label:
        file_name = f'SC_cluster_var_nant: n_trials={n_iter-2}_device={file_label}_{tstamp}'
    
    title = r"$\mathbf{{{}\ ({}\ Eigenmodes,\ {}\ Sources)}}$".format(title_label, n_eig, n_src)
    
    #plotting
    plt.rcParams["text.usetex"] = False
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['figure.figsize'] = (13, 5)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        })

    fig, ax = plt.subplots()
    ax.loglog(n_ants, test_times, '-P', ms = 9,  label = 'Custom GPU Inverse Cov')
    ax.loglog(n_ants, reference_times, '-p', ms = 9, label = 'CuPy Inverse Cov')
    ax.loglog(n_ants, cpu_reference_times, '-p', ms = 9, label = 'CPU CorrCal Inverse Cov')


    # -------------- Power law best-fit in compute-dominated regime ---------------------------
    # Fit t = A * N^alpha in log-log space for n_ant >= n_ant_cutoff,
    # then anchor the line to pass through the measured data at n_ant_ref.
    n_ant_ref = 1000     # power law line passes through data here
    n_ant_cutoff = 500   # only fit points at or above this (compute-dominated)
 
    fit_colors = [None, None, None]  # grab from the data lines
    for idx_line, line in enumerate(ax.get_lines()[:3]):
        fit_colors[idx_line] = line.get_color()
 
    for k, (times_arr, lbl) in enumerate([
        (test_times,          'Custom'),
        (reference_times,     'CuPy'),
        (cpu_reference_times, 'CPU'),
    ]):
        mask = n_ants >= n_ant_cutoff
        if mask.sum() < 2:
            continue
 
        log_n = np.log10(n_ants[mask])
        log_t = np.log10(times_arr[mask])
        alpha, _ = np.polyfit(log_n, log_t, 1)   # slope from the fit
 
        # Anchor: force line through (n_ant_ref, t_measured_at_ref)
        idx_ref = np.argmin(np.abs(n_ants - n_ant_ref))
        t_ref = times_arr[idx_ref]
        A_anchored = t_ref / (n_ant_ref ** alpha)
 
        n_fit = np.logspace(np.log10(n_ants.min()),
                            np.log10(n_ants.max()), 200)
        t_fit = A_anchored * n_fit ** alpha
 
        ax.loglog(n_fit, t_fit, '--', color=fit_colors[k],
                  linewidth=2.5, alpha=0.55,
                  label=rf'{lbl} fit: $\propto N^{{{alpha:.2f}}}$')

    # ------------------------- CHORD 512-antenna marker ---------------------------------------
    # temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | 
    # temp v temp v temp v temp v temp v temp v temp v temp v temp v temp v temp v temp v temp v 

    # from scipy.interpolate import interp1d
    # f_gpu = interp1d(np.log10(n_ants), np.log10(test_times))
    # t_512 = 10**f_gpu(np.log10(512))

    # ax.plot(512, t_512, 'o', color='red', markersize=5, zorder=5)

    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # ax.plot([512, 512], [ylim[0], t_512], '--', color='red', alpha=0.4, lw=1)
    # ax.plot([xlim[0], 512], [t_512, t_512], '--', color='red', alpha=0.4, lw=1)
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    # ax.text(512 * 1.3, t_512, f'(N_ant=512, {t_512:.0f} μs)', fontsize=10, color='red', va='center')

    # temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ temp ^ 
    # temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | temp | 

    ax.xaxis.set_major_locator(FixedLocator(n_ants))
    # ax.xaxis.set_major_formatter(
    #     FixedFormatter([rf"${int(np.sqrt(n))}^2$" for n in n_ants])
    # )
    ax.xaxis.set_major_formatter(
        FixedFormatter([rf"${int(n)}$" for n in n_ants])
    )
    ax.tick_params(axis='both', which='major',
        labelsize=18, length=6, width=1.5)
    
    # ax.tick_params(axis='x', labelrotation=-20)
    
    ax.set_xlabel(r"$\mathbf{Number\ of\ Antennas}$", fontsize=18)
    ax.set_ylabel(r"$\mathbf{Average\ Run\ Time\ (s)}$", fontsize=18)
    ax.set_title(title, fontsize='19')
    ax.grid(axis='y', alpha=0.5)
    ax.legend(fontsize=16)
    
    if save_plot:
        plt.savefig(f'{dir_name}/{file_name}.png', format = 'png', dpi = 300, bbox_inches = 'tight')
 
    # plt.show()
    plt.show()
    # plt.pause(0.1)
# =========================================================================================================


# =========================================================================================================
def replot_nant(npz_path, n_ant_ref=1000, n_ant_cutoff=500, save_plot=True):
    """Replot a varying-nant benchmark from saved .npz data."""
    d = np.load(npz_path)
    n_ants           = d['n_ants']
    test_times       = d['test_times']
    reference_times  = d['reference_times']
    cpu_reference_times = d['cpu_reference_times']
    n_eig, n_src     = int(d['n_eig']), int(d['n_src'])

    # --- plotting (same as before, no GPU needed) ---
    plt.rcParams["text.usetex"] = False
    plt.rcParams['figure.figsize'] = (13, 5)
    plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

    fig, ax = plt.subplots()
    ax.loglog(n_ants, test_times,       '-P', ms=9, label='Custom GPU Inverse Cov')
    ax.loglog(n_ants, reference_times,  '-p', ms=9, label='CuPy Inverse Cov')
    ax.loglog(n_ants, cpu_reference_times, '-p', ms=9, label='CPU CorrCal Inverse Cov')

    # power law fits
    fit_colors = [line.get_color() for line in ax.get_lines()[:3]]
    for k, (times_arr, lbl) in enumerate([
        (test_times, 'Custom'), (reference_times, 'CuPy'), (cpu_reference_times, 'CPU'),
    ]):
        mask = n_ants >= n_ant_cutoff
        if mask.sum() < 2:
            continue
        alpha, _ = np.polyfit(np.log10(n_ants[mask]), np.log10(times_arr[mask]), 1)
        idx_ref = np.argmin(np.abs(n_ants - n_ant_ref))
        A = times_arr[idx_ref] / (n_ant_ref ** alpha)
        n_fit = np.logspace(np.log10(n_ants.min()), np.log10(n_ants.max()), 200)
        t_fit = A * n_fit**alpha
        keep = t_fit >= 10e-6
        n_fit, t_fit = n_fit[keep], t_fit[keep]
        ax.loglog(n_fit, t_fit, '--', color=fit_colors[k],
                  linewidth=2.5, alpha=0.55, label=rf'{lbl} fit: $\propto N^{{{alpha:.2f}}}$')

    ax.xaxis.set_major_locator(FixedLocator(n_ants))
    ax.xaxis.set_major_formatter(FixedFormatter([rf"${int(n)}$" for n in n_ants]))
    ax.tick_params(axis='both', which='major', labelsize=18, length=6, width=1.5)
    ax.set_xlabel(r"$\mathbf{Number\ of\ Antennas}$", fontsize=18)
    ax.set_ylabel(r"$\mathbf{Average\ Run\ Time\ (s)}$", fontsize=18)
    ax.set_title(rf"$\mathbf{{{n_eig}\ Eigenmodes,\ {n_src}\ Sources}}$", fontsize=19)
    ax.grid(axis='y', alpha=0.5)
    ax.legend(fontsize=16)

    if save_plot:
        out = npz_path.replace('.npz', '.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {out}")
    plt.show()
# =========================================================================================================


# =========================================================================================================
def timing_plot_neig_varies(
        rc_tuple, eig_range, n_src, tpb, random_seed, save_plot=True
        ):

    eig_start, eig_stop = eig_range
    neigs = np.arange(eig_start, eig_stop)

    test_times = np.zeros(len(neigs))
    reference_times = np.zeros(len(neigs))
    cpu_reference_times = np.zeros(len(neigs))

    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    for i, n_eig in enumerate(neigs):
        gpu_t, ref_gpu_t, ref_cpu_t = timing_test(n_eig, n_src, rc_tuple, tpb, random_seed)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        cpu_reference_times[i] = ref_cpu_t

    # ---------------------------------------------------------------------
    # Returning the correct file name and title (cluster vs Device)
    title_label, file_label = get_machine_label()
    # dir_name = 'full_inv_cov_tests'
    dir_name = 'test_plots'
    if "5070" in file_label:
        file_name = f'device_var_neig: nant={n_ant}_nsrc={n_src}_neig={eig_start}-{eig_stop}_device={file_label}'
        print(f"file name: {file_label}")
    elif "A40" in file_label:
        file_name = f'cluster_var_neig: nant={n_ant}_nsrc={n_src}_neig={eig_start}-{eig_stop}_device={file_label}'        

    title = r"$\mathbf{{{}\ ({}\ Antennas,\ {}\ Sources)}}$".format(title_label, n_ant, n_src)

    #plotting
    plt.rcParams["text.usetex"] = False
    plt.rcParams['axes.labelsize'] = 13 
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

    fig, ax = plt.subplots()
    ax.semilogy(neigs, test_times, '-P', ms=9, label='Custom GPU Inverse Cov')
    ax.semilogy(neigs, reference_times, '-p', ms=9, label='CuPy Inverse Cov')
    ax.semilogy(neigs, cpu_reference_times, '-p', ms=9, label='CPU CorrCal Inverse Cov')

    ax.xaxis.set_major_locator(FixedLocator(neigs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(n)) for n in neigs]))
    ax.tick_params(axis='both', which='major',
                   labelsize=13, length=6, width=1.5)

    ax.set_xlabel(r"$\mathbf{Number\ of\ Eigenmodes}$")
    ax.set_ylabel(r"$\mathbf{Time\ (\mu s)}$")
    ax.set_title(title, fontsize='14')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    if save_plot:
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f'{dir_name}/{file_name}.png', format='png', dpi=300, bbox_inches='tight')

    # plt.show()
    plt.show(block=False)
    plt.pause(0.1)


def timing_plot_nsrc_varies(
        rc_tuple, src_range, n_eig, tpb, random_seed, save_plot=True
        ):

    src_start, src_stop = src_range
    nsrcs = np.arange(src_start, src_stop)

    test_times = np.zeros(len(nsrcs))
    reference_times = np.zeros(len(nsrcs))
    cpu_reference_times = np.zeros(len(nsrcs))

    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    for i, n_src in enumerate(nsrcs):
        gpu_t, ref_gpu_t, ref_cpu_t = timing_test(n_eig, n_src, rc_tuple, tpb, random_seed)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        cpu_reference_times[i] = ref_cpu_t

    # ---------------------------------------------------------------------
    # Returning the correct file name and title (cluster vs Device)
    title_label, file_label = get_machine_label()
    # dir_name = 'full_inv_cov_tests'
    dir_name = 'test_plots'
    if "5070" in file_label:
        file_name = f'device_var_nsrc: nant={n_ant}_neig={n_eig}_nsrc={src_start}-{src_stop}_device={file_label}'
        print(f"file name: {file_label}")
    elif "A40" in file_label:
        file_name = f'cluster_var_nsrc: nant={n_ant}_neig={n_eig}_nsrc={src_start}-{src_stop}_device={file_label}'        

    title = r"$\mathbf{{{}\ ({}\ Antennas,\ {}\ Eigenmodes)}}$".format(title_label, n_ant, n_eig)

    #plotting
    plt.rcParams["text.usetex"] = False
    plt.rcParams['axes.labelsize'] = 13 
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

    fig, ax = plt.subplots()
    ax.semilogy(nsrcs, test_times, '-P', ms=9, label='Custom GPU Inverse Cov')
    ax.semilogy(nsrcs, reference_times, '-p', ms=9, label='CuPy Inverse Cov')
    ax.semilogy(nsrcs, cpu_reference_times, '-p', ms=9, label='CPU CorrCal Inverse Cov')

    ax.xaxis.set_major_locator(FixedLocator(nsrcs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(n)) for n in nsrcs]))
    ax.tick_params(axis='both', which='major',
                   labelsize=13, length=6, width=1.5)

    ax.set_xlabel(r"$\mathbf{Number\ of\ Sources}$")
    ax.set_ylabel(r"$\mathbf{Time\ (\mu s)}$")
    ax.set_title(title, fontsize='14')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    if save_plot:
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f'{dir_name}/{file_name}.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()


def timing_plot_neig_nsrc_varies(
        rc_tuple, eig_src_range, tpb, random_seed, save_plot=True
        ):

    start, stop = eig_src_range
    vals = np.arange(start, stop, 2)

    test_times = np.zeros(len(vals))
    reference_times = np.zeros(len(vals))
    cpu_reference_times = np.zeros(len(vals))

    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    for i, v in enumerate(vals):
        gpu_t, ref_gpu_t, ref_cpu_t = timing_test(v, v, rc_tuple, tpb, random_seed)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t
        cpu_reference_times[i] = ref_cpu_t

    # ---------------------------------------------------------------------
    title_label, file_label = get_machine_label()
    dir_name = 'test_plots'
    file_name = f'var_neig_nsrc: nant={n_ant}_range={start}-{stop}_device={file_label}'

    title = r"$\mathbf{{{}\ ({}\ Antennas)}}$".format(title_label, n_ant)

    plt.rcParams["text.usetex"] = False
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

    fig, ax = plt.subplots()
    ax.semilogy(vals, test_times, '-P', ms=9, label='Custom GPU Inverse Cov')
    ax.semilogy(vals, reference_times, '-p', ms=9, label='CuPy Inverse Cov')
    ax.semilogy(vals, cpu_reference_times, '-p', ms=9, label='CPU CorrCal Inverse Cov')

    ax.xaxis.set_major_locator(FixedLocator(vals))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in vals]))
    ax.tick_params(axis='both', which='major',
                   labelsize=13, length=6, width=1.5)

    ax.set_xlabel(r"$\mathbf{Number\ of\ Eigenmodes\ and\ Sources}$")
    ax.set_ylabel(r"$\mathbf{Time\ (\mu s)}$")
    ax.set_title(title, fontsize='14')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    if save_plot:
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f'{dir_name}/{file_name}.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()

# =========================================================================================================

if __name__ == "__main__":
    # Test params
    # -------------------------------
    rows = 16
    cols = 32
    n_eig = 3
    n_src = 5
    rc = (rows, cols)
    n_ant = rows*cols
    random_seed=42
    T = True
    F = False
    # -------------------------------


    # Switch board for running tests
    # -------------------------------

    one_timing_test = F
    many_timing_tests = F
    plot_benchmark_nant = T
    plot_benchmark_neig = F
    plot_benchmark_nsrc = F
    plot_benchmark_neig_nsrc = F
    save_plot=False

    if one_timing_test:
        timing_test(n_eig, n_src, rc, 128, random_seed)
    
    if many_timing_tests:
        eig_range = (1, 10)
        src_range = (1, 10)
        time_multiple(rc, 128, eig_range, src_range, random_seed, time_eigs=False, time_srcs=True)
    
    if plot_benchmark_nant:
        n_trials = 2
        timing_plot_nant_varies(n_eig, n_src, n_trials, 128, random_seed, save_plot=save_plot)

    if plot_benchmark_neig:
        eig_range = (1, 20)
        timing_plot_neig_varies(rc, eig_range, n_src, 128, random_seed, save_plot=save_plot)

    if plot_benchmark_nsrc:
        src_range = (1, 20)
        timing_plot_nsrc_varies(rc, src_range, n_eig, 128, random_seed, save_plot=save_plot)

    if plot_benchmark_neig_nsrc:
        eig_src_range = (2, 22)  # 2, 4, 6, 8, ..., 20
        timing_plot_neig_nsrc_varies(rc, eig_src_range, 128, random_seed, save_plot=save_plot)


    # replot_nant('/home/mikej/phd_work/thesis_projects/gpu_corrcal/src/gpu_corrcal/tests/benchmark_data/nant_neig2_nsrc1_20260615_090227.npz')
    # rcl = pop_row_col_input_tens(2)
    # # print(rcl)
    # # # print(f"lenk: {len(rcl)}")

    # for i, rc in enumerate(rcl):
    #     # print(i)
    #     # print(rc)

    #     print(rc[0]*rc[1])