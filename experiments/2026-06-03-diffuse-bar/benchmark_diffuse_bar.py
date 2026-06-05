import cupy as cp
import ctypes
from pathlib import Path
from gpu_corrcal.inverse_covariance import *
from gpu_corrcal.utils.gridding import *
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.zp_puregpu_funcs_py import *
from gpu_corrcal.utils.tools import *
from gpu_corrcal.cupy_reference.cupy_inverse_covariance import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from cupyx.profiler import benchmark


lib_dir = Path(__file__).resolve().parent

#change this if more than one .so file in this parent directory
file_path = list(lib_dir.glob("*.so"))[0]

lib = ctypes.cdll.LoadLibrary(file_path)

mul_temp_by_diffuse_chol_inv = lib.launch_mul_temp_by_chol
mul_temp_by_diffuse_chol_inv.argtypes = [
    ctypes.c_void_p,   #noise input
    ctypes.c_void_p,   #diffuse input
    ctypes.c_void_p,   #diffuse chol inv T mat
    ctypes.c_void_p,   #output mat    
    ctypes.c_void_p,      #edges
    ctypes.c_int,      #num blocks
    ctypes.c_int       #n_eig
]
def mul_temp_by_diffuse_chol(noise, diffuse, L_inv_T, out, edges):
    n_eig = diffuse.shape[1]
    nb = len(edges) - 1
    mul_temp_by_diffuse_chol_inv(
        noise.data.ptr,
        diffuse.data.ptr,
        L_inv_T.data.ptr,
        out.data.ptr,
        edges.data.ptr,
        nb,
        n_eig
    )
    return out

# ============================================================
# Timing test
# ============================================================

def timing_test(
        n_eig, rc_tuple, threads_per_block, seed,
        ):
    """Time the GPU kernel."""
    # Generate the test data

    test_data = make_test_data(n_eig, rc_tuple, seed)
    diffuse = test_data["diffuse"]
    noise = test_data["noise"]
    edges = test_data["edges"]
    source = test_data["source"]

    # ------------------------------------------------------
    # Benchmark [fused cholesky inverse kernel]
    ws = InvCovWorkspace(diffuse, source, edges)
    L_del_inv_T = inv_cov(noise, diffuse, source, edges, ws)
    sync()
    times = benchmark(mul_temp_by_diffuse_chol, 
                      (noise, diffuse, L_del_inv_T, ws.diffuse_bar, edges), 
                      n_repeat= 1000)
    # -------------------------------------------------------

    avg_gpu = float(cp.mean(times.gpu_times)) * 1e6  # microseconds
    avg_cpu = float(cp.mean(times.cpu_times)) * 1e6
    

    # Benchmark [INSERT REF FUNCTION] -> CuPy ref right now
    temp, t3 = setup_cupy_ref(noise, diffuse, edges)

    ref_times = benchmark(cupy_ref, 
                          (temp, edges, L_del_inv_T), 
                          n_repeat= 1000)
    
    ref_avg_gpu = float(cp.mean(ref_times.gpu_times)) * 1e6  # microseconds
    ref_avg_cpu = float(cp.mean(ref_times.cpu_times)) * 1e6


    n_sym = n_eig * (n_eig + 1) // 2
    # Print out the results
    print(f"  [n_eig={n_eig}  n_sym={n_sym:>3}]   "
            f"KERNEL: gpu={avg_gpu:>5.1f} us  cpu={avg_cpu:>5.1f} us | "
            f"CUPY: gpu={ref_avg_gpu:>7.1f} us  cpu={ref_avg_cpu:>5.1f} us")
    
    return avg_gpu, ref_avg_gpu
# =========================================================================================================

# =========================================================================================================
def time_multiple(rc_tuple, tpb, eig_range, random_seed,
                  kenel_fun=None, kernel_params=None
                  ):
    eig_start, eig_stop = eig_range[0], eig_range[1]

    print()
    print("=" * 65)
    print(f"TIMING TESTS -> n_ant = {rc_tuple[0]} * {rc_tuple[1]} = {rc_tuple[0]*rc_tuple[1]}")
    print("=" * 65)

    for n_eig in range(eig_start, eig_stop):
        timing_test(n_eig, rc_tuple, tpb, random_seed)

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
# =========================================================================================================

# =========================================================================================================
def timing_plot_nant_varies(
        n_eig, n_trials, tpb, random_seed,
        kenel_fun=None, kernel_params=None,
        save_plot=True
        ):

    n_iter = n_trials + 2
    row_col_inputs = pop_row_col_input(n_iter)
    n_ants = np.zeros(len(row_col_inputs))
    for i, rc in enumerate(row_col_inputs):
        n_ant = rc[0]*rc[1]
        n_ants[i] = n_ant
    print(f"row-col combos used are: {row_col_inputs}")

    test_times = np.zeros(len(row_col_inputs))
    reference_times = np.zeros(len(row_col_inputs))

    for i, rc in enumerate(row_col_inputs):
        print(f"on iteration {i}")
        # test_data = make_test_data(n_eig, rc, random_seed)
        gpu_t, ref_gpu_t = timing_test(n_eig, rc, tpb, random_seed)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t

    # ---------------------------------------------------------------------
    # Returning the correct file name and title (cluster vs Device)
    title_label, file_label = get_machine_label()
    dir_name = 'bmark_plots'
    file_name = f'cluster_var_nant: n_trials={n_iter-2}_device={file_label}'
    title = r"$\mathbf{{{}\ ({}\ Eigenmodes)}}$".format(title_label, n_eig)
    
    #plotting
    plt.rcParams["text.usetex"] = False
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        })

    fig, ax = plt.subplots()
    ax.loglog(n_ants, test_times, '-P', ms = 9,  label = 'fused cholesky inv')
    ax.loglog(n_ants, reference_times, '-p', ms = 9, label = 'CuPy')
    
    ax.xaxis.set_major_locator(FixedLocator(n_ants))
    ax.xaxis.set_major_formatter(
        FixedFormatter([rf"${int(np.sqrt(n))}^2$" for n in n_ants])
    )
    ax.tick_params(axis='both', which='major',
        labelsize=13, length=6, width=1.5)
    
    # ax.tick_params(axis='x', labelrotation=-20)
    
    ax.set_xlabel(r"$\mathbf{Number\ of\ Antennas}$")
    ax.set_ylabel(r"$\mathbf{Time\ (\mu s)}$")
    ax.set_title(title, fontsize='14')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    if save_plot:
        plt.savefig(f'{dir_name}/{file_name}.png', format = 'png', dpi = 300, bbox_inches = 'tight')
 
    plt.show()
# =========================================================================================================

# =========================================================================================================
def timing_plot_neig_varies(
        rc_tuple, eig_range, tpb, random_seed, 
        kenel_fun=None, kernel_params=None,
        save_plot=True
        ):

    eig_start, eig_stop = eig_range
    neigs = np.arange(eig_start, eig_stop)

    test_times = np.zeros(len(neigs))
    reference_times = np.zeros(len(neigs))

    n_ant = rc_tuple[0] * rc_tuple[1]
    print(f"n_ant fixed at {n_ant}")

    for i, n_eig in enumerate(neigs):
        gpu_t, ref_gpu_t = timing_test(n_eig, rc_tuple, tpb, random_seed)
        test_times[i] = gpu_t
        reference_times[i] = ref_gpu_t

    # ---------------------------------------------------------------------
    # Returning the correct file name and title (cluster vs Device)
    title_label, file_label = get_machine_label()
    dir_name = 'bmark_plots'
    file_name = f'var_neig: nant={n_ant}_neig={eig_start}-{eig_stop}_device={file_label}'
    title = r"$\mathbf{{{}\ ({}\ Antennas)}}$".format(title_label, n_ant)


    plt.rcParams["text.usetex"] = False
    plt.rcParams['axes.labelsize'] = 13 
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

    fig, ax = plt.subplots()
    ax.semilogy(neigs, test_times, '-x', ms=7, label='Reduction Kernel')
    ax.semilogy(neigs, reference_times, '-p', ms=7, label='CuPy')

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

    plt.show()

# =========================================================================================================

if __name__ == "__main__":
    # Test params
    # -------------------------------
    rows = 32
    cols = 18
    n_eig = 3
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
    plot_benchmark_nant = F
    plot_benchmark_neig = T
    save_plot=True

    if one_timing_test:
        timing_test(n_eig, rc, 128, random_seed)
    
    if many_timing_tests:
        eig_range = (1, 10)
        time_multiple(rc, 128, eig_range, random_seed)
    
    if plot_benchmark_nant:
        n_trials = 10
        timing_plot_nant_varies(n_eig, n_trials, 128, random_seed, save_plot=save_plot)

    if plot_benchmark_neig:
        eig_range = (1, 21)
        timing_plot_neig_varies(rc, eig_range, 128, random_seed, save_plot=save_plot)
