# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    lib = load_kernel()

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
    correctness = F
    one_timing_test = F
    many_timing_tests = F
    plot_benchmark_nant = F
    plot_benchmark_neig = T
    save_plot=False

    if correctness:
        eig_range = (1, 7)   
        test_multiple_correct(lib, rc, eig_range, threads_per_block=128)

    if one_timing_test:
        timing_test(lib, n_eig, rc, 128, random_seed)
    
    if many_timing_tests:
        eig_range = (1, 7)
        time_multiple(lib, rc, 128, eig_range, random_seed)
    
    if plot_benchmark_nant:
        n_trials = 6
        timing_plot_nant_varies(lib, n_eig, n_trials, 128, random_seed, save_plot=save_plot)

    if plot_benchmark_neig:
        eig_range = (1, 7)
        timing_plot_neig_varies(lib, rc, eig_range, 128, random_seed, save_plot=save_plot)
