"""
3-way benchmark timings between warp reduction kernel, CuPy implimentation, and current CPU 
implimentations
"""

#import libraries etc...
from corrcal.linalg import *
from warp_red_kern_r3 import *

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter

import time
import sys
import os

# sys.path.insert(0, os.path.abspath("../cuBLAS/gemm_grouped_batched"))

current_path = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_path, "..", "gemm_grouped_batched"))
sys.path.insert(0, base_dir)

from utils.gridding import *
from utils.simulate_params import *
from utils.zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark


if __name__ == "__main__":
    
    "---------------Simulate Parameters---------------"
    print()
    n_eig = 3
    rows = 10
    cols = 10
    n_ant = rows*cols


    spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp = cp)
    edges_np = spms.edges(rows, cols, use_random=False)
    # print(edges_np.dtype)


    #Generate the noise and diffuse matrices
    sim_data = spms.sim_data()
    N_cp = sim_data[0]
    D_cp = sim_data[1]
    edges_cp = cp.asarray(edges_np) #gridding leaves array as numpy if use_random=False


    #inverse noise
    N_inv_cp = 1/N_cp  #don't need anymore in the reduction kernel since the kernel does this

    #numpy data
    N_np = cp.asnumpy(N_cp)
    D_np = cp.asnumpy(D_cp)

    #zeropadded matrices for CuPy
    zp_N_inv_cp, nb, lb = zeroPad(N_cp, edges_cp, return_inv=True, dtype=cp.float32)
    zp_D_cp, nb, lb = zeroPad(D_cp, edges_cp, return_inv=False, dtype=cp.float32)


    "---------------COMPUTATION OF diff.T @ N^-1 @ diff---------------"
    #CuPy results
    def cupy_block_mul(inv_noise, diff):  #cupy computation of diff.T @ N^-1 @ diff
        tmp = inv_noise[..., None] * diff
        res = cp.transpose(diff, [0, 2, 1]) @ tmp
        cp.cuda.Stream.null.synchronize()
        return res
    
    # return out = diff.T @ N^-1 @ diff
    out_cupy = cupy_block_mul(zp_N_inv_cp, zp_D_cp)    

    #corrcal results
    out_corrcal = make_small_blocks_fp32(N_np, D_np, edges_np)

    #reduction kernel results 
    out_redkern = cov_reduce_sym_r3(N_cp, D_cp, edges_cp)

    #CHECK IF 3 DIFFERENT APPROACHES ARE COMPUTING THE SAME THING
    check_match = False
    if check_match:
        print(f"Number of antennas used: {n_ant}")

        print("------------------------------------------------------------")
        if np.allclose(out_corrcal, out_redkern) and np.allclose(out_cupy, out_redkern):
            print('Checking correctness with np.allclose (default params):' \
            '\nCORRECT! -> CorrCal, CuPy, and Warp Reduction Implimentation MATCH')
        else:
            print('Checking correctness with np.allclose (default params):' \
            '\nWARNING!! -> The 3 Implimentations DO NOT MATCH with each other')
        print("------------------------------------------------------------")



    "---------------FULL BENCHMARKING (BENCHMARK PLOT)---------------"
    bmark = True
    if bmark:

        n_eig = 3
        def generate_input_matrices(row_col_list):
            rows = row_col_list[0]
            cols = row_col_list[1]
            n_ant = rows*cols
            print(f"Number of antennas used: {n_ant}")


            spms = SimCorrcalParams(n_ant, n_eig, n_src=1, precision='float32', xp = cp)
            edges_np = spms.edges(rows, cols, use_random=False)

            #Generate the noise and diffuse matrices
            sim_data = spms.sim_data()
            N_cp = sim_data[0]
            D_cp = sim_data[1]
            edges_cp = cp.asarray(edges_np, dtype='int32') #gridding leaves array as numpy if use_random=False

            #inverse noise
            N_inv_cp = 1/N_cp

            #numpy data
            N_np = cp.asnumpy(N_cp)
            D_np = cp.asnumpy(D_cp)

            N_cp = cp.array(N_cp, dtype='float32')
            D_cp = cp.array(D_cp, dtype='float32')

            # print(N_np.dtype)
            # print(D_np.dtype)
            # print(edges_np.dtype)

            # print(N_cp.dtype)
            # print(D_cp.dtype)
            # print(edges_cp.dtype)

            #zeropadded matrices for CuPy
            zp_N_inv_cp, nb, lb = zeroPad(N_cp, edges_cp, return_inv=True, dtype=cp.float32)
            zp_D_cp, nb, lb = zeroPad(D_cp, edges_cp, return_inv=False, dtype=cp.float32)

            return {
                "CuPyNoise" : zp_N_inv_cp,
                "CuPyDiff" : zp_D_cp,
                "CorrCalNoise" : N_np,
                "CorrCalDiff" : D_np,
                "RedKernNoise" : N_cp,
                "RedKernDiff" : D_cp,
                "NumpyEdges" : edges_np,
                "CuPyEdges" : edges_cp,
            }
        
        # out = generate_input_matrices([10, 10])

        def run_one_benchmark(input_mat_dict):
            #cupy times
            cupy_times = benchmark(
                cupy_block_mul,
                (input_mat_dict["CuPyNoise"], 
                 input_mat_dict["CuPyDiff"]),
                n_repeat = 100
                )
            
            cupy_GPU_times = cupy_times.gpu_times
            cupy_avg_t = cp.mean(cupy_GPU_times)
            print(f"cupy gpu times: {cupy_avg_t}")

            #corrcal times
            corrcal_times = benchmark(
                make_small_blocks_fp32, 
                (input_mat_dict["CorrCalNoise"], 
                 input_mat_dict["CorrCalDiff"], 
                 input_mat_dict["NumpyEdges"]),
                n_repeat = 100
                )
            
            corrcal_CPU_times = corrcal_times.cpu_times
            corrcal_avg_t = cp.mean(corrcal_CPU_times)
            print(f"corrcal cpu times: {corrcal_avg_t}")

            #reduction kernel times
            redkern_times = benchmark(
                cov_reduce_sym_r3,
                    (input_mat_dict["RedKernNoise"], 
                    input_mat_dict["RedKernDiff"],
                    input_mat_dict["CuPyEdges"]), 
                    n_repeat = 100
                    )
            
            RedKern_GPU_times = redkern_times.gpu_times
            redkern_avg_t = cp.mean(RedKern_GPU_times)
            print(f"reduction kernel gpu times: {redkern_avg_t}")

            return cupy_avg_t, corrcal_avg_t, redkern_avg_t

        #can run one single benchmark test for a row-col combo like this
        # input_mats = generate_input_matrices([32, 32])
        # run_one_benchmark(input_mats)
        
        #for generating combos of rows and columns for full benchmarking        
        def pop_row_col_input(end_iter_num):
            row_col_list = []
            j = 0
            for i in range(2, end_iter_num):
                j += i
                row_col = [j, j]
                row_col_list.append(row_col)
            return row_col_list

        #generates the full list of benchmark results for an array of antenna sizes
        #be sure to comment back in savefig if want to save the plot
        full_bmark = True
        if full_bmark:
            n_iter = 13
            row_col_inputs = pop_row_col_input(n_iter)
            n_ants = np.zeros(len(row_col_inputs))
            for i, rc in enumerate(row_col_inputs):
                n_ant = rc[0]*rc[1]
                n_ants[i] = n_ant
            print(f"row-col combos used are: {row_col_inputs}")

            test_input = [[10, 10], [20, 20]]

            cupy_times = np.zeros(len(row_col_inputs))
            corrcal_times = np.zeros(len(row_col_inputs))
            redkern_times = np.zeros(len(row_col_inputs))
            
            for i, rc in enumerate(row_col_inputs):
                print(f"on iteration {i}")
                input_mats = generate_input_matrices(rc)
                cupy_t, corrcal_t, redkern_t = run_one_benchmark(input_mats)
                cupy_times[i] = cupy_t
                corrcal_times[i] = corrcal_t
                redkern_times[i] = redkern_t
            
            #CAUTION -- DO NOT SET TO TRUE UNLESS ON CORRECT MACHINE
            cluster = False
            if cluster:
                dir_name = 'cluster_bmark_plots'
                title = r"$\mathbf{On\ Cluster\ \rightarrow\ A40\ GPU}$"
            else:
                dir_name = 'local_bmark_plots'
                title = r"$\mathbf{Ran\ Locally\ \rightarrow\ Mobile\ 4070\ GPU}$"

            file_name = f'n_trials={n_iter-2}'

            #plotting
            plt.rcParams["text.usetex"] = False
            plt.rcParams['axes.labelsize'] = 13
            plt.rcParams['figure.figsize'] = (8, 5)
            plt.rcParams.update({
                "mathtext.fontset": "cm",
                "font.family": "serif",
                })

            fig, ax = plt.subplots()
            ax.loglog(n_ants, cupy_times, '-x', ms = 7,  label = 'cupy')
            ax.loglog(n_ants, corrcal_times, '-p', ms = 7, label = 'corrcal')
            ax.loglog(n_ants, redkern_times, '-D', ms = 6, label = 'reduction kernel')

            ax.xaxis.set_major_locator(FixedLocator(n_ants))
            ax.xaxis.set_major_formatter(
                FixedFormatter([rf"${int(np.sqrt(n))}^2$" for n in n_ants])
            )
            ax.tick_params(axis='both', which='major',
               labelsize=13, length=6, width=1.5)
            
            # ax.tick_params(axis='x', labelrotation=-20)
            
            ax.set_xlabel(r"$\mathbf{Number\ of\ Antennas}$")
            ax.set_ylabel(r"$\mathbf{Time\ (s)}$")
            ax.set_title(title, fontsize='14')
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            
            save_plot = False
            if save_plot:
                plt.savefig(f'{dir_name}/{file_name}.png', format = 'png', dpi = 300, bbox_inches = 'tight')

            plt.show()

    #for better readability in the terminal
    print()
