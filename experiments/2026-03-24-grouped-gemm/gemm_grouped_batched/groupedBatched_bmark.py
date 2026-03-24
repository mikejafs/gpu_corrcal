import cupy as cp
import numpy as np
from simulate_params import *
from groupedBatched_funcv2 import *
from zp_puregpu_funcs_py import *
from cupyx.profiler import benchmark
import sys

save_to_results = False

if save_to_results:
    sys.stdout = open("bmark_results.txt", "a")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()
print('NEW RESULT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print()


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SET UP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

#Parameter set up
ant_dim_x = 18
ant_dim_y = 32
n_eig = 5
n_src = 1
# n_ant = 500
n_ant = ant_dim_x*ant_dim_y

print('Parameters:')
print('Number of antennas (Re Im split):', n_ant,
      '\nNumber of eigenmodes:', n_eig,
      '\nNumbe of sources:', n_src)
print()

cp.random.seed(10)
spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
edges = spms.edges(ant_dim_x, ant_dim_y, use_random=False)
# print(edges)

#simulated matrices with correct shapes
sim_data = spms.sim_data()
noise = sim_data[0]
diff = sim_data[1]
# print(len(diff))

#zeropad diff and noise 
zp_noise, nb, lb = zeroPad(noise, edges, return_inv=True, dtype=cp.float32)
zp_diff, nb, lb = zeroPad(diff, edges, return_inv=False, dtype=cp.float32)


#need this if wanting to compare to zped stuff since the true matmul is diff.T@N^-1@diff
noise = 1/noise


# print(max([(edges[i+1] - edges[i]) for i in range(len(edges)-1)]))
# print(min([(edges[i+1] - edges[i]) for i in range(len(edges)-1)]))



"""~~~~~~~~~~~~~~~~~~~~~ COMPUTATION OF diff.T @ N^-1 @ diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

#set up the temp mat just before the mat mul we are interested in

#cublas temp (no zeropadding)
temp = noise[..., None] * diff

#zeropadded temp
zp_temp = zp_noise[..., None] * zp_diff


#cupy computation of diff.T @ N^-1 @ diff
def cupy_block_mul(diff, tmp):
    res = cp.transpose(diff, [0, 2, 1]) @ tmp
    cp.cuda.Stream.null.synchronize()
    return res

#return temp2 = diff.T @ N^-1 @ diff
temp2 = cupy_block_mul(zp_diff, zp_temp)
# print(temp2)

#cublas compuation of diff.T @ N^-1 @ diff
grouped_batched_param_dict = prepare_grouped_batched_params(diff, temp, edges) 
out_ptr = groupedBatchedMatmul(grouped_batched_param_dict)


check_match = False
if check_match:
    out = reshape_out(out_ptr, edges)
    # print(out)
    print()
    print('cuBLAS output has dtype:', cp.dtype(out))
    print()

    #CHECK IF CUPY AND CUBLAS ARE COMPUTING THE SAME THING
    if np.allclose(temp2, out):
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and cuBLAS match')
    else:
        print('Checking correctness with np.allclose (default params):' \
        '\nCuPy and cuBLAS DO NOT match')

    print(temp2[0])
    print(out[0])


"""~~~~~~~~~~~~~~~~~~~~~ BENCHMARKING CUPY AND CUBLAS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

#CUPY TIMES
times = (benchmark(cupy_block_mul, (zp_diff, zp_temp), n_repeat = 100))
# gpu_times = gpu_times.split()
# gpu_cpu_t = float(gpu_times[3])/1e6
# gpu_gpu_t = float(gpu_times[14])/1e6

gpu_t_s = times.gpu_times
cpu_t_s = times.cpu_times

avg_gpu_t = cp.mean(gpu_t_s)
avg_cpu_t = cp.mean(cpu_t_s)

# print(gpu_cpu_t, gpu_gpu_t)
print()
print('CuPy times:')
print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)

#CUBLAS TIMES
times = (benchmark(groupedBatchedMatmul, (grouped_batched_param_dict,), n_repeat = 100))
# gpu_times = gpu_times.split()
# gpu_cpu_t = float(gpu_times[3])/1e6
# gpu_gpu_t = float(gpu_times[14])/1e6

gpu_t_s = times.gpu_times
cpu_t_s = times.cpu_times

avg_gpu_t = cp.mean(gpu_t_s)
avg_cpu_t = cp.mean(cpu_t_s)

# print(gpu_cpu_t, gpu_gpu_t)
print()
print('CuBLAS times:')
print('cpu:', avg_cpu_t, ' gpu:', avg_gpu_t)








#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()
