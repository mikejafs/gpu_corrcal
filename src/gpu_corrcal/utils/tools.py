import cupy as cp
import re
from gpu_corrcal.utils.simulate_params import *
from gpu_corrcal.utils.gridding import *
from gpu_corrcal._kernels import *


def sync():
    """Synchronize the device."""
    sync_device()


def get_machine_label():
    gpu_name = cp.cuda.Device(0).attributes  # not ideal, better to use:
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    plot_label = gpu_name.replace(' ', r'\ ')
    file_label = re.search(r'[A-Z]\d+|\d+', gpu_name).group()
    return plot_label, file_label


def make_test_data(n_eig, n_src, rc_list, seed):
    """Generate random test data using the simulate params library"""
    cp.random.seed(seed)

    rows = rc_list[0]
    cols = rc_list[1]

    n_ant = rows*cols
    # print(f" n_eig={n_eig}", end="", flush=True)

    spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
    edges = spms.edges(rows, cols, use_random=False)
    edges_gpu = cp.asarray(edges)

    sim_data = spms.sim_data()
    noise = sim_data[0]
    diffuse = sim_data[1]
    source = sim_data[2]
    gains = sim_data[3]
    data_vec = sim_data[4]

    ant_arrays = spms.ant_arrays()

    return {
            "noise": noise,
            "diffuse": diffuse,
            "source" : source,
            "edges": edges_gpu,
            "gains": gains,
            "data_vec" : data_vec,
            "ant_arrays": ant_arrays
            }