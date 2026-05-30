import cupy as cp
import re

def get_machine_label():
    gpu_name = cp.cuda.Device(0).attributes  # not ideal, better to use:
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    plot_label = gpu_name.replace(' ', r'\ ')
    file_label = re.search(r'[A-Z]\d+|\d+', gpu_name).group()
    return plot_label, file_label
