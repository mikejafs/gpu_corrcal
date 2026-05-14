# Folder Description

Generating ~random files supporting the convergence of results out outcome seen in the warp reduction kernel. 

Files like optimized_red_kernels, small_red_kernel and warp_red_kernel were initially generated to show the efficacy of combining techniques like 2 level block reduction using warp sum primitives and shared memory writes.

red_kernel_intuition showcases using simple python code how the kernels accumulate the inner product as a combination of small outerproducts.

The key files are the following:

## warp_red_kern_r3

The eventual kernel that I settled on showing strong performance over previous techniques. The file houses the kernel written using a string literal and called using CuPy rawkernel functionalities. Tests live at the bottom of the file to show efficacy.

## warp_red_kernel_r3_r4

Similar to above, though this file came first showcasing the modifications to the kernel necessary if we wanted to scale to 4 eigenmodes.

## cupy_cpu_warped_bmark.py

This houses the location of the main tests benchmarking times between the r3 kernel, cupy, and cpu_corrcal. Accompanying plots can be found in the main gpu_corrcal results section, and more locally, in the bmark plots folders inside this experiment.

