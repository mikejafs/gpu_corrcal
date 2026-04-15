# Experiment Description
Finally a kernel for one part of the inversion routine that significantly outperforms cpu corrcal

## Important bits
The cupy_cpu_warped_bmark.py file was used to perform the 3-way benchmark test comparing the speed
of the cupy, corrcal, and warped reduction kernel way of computing the block multiply $\Delta^T N^{-1}\Delta$.
Output plots can be found in the neighbouring plots folders with local and cluster reference names.