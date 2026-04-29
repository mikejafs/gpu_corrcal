# Experiment Description
Finally a kernel for one part of the inversion routine that significantly outperforms cpu corrcal

## Important bits
- The cupy_cpu_warped_bmark.py file was used to perform the 3-way benchmark test comparing the speed
of the cupy, corrcal, and warped reduction kernel way of computing the block multiply $\Delta^T N^{-1}\Delta$.
Output plots can be found in the neighbouring plots folders with local and cluster reference names.
- The main file that outlines a detailed account of the main buliding block of the above matrix multiplication can 
be found in warp_red_kern_r3.py -- Name should probably change, but this is the file that contains my own
documentation as to what's going on in this kernel as I've come to understand things. 
- The 3_&_$_way_bmark_tests folder is where the tests involving all 4 of the gen eig and non-gen eig, cupy, and corrcal bmark tests live and the gen_eig code lives in its own folder under of course gen_n_eig_warp_red