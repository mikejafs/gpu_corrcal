For full profiling using nsight compute (keep in mind this is kernel-level profiling)

	ncu python main.py

It's possible to isolate the kernel of interest after pinpointing which ones may be slower

	ncu --kernel-name regex:gpu_grad_nll python main.py

--------------------------------------------------------
USING THE GUI

It could be easier to use the GUI. To do this, save the report using

	ncu -o report python main.py

This doesn't seem to open cleanly using WSL. To view the report in the GUI, navigate to the file
location by simply following the directory name and double click on the report. This will open the
report in the Nsight GUI.
--------------------------------------------------------


--------------------------------------------------------
Note that Nsight Systems 
	nsys
is good for finding which kernels dominate the runtime.

While Nsight Compute
	ncu
is good for understanding why they're slow.
--------------------------------------------------------



