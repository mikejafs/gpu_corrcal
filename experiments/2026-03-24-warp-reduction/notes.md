# Fused Reduction-Specific Experiment Notes

- [Fused Reduction-Specific Experiment Notes](#fused-reduction-specific-experiment-notes)
    - [Feb 26 2026](#feb-26-2026)
    - [April 4 2026](#april-4-2026)
    - [April 15 2026](#april-15-2026)
    - [May 14 2026](#May-14-2026)

### Feb 26 2026
- Was interested after running my '3 way' benchmark tests how this compared to bobby's corrcal paper results from Figure 2 in https://arxiv.org/pdf/2602.06109, but I recalled that his code was built around the input matrices taking double precision data types even though all of my comparison tests and the benchmark plot in the folder [fused reduction](cuBLAS/fused_reduction) were built using fp32 everywhere. I tested this today, cleaning up the corrcal code to allow for explicit fp32 function calling of the functions of interest - essentially just building a new function allowing for fp32 - and cleaned up the main benchmark code in the same *fused reduction* folder.
- Benchmark results seem to suggest that fp64 vs fp32 print ~the same time when using close to 1000 antennas for the cpu code -- so I'm not too sure what's goping on there **This could be something to look into**.
  - It seems after a quick google search that going from fp64 -> fp32 may not actually provide that large of an improvement from a cpu perspective compared to the type of improvement on gpus 
  - This could make sense??? If we're using a serialized (i.e., cpu) algorithm where you would have to wait for each step to finish before starting the next, clearly one is waiting a longer time to move things around in general than for all of the operations to be completed... $\rightarrow$ Essentially memory bendwidth limited


### April 4 2026
- So far, I've hardcoded the notion of having 3 eigenmodes into the kerel $\rightarrow$ this needs to be explored asap for whether this is going to be a problem down the road. Need to see
  - 1. how much work it is to hardcode kernels for more eigenmodes
  - 2. if this all makes sense as far as the expected bandpass covariance structure goes.


### April 15 2026
- After chatting with Jon, we should definitely not have a hard coded version of anything since we don't know what the data looks like yet. In light of this, I've switched things to a more general version that accepts any number of eigenmodes $\rightarrow$ Need to see if a) this is actually working correctly, and b) how the speeds compare to before


### May 14 2026
- Jon seems to think that everything going on with register initialization of arrays may not be important and that shared memory initialization should work just fine. I can't remember the argument atm, but something to the effect of we're still bandwidth limited or something at some point in the memory hierarchy and so it won't matter. Something tells me this doesn't seem quite right, so the current tests are surrounding the difference between register memory allocation vs flexible shared mem allocation. The templated solution with strict reg mem initialition of arrays may be the best solution here, but I need to get some profiling done for each of these.

#### Update: 

Currently trying to impliment this above step under the gen_n_eig_warp folder. Specifically under the geneig_warp_red_kernel_run.py file