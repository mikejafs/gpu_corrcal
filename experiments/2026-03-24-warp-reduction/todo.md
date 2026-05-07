# To Dos for the warped reduction project

- [To Dos for the warped reduction project](#to-dos-for-the-warped-reduction-project)
  - [Clean up](#clean-up)
  - [Next](#next)
  - [Ongoing](#ongoing)
  - [2026-04-20](#2026-04-20)

## Clean up

- Add some documentation tying together how someone should interpret each of the versions of the 3 and 4 eigenmode hardcoded reduction kernels, how they work, and why and how we have shifted to the general eignmode solution
- Add some notes about why we have shifted to using traditional cuda kernels once again and adequate usecases (not good for this final code) for cupy user defined kernels

## Next
- make tests for gen_eig kernel
  - 1. One plot comparing all (now) 4 methods at 3 eigenmodes
  - 2. Another plot comparing the 3 methods now with a range of eigenmodes
    - this is easiest as actually 2 different sets of plots: the first with a bunch of panels all with different numbers of eigenmodes, showing the timings of each e-mode, and the second holding n_antenna fixed at something like 512 and varying n_eign from say 2 - 30, trying to also notive where register spillover into local memory becomes obvious  
  - baased on tests -- explore other options to solve register spillage (likely ok for now -- check where performance dropoff actually lives -- though might have to write another kernel later if we need to go above this)
- Write a small kernel with implimentation for additional concepts that are confusing: shared mem, simpler multilevel block reduction, basic gpu program to get used to threadIdx notion again etc...
- Write own kernel for 4 eigenmodes hardcoded from scratch with tests to be sure we understand how the whole kernel is functioning
  - do this in a custom cuda kernel with ctypes for extra practice and do the same with the 3 eigenmode version too just for extra practice
    - same these to a useful location with some half-decent documentation
- create latex writeup for gen_eig kernel explaining everybit
  - add to main concept notes
  - $\Rightarrow$ very helpful for the paper I'm writing for this anyway and as a future reference
- begin work on bandpass stuff and other C projects before planning out how to build the next function
- begin building out the next function in line

## Ongoing
- determine the best way to slot in PMPP reading + exercises (leaving this here, since I want to progress this but not too sure where is the best place to write this...)

- Also working through small examples from the NVIDIA HPC GPU hub on github (see useful resources) -- keep this on going for areas of difficulty atm

### Update 
- Probably easiest to pick a day or two each week and just straight up read PMPP chapters with some additional hands-on stuff from GPU HPC hub -- ***DON'T SPEND A LOT OF TIME HERE $\Rightarrow$ JUST KICK-START EACH DAY WITH THIS FOR FURTHER GPU INTUITION/BACKGROUND***


## 2026-04-20
- Continue working on understanding the main kernel for $\Delta^T N^{-1} \Delta$
    - Read pragma_unroll doc, this is where we left off last time
    - finish handwritten notes about kernel
    - shift attention to understanding gen_eig kernel
      - ~~Transfer code to true cuda kernel that is compiled and loaded in using ctypes~~
      - TO TRULY SEE WHAT'S GOING ON WITH THIS REGISTER MEM ALLOCATION:
        - WRITE A KERNEL THAT DYNAMICALLY ALLOCATES MEMORY AND SEE IF THERE'S AN ERROR
        - IF NO ERROR, BENCHMARK THAT CODE AGAINST THE HARDCODED VERSIONS
        - ADD BACK IN PRAGMA UNROLL AND SEE IF THERE'S ANY DIFFERENCE
        - BENCHMARK THE TEMPLATED CODE AGAINST THE KERNEL THAT (I BELIEVE THEN WOULD BE) DYNAMICALLY ALLOCATING MEMORY USED SHARED MEM WHEN INITIALIZING LOCAL ARRAYS
 - Slot in some time (~45 mins) for GPU HPC hub -- focus on GPU basics that will help build intuition better

