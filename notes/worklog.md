# Short Summaries of Working Sessions

e.g., 
## 2026-04-02

### Work
- Benchmarked grouped GEMM vs CuPy (N=256–512, r=5)

### Results
- CuPy still faster even with padding
- grouped version dominated by setup overhead

### Notes
- likely memory-bound behavior
- pointer construction cost non-negligible

### Decision
- deprioritize grouped GEMM for now
- shift focus to fused reduction

### Next
- implement minimal fused reduction kernel
- profile with Nsight

# Working Session Log 

- [April 15 2026](#april-15-2026)
- [April 29 2026](#april-29-2026)


## April 15 2026
- Chatted with Jon -- needed to get rid of hardcoded eigenmode in the reduction kernel
- re-wrote kernel for general eigenmodes; still awaiting detailed tests for correctness etc..

## April 29 2026

### Coding Work

- 

### Results



