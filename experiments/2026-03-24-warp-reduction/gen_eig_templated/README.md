# Folder Description

*Add folder description*

## Important Conclusions

### Hardcode vs Templated Code

- The result of testing the hardcoded version (O.G. version - warp_red_kernel-r3) against the templated version at of course just 3 eigenmodes returns the correct result - a.k.a. the results are the same to within machine precision. Specifically:

'''
WITH unroll
=================================================================
CORRECTNESS TESTS
=================================================================
AGAINST HARDCODED VERSION:  n_eig= 3  |  allclose: True  |  max |diff|: 1.56e-02
-----------------------------------------------------------------
ALL PASSED
'''

- Benchmarking the hardcoded version vs the templated version also reveals that the templated code runs slightly faster for the same parameters, and although the speedup is small, it is reliably ~7 us faster. This could simply be the result of python bloating using the f-string approach, but I'm not completely sure.

'''
WITH unroll
=================================================================
TIMING TESTS -> n_ant = 32 * 24 = 768
=================================================================
  [n_eig=3  n_sym=  6]   KERNEL: gpu=    27.1 us  cpu=     5.0 us | HARDCODE: gpu=    34.2 us  cpu=    12.6 us
=================================================================
'''