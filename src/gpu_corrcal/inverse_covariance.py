from gpu_corrcal.linalg import *
# from gpu_corrcal._kernels import *

def inv_cov(noise, diffuse, source, edges):
    # inv_noise = noise.copy()
    # inv_diffuse = diffuse.copy()
    # noise = noise.copy()

    inv_noise, temp2 = make_small_blocks(noise, diffuse, edges)

    return inv_noise, temp2