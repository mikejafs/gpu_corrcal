from gpu_corrcal.linalg import *

def inv_cov(noise, diffuse, source, edges):

    inv_noise, temp2 = make_small_blocks(noise, diffuse, edges)

    return inv_noise, temp2