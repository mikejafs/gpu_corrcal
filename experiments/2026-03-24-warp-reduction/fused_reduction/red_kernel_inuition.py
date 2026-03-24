import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.abspath("../cuBLAS/gemm_grouped_batched"))

from gridding import *

# print(os.path.abspath("../cuBLAS/gemm_grouped_batched"))

np.random.seed(0)

# ------------------------------------------------------------
# Problem setup (small & clear)
# ------------------------------------------------------------

rows = 20
cols = 20
n_ant = rows*cols

r = 3                    # eigenmodes (small!)
block_sizes = [4, 3, 5]  # pretend redundancy classes
# edges = np.zeros(len(block_sizes)+1, dtype=int)
edges, s, k = grid_redundancy_edges(rows, cols)
# edges[1:] = np.cumsum(block_sizes)

N_rows = edges[-1]

# Diffuse matrix (each row is d_i)
D = np.random.randn(N_rows, r).astype(np.float32)

# Noise weights (diagonal of N^{-1})
w = np.random.rand(N_rows).astype(np.float32)

# print("Edges:", edges)
print("D shape:", D.shape)
print()

# ------------------------------------------------------------
# Method 1: Matrix formulation (your current GEMM logic)
# ------------------------------------------------------------

def block_cov_matrix_form(D, w, edges):
    blocks = []
    for b in range(len(edges)-1):
        s, e = edges[b], edges[b+1]
        Db = D[s:e]
        wb = w[s:e]

        # W D  (diagonal multiply)
        WD = wb[:, None] * Db

        # D^T (W D)
        C = Db.T @ WD
        blocks.append(C)
    return blocks

# ------------------------------------------------------------
# Method 2: Reduction of outer products (what GPU kernel does)
# ------------------------------------------------------------

def block_cov_reduction(D, w, edges):
    blocks = []
    for b in range(len(edges)-1):
        s, e = edges[b], edges[b+1]

        C = np.zeros((r, r), dtype=np.float32)

        for i in range(s, e):
            d = D[i]
            C += w[i] * np.outer(d, d)

        blocks.append(C)
    return blocks

# ------------------------------------------------------------
# Compare both methods
# ------------------------------------------------------------

start = time.time()
C_mat = block_cov_matrix_form(D, w, edges)
stop = time.time()

print(stop - start)

start = time.time()
C_red = block_cov_reduction(D, w, edges)
stop = time.time()

print(stop - start)

# for i, (A, B) in enumerate(zip(C_mat, C_red)):
#     print(f"Block {i}")
#     print("Matrix form:\n", A)
#     print("Reduction form:\n", B)
#     print("Match:", np.allclose(A, B))
#     print()
