# quick sim

if __name__ == "__main__":
    n_eig=3
    nb = 50
    block_sizes = cp.random.randint(5, 30, size=nb)
    print(block_sizes)

    edges = cp.zeros(nb + 1, dtype=cp.int32)
    edges[1:] = cp.cumsum(block_sizes)
    print(edges)
    
    n_mat = len(edges) - 1
    
    test_mat = cp.random.randn(n_mat, n_eig, n_eig, dtype=cp.float32)
    matrices = test_mat @ cp.transpose(test_mat, [0, 2, 1])
    matrices += cp.eye(n_eig, dtype=cp.float32)*0.01
    
    out = fused_cholesky_inverse(matrices, cp.zeros_like(matrices), edges, n_eig) 
    print(out)