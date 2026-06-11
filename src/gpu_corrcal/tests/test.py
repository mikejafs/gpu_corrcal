from corrcal.sparse import *
from gpu_corrcal.utils.tools import *
import matplotlib.pylab as plt
from cupyx.profiler import benchmark
import math


# td = make_test_data(3, (25, 25), 12)

# N = cp.asnumpy(td["noise"])
# D = cp.asnumpy(td["diffuse"])
# eds = cp.asnumpy(td["edges"])
# S = cp.asnumpy(td["source"])

# # print(S.shape)

# sc = SparseCov(N, S, D, eds, 3, isinv=False)
# # inv_sc = sc.inv()

# # sc_dense = sc.expand()
# # inv_sc_dense = inv_sc.expand()

# # out = sc_dense @ inv_sc_dense



# # diff_cov_block1 = sc.diff_mat[eds[0]:eds[1]] @ sc.diff_mat[eds[0]:eds[1]].T
# # inv_diff_cov_block1 = inv_sc.diff_mat[eds[0]:eds[1]] @ inv_sc.diff_mat[eds[0]:eds[1]].T


# # print(diff_cov_block1.shape)

# # out = diff_cov_block1 @ inv_diff_cov_block1
# # print(out)

# # plt.imshow(out)
# # plt.show()


# ref_cpu_times = benchmark(sc.inv, (), n_repeat= 1000)
# print(ref_cpu_times)

# # print(sc.diff_mat[eds[0]:eds[1]].T @ inv_sc.diff_mat[eds[0]:eds[1]])
# # print(inv_sc.diff_mat)

nums = [10, 20, 30] #...

for i in range(int(math.isqrt(nums[0])), 0, -1):
    if nums[0] % i == 0:
        out = [i, nums[0] // i]
        break
print(out)