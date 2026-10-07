"""
Shapes walkthrough for the zero-padded CuPy gain application.

    python example_apply_gains.py

Requires the one-line fix in cupy_reference/cupy_utils.py, zeropad_gains():
    zeroPad(gain_mat, edges, return_inv=False)
 -> zeroPad(gain_mat, edges, return_inv=False, dtype=cp.float64)
(zeroPad has no default for dtype, so the original line raises TypeError.)
"""

import numpy as np
import cupy as cp
from gpu_corrcal.utils.simulate_params import SimCorrcalParams
from gpu_corrcal.utils.zp_puregpu_funcs_py import zeroPad, undo_zeroPad
from gpu_corrcal.cupy_reference.cupy_utils import zeropad_gains, apply_gains
from corrcal.utils import apply_gains_to_mat

# ---------------------------------------------------------------------------
# small array so everything prints
# ---------------------------------------------------------------------------
rows, cols = 4, 5
n_eig, n_src = 1, 2
cp.random.seed(12)

spms = SimCorrcalParams(rows * cols, n_eig, n_src, precision="float32", xp=cp)
edges = cp.asarray(spms.edges(rows, cols))
noise, diffuse, source, gains, data = spms.sim_data()
ant_1, ant_2 = spms.ant_arrays()

sizes = cp.diff(edges)                         # rows per block (Re/Im split)
n_ant, n_bl = rows * cols, int(ant_1.size)

print("=" * 70)
print("INPUTS (flat layout)")
print("=" * 70)
print(f"n_ant = {n_ant}, n_bl = {n_bl}, n_row = 2*n_bl = {int(edges[-1])}")
print(f"gains    {gains.shape}   (Re/Im alternating, 2*n_ant)")
print(f"ant_1/2  {ant_1.shape}")
print(f"diffuse  {diffuse.shape}   (n_row, n_eig)")
print(f"edges    {edges.shape}   -> n_blocks = {edges.size - 1}")
print(f"block sizes (rows): {cp.asnumpy(sizes)}")

# ---------------------------------------------------------------------------
# zero-pad the diffuse matrix (done ONCE per calibration: independent of gains)
# ---------------------------------------------------------------------------
zp_diffuse, lb, nb = zeroPad(diffuse, edges, return_inv=False, dtype=cp.float32)

# ---------------------------------------------------------------------------
# build + zero-pad the gain matrix (depends on gains -> every likelihood call)
# ---------------------------------------------------------------------------
G = zeropad_gains(gains, edges, ant_1, ant_2, dtype=cp.float32)

# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
out = apply_gains(G, zp_diffuse)

print()
print("=" * 70)
print("ZERO-PADDED SHAPES")
print("=" * 70)
print(f"largest_block = {lb}, n_blocks = {nb}")
print(f"zp_diffuse  {zp_diffuse.shape}   (n_blocks, largest_block, n_eig)")
print(f"G           {G.shape}   (n_blocks, largest_block/2, 1)  complex: one entry per baseline")
print(f"out         {out.shape}   (n_blocks, largest_block, n_eig)")

# ---------------------------------------------------------------------------
# is the output still zero-padded?
# ---------------------------------------------------------------------------
row = cp.arange(lb)[None, :]                   # (1, lb)
pad = row >= sizes[:, None]                    # (nb, lb): True on padded rows
pad_bl = cp.arange(lb // 2)[None, :] >= (sizes // 2)[:, None]

print()
print("=" * 70)
print("PADDING CHECK")
print("=" * 70)
print(f"padded rows in zp_diffuse: {int(pad.sum())} of {nb * lb}")
print(f"max |zp_diffuse| on padded rows: {float(cp.abs(zp_diffuse[pad]).max()):.1e}")
print(f"max |G|          on padded rows: {float(cp.abs(G[pad_bl]).max()):.1e}")
print(f"max |out|        on padded rows: {float(cp.abs(out[pad]).max()):.1e}")

b = int(cp.argmin(sizes))
print(f"\nsmallest block b={b}: {int(sizes[b])} real rows, {lb - int(sizes[b])} padded rows")
print("G[b, :, 0] (per-baseline complex gain):")
print(cp.asnumpy(G[b, :, 0]))
print("out[b] (Re/Im rows alternate, padded rows at the bottom):")
print(cp.asnumpy(out[b]))

# ---------------------------------------------------------------------------
# un-pad and compare with CorrCal's CPU apply_gains_to_mat
# ---------------------------------------------------------------------------
out_flat = undo_zeroPad(out, edges, dtype=cp.float32, ReImsplit=True)
ref = apply_gains_to_mat(
    cp.asnumpy(gains).astype(np.float64),
    cp.asnumpy(diffuse).astype(np.float64),
    cp.asnumpy(ant_1),
    cp.asnumpy(ant_2),
)
rel = np.abs(cp.asnumpy(out_flat) - ref).max() / np.abs(ref).max()

print()
print("=" * 70)
print("VS CORRCAL (after undo_zeroPad)")
print("=" * 70)
print(f"out_flat {out_flat.shape}   ref {ref.shape}   max rel err = {rel:.2e}")