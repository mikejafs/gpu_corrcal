import numpy as np

def grid_redundancy_edges(n_rows, n_cols, precision = 'int32'):
    """
    Generate 'edges' array for a CHORD-like antenna layout on an n_rows × n_cols grid.
    
    Returns:
        edges: np.ndarray (len = Nblocks + 1)
            edges[k] : edges[k+1] give the index range for the k-th redundant block.
        blocks: dict mapping (dx, dy) → list of baseline index pairs
            (Useful for debugging or block matrix construction.)
        key_sorted: dict
            sorted keys for each of the redundant blocks (I think)
    """

    # list all antenna coordinates
    ants = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    N = len(ants)

    # all unique baseline pairs
    blocks = {}  # maps (dx, dy) → list of baselines

    for a in range(N):
        x1, y1 = ants[a]
        for b in range(a+1, N):
            x2, y2 = ants[b]
            dx = x2 - x1
            dy = y2 - y1

            # redundancy class key
            key = (dx, dy)

            if key not in blocks:
                blocks[key] = []
            blocks[key].append((a, b))

    # Now compute edges: sorted by (dx, dy)
    keys_sorted = sorted(blocks.keys(), key=lambda x: (x[0], x[1]))

    block_sizes = [len(blocks[k]) for k in keys_sorted]
    edges = np.zeros(len(block_sizes) + 1, dtype=precision)
    edges[1:] = np.cumsum(block_sizes)

    return edges, blocks, keys_sorted
