import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from gpytoolbox import fd_interpolate

def linear_interpolation(x_query, x_grid):
    """
    Function to perform linear interpolation on a uniform grid. Finds closest bounding grid points
    then calculates the weights such that x_query = (1-w)x_1 + wx_2
    
    Args:
        x_query (int) : the usually off-grid query location
        x_grid (list) : list of grid points in order
    
    Returns:
        w (float)               : interpolation weight
        nearest_inducing (list) : the indices of the bounding grid points in the input list
    """
    kd_tree = KDTree(x_grid)
    if x_query.ndim ==1:
        x_query = x_query.reshape(-1,1)
    _, nearest_grid_pts = kd_tree.query(x_query,k=2)
    nearest_grid_pts.sort()
    bounding_grid_xs = x_grid[nearest_grid_pts]
    bounding_grid_xs.sort()
    w = np.linalg.norm(x_query - bounding_grid_xs[:,0], axis=1) / np.linalg.norm(bounding_grid_xs[:,1] - bounding_grid_xs[:,0], axis=1)
    return w, nearest_grid_pts

def fd_interpolate_cuts(x_query, grid_size, grid_spacing, full_edges, sub_edges):
    """
    Grid interpolation for off-grid queries/measurements.

    Args:
        x_query (np.array)  : the query locations, (Q,d)
        grid_size (list)    : [nx, ny, (nz)] indicating size of grid in each dim
        grid_spacing (list) : [hx, hy, (hz)] indicating grid spacing in each direction
        full_edges (list)   : list of tuples describing edges in the fully 4-connected grid
        sub_edges (list )   : list of tuples describing edges in sparsely connected grid
    
    Returns:
        W (csr_matrix)      : weight matrix indicating which nodes to interpolate to with what strength.
    """
    # store cut edge set
    cut_edges = frozenset(full_edges) - frozenset(sub_edges)
    W = fd_interpolate(x_query, grid_size, grid_spacing)
    if not cut_edges:
        # if there are no cut edges just return the fd_interpolate result
        return W
    else:
        # build arrays storing cut edge vertices 
        cut_i = np.array([e[0] for e in cut_edges], dtype=np.int32)
        cut_j = np.array([e[1] for e in cut_edges], dtype=np.int32)

        # Extract weights at cut edge vertices (Q, C = num_cuts)
        W_i = np.asarray(W[:, cut_i].todense())  
        W_j = np.asarray(W[:, cut_j].todense())  

        # For each query and each cut edge, zero out the smaller endpoint
        # mask_i[q, c] = True means node cut_i[c] should be zeroed for query q
        mask_i = (W_i <= W_j) & (W_i > 0) & (W_j > 0)  # (Q, C)
        mask_j = (W_j <  W_i) & (W_i > 0) & (W_j > 0)  # (Q, C)

        # Build delta matrix from extracted weights and masks
        Q, N = W.shape
        delta = np.zeros((Q, N), dtype=np.float64)
        np.add.at(delta, (np.where(mask_i)[0], cut_i[np.where(mask_i)[1]]), 
                W_i[mask_i])
        np.add.at(delta, (np.where(mask_j)[0], cut_j[np.where(mask_j)[1]]), 
                W_j[mask_j])

        W = W - csr_matrix(delta)

        # Renormalise each row
        row_sums = np.asarray(W.sum(axis=1)).ravel()  # (Q,)
        row_sums[row_sums == 0] = 1.0                 # avoid div-by-zero
        W = W.multiply(1.0 / row_sums[:, None])

    return W.tocsr()

if __name__ == "__main__":
    grid_size = [3,3]
    grid_spacing = 1
    full_edges = [(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (3,6), (4,5), (4,7), (5,8), (6,7), (7,8)]
    sub_edges = [(0,1), (0,3), (1,2), (2,5),  (3,6), (5,8), (6,7), (7,8)]

    W = fd_interpolate_cuts(np.array([[0.5,0.5],[0.8,0.8],[0.9,0.9],[0.85,0.85]]), grid_size, grid_spacing, full_edges, full_edges)
    W_cut = fd_interpolate_cuts(np.array([[0.5,0.5],[0.8,0.8],[0.9,0.9],[0.85,0.85]]), grid_size, grid_spacing, full_edges, sub_edges)
    print("full", W)
    print("cut", W_cut)
