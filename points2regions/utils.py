import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import normalize
from typing import Optional

def connectivity_matrix(
    xy: np.ndarray,
    method="knn",
    k: int = 5,
    r: Optional[float] = None,
    include_self: bool = False,
) -> sp.spmatrix:
    """
    Compute the connectivity matrix of a dataset based on either k-NN or radius search.

    Parameters
    ----------
    xy : np.ndarray
        The input dataset, where each row is a sample point.
    method : str, optional (default='knn')
        The method to use for computing the connectivity.
        Can be either 'knn' for k-nearest-neighbors or 'radius' for radius search.
    k : int, optional (default=5)
        The number of nearest neighbors to use when method='knn'.
    r : float, optional (default=None)
        The radius to use when method='radius'.
    include_self : bool, optional (default=False)
        If the matrix should contain self connectivities.

    Returns
    -------
    A : sp.spmatrix
        The connectivity matrix, with ones in the positions where two points are
            connected.
    """
    if method == "knn":
        A = kneighbors_graph(xy, k, include_self=include_self).astype('bool')
    else:
        A = radius_neighbors_graph(xy, r, include_self=include_self).astype('bool')
    return A

def attribute_matrix(
    cat: np.ndarray,
    unique_labels=None
):
    """
    Compute the attribute matrix from categorical data, based on one-hot encoding.

    Parameters
    ----------
    cat : np.ndarray
        The categorical data, where each row is a sample and each column is a feature.
    unique_cat : np.ndarray
        Unique categorical data used to setup up the encoder. If "auto", unique
        categories are automatically determined from cat.

    Returns
    -------
    y : sp.spmatrix
        The attribute matrix, in sparse one-hot encoding format.
    categories : list
        The categories present in the data, as determined by the encoder.
    """
    if unique_labels is None:
        categories, col_ind = np.unique(cat, return_inverse=True)
    else:
        categories = unique_labels
        map_to_ind = {u : i for i,u in enumerate(categories)}
        col_ind = np.array([map_to_ind[i] for i in cat])

    row_ind = np.arange(len(cat))
    shape = (len(cat), len(categories))
    val = np.ones(len(cat), dtype=bool)
    y = sp.csr_matrix((val,(row_ind, col_ind)), shape=shape, dtype=bool)
    return y, categories

def spatial_binning_matrix(
    xy: np.ndarray, bin_width: float, return_grid_props: bool = False
) -> sp.spmatrix:


    # Compute shifted coordinates
    mi, ma = xy.min(axis=0, keepdims=True), xy.max(axis=0, keepdims=True)
    xys = xy - mi

    # Compute grid size
    grid = ma - mi
    grid = grid.flatten()

    # Compute bin index
    bin_ids = xys // bin_width
    bin_ids = bin_ids.astype("int")
    bin_ids = tuple(x for x in bin_ids.T)

    # Compute grid size in indices
    size = grid // bin_width + 1
    size = tuple(x for x in size.astype("int"))

    # Convert bin_ids to integers
    linear_ind = np.ravel_multi_index(bin_ids, size)

    # Create a matrix indicating which markers fall in what bin
    bin_matrix, linear_unique_bin_ids = attribute_matrix(linear_ind)


    bin_matrix = bin_matrix.T
    sub_unique_bin_ids = np.unravel_index(linear_unique_bin_ids, size)

    offset = mi.flatten()
    xy_bin = np.vstack(tuple( sub_unique_bin_ids[i] * bin_width + offset[i] for i in range(2))).T

    bin_matrix = bin_matrix.astype('float32').tocsr()
    if return_grid_props:
        grid_props = dict(
            non_empty_bins=linear_unique_bin_ids,
            grid_coords=sub_unique_bin_ids,
            grid_size=size,
            grid_offset=mi.flatten(),
            grid_scale=1.0/bin_width
        )
        return bin_matrix, xy_bin, grid_props
    return bin_matrix, xy_bin



def create_neighbors_matrix(grid_size, non_empty_indices):
    # Total number of grid points
    n_grid_pts = grid_size[0] * grid_size[1]

    # Create 2D arrays of row and column indices for all grid points
    rows, cols = np.indices(grid_size)
    linear_indices = rows * grid_size[1] + cols

    # Convert linear indices of non-empty grid points to subindices
    non_empty_subindices = np.unravel_index(non_empty_indices, grid_size)

    # Create arrays representing potential neighbors in four directions
    neighbors_i = np.array([0, 0, -1, 1, 0])
    neighbors_j = np.array([-1, 1, 0, 0, 0])

    # Compute potential neighbors for all non-empty grid points
    neighbor_candidates_i = non_empty_subindices[0][:, np.newaxis] + neighbors_i
    neighbor_candidates_j = non_empty_subindices[1][:, np.newaxis] + neighbors_j


    # Filter out neighbors that are outside the grid
    valid_neighbors = np.where(
        (0 <= neighbor_candidates_i) & (neighbor_candidates_i < grid_size[0]) &
        (0 <= neighbor_candidates_j) & (neighbor_candidates_j < grid_size[1])
    )

    # Create COO format data for the sparse matrix
    non_empty_indices = np.array(non_empty_indices)
    data = np.ones_like(valid_neighbors[0])
    rows = non_empty_indices[valid_neighbors[0]]
    cols = (neighbor_candidates_i[valid_neighbors], neighbor_candidates_j[valid_neighbors])
    cols = cols[0] * grid_size[1] + cols[1]

    # Create the sparse matrix using COO format
    neighbors = sp.csr_matrix((data, (rows, cols)), shape=(n_grid_pts, n_grid_pts), dtype=bool)

    # Extract the submatrix for non-empty indices
    neighbors = neighbors[non_empty_indices, :][:, non_empty_indices]

    return neighbors


def find_inverse_distance_weights(ij, A, B, bin_width):

    # Create sparse matrix
    num_pts = A.shape[0]
    num_other_pts = B.shape[0]
    cols, rows = ij


    # Inverse distance weighing
    distances = np.linalg.norm(A[rows,:]-B[cols,:], axis=1)
    good_ind = distances <= bin_width*1.000005

    vals = 1.0 / (distances+1e-5)
    #vals = vals / vals.sum(axis=1, keepdims=True)
    vals = vals.flatten()
    #data = np.ones_like(rows, dtype=float)
    vals = vals[good_ind]
    rows = rows[good_ind]
    cols = cols[good_ind]
    sparse_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_pts, num_other_pts), dtype='float32')
    sparse_matrix.eliminate_zeros()
    normalize(sparse_matrix, norm='l1', copy=False)
    return sparse_matrix



def inverse_distance_interpolation(
    xy:np.ndarray,
    labels:np.ndarray,
    unique_labels:np.ndarray,
    pixel_width:float,
    smooth:float,
    min_markers_per_pixel:int
):
    
     
    # Create attribute matrix (nobs x n_unique_labels)
    attributes, _ = attribute_matrix(labels, unique_labels)

    # Number of resolution levels.
    num_levels = 4
    pixel_widths = np.linspace(pixel_width, pixel_width*smooth, num_levels)

    # B maps each gene to a pixel of the highest resolution (smallest pixel width)
    B, xy, grid_props = spatial_binning_matrix(xy, bin_width=pixel_width, return_grid_props=True)

    # Compute features (frequency of each label in each pixel)
    features = B.dot(attributes)

    # Compute center of each pixel
    bin_center_xy = xy + 0.5 * pixel_width

    # Compute lower resolution pixels
    # as well as weightes between neighboring bins
    Ws, Bs = [], []
    density = features.sum(axis=1) 
    X = density.copy()
    
    for level in range(1, num_levels):

        Bi, xyi, props = spatial_binning_matrix(xy, bin_width=pixel_widths[level], return_grid_props=True)
        N = create_neighbors_matrix(props['grid_size'], props['non_empty_bins'])
        # Find a 4-connectivity graph that connects adjacent non-empty bins

        # Find which high-resolution pixels are connected to
        # what low-resolution pixel.
        neighbors = N.dot(Bi).nonzero()
        # Neighbors is of shape 2 x n
        # where the first row are indices to low-resolution pixels
        # and the second row are indices to high-resolution pixels

        # Find weights between the low and high resolution pixels
        low_res_pixel_center_xy = xyi + 0.5 * pixel_widths[level]

        W = find_inverse_distance_weights(
            neighbors, 
            bin_center_xy, 
            low_res_pixel_center_xy,
            pixel_widths[level]
        )

        # Append matrices for later use
        Ws.append(W)
        Bs.append(Bi)

        # Compute density
        density += W.dot(Bi.dot(X))

    # Remove bins with low density
    passed_threshold = density/num_levels >= min_markers_per_pixel
    features = features.multiply(passed_threshold)
    features.eliminate_zeros()

    # Compute features by aggregating different resolutions
    X = features.copy()

    for Wi, Bi in zip(Ws, Bs):
        features +=  Wi.dot(Bi.dot(X))

    # Prepare outputs
    # Convert to numpy array
    passed_threshold = passed_threshold.A.flatten()

    # Log data
    features.data = np.log1p(features.data)

    # Normalizing factor each bin
    s = features.sum(axis=1)

    # Normalize
    norms_r = 1.0 / (s + 1e-5)
    norms_r[np.isinf(norms_r)] = .0
    features = features.multiply(norms_r).tocsr()

    return dict(
        # Features (num_high_res_pixels x n_unique_markers)
        features=features,

        # Position of each pixel
	    xy_pixel=xy,

        # Normalizing factor for each pixel
        norms=norms_r.A.flatten(),

        # Dictionary with grid properties. 
        # Such as the shape of the grid
        grid_props=grid_props,

        # Whether a pixel (bin) passed the
        # density threshold
        passed_threshold=passed_threshold,

        # Sequence of indicies of length
        # nobs that indicates which high-res
        # pixel each observed marker belongs to.
        pix2marker_ind=B.T.nonzero()[1]
    )

