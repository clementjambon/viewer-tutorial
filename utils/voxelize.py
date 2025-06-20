import numpy as np
import trimesh

# Vibe-coded with ChatGPT!


def mesh_to_voxel_grid_indices(
    mesh: trimesh.Trimesh, resolution: int = 100, method: str = "subdivide"
) -> tuple:
    """
    Voxelize a triangular mesh and return integer voxel indices of occupied cells.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to voxelize.
    resolution : int
        Number of voxels along the longest axis of the mesh’s bounding box.
    method : str, {'subdivide', 'ray', 'scipy'}
        Voxelization method for trimesh.voxelized().

    Returns
    -------
    occupancy : np.ndarray of shape (nx, ny, nz), dtype=bool
        Boolean array where True indicates an occupied voxel.
    indices : np.ndarray of shape (N, 3), dtype=int
        Integer grid coordinates (i, j, k) of each occupied voxel.
    """
    # 1. Compute bounding box and pitch
    bounds = mesh.bounds
    min_bound, max_bound = bounds
    extents = max_bound - min_bound
    longest = extents.max()
    pitch = longest / resolution

    # 2. Voxelize
    voxel_grid = mesh.voxelized(pitch, method=method)
    occupancy = voxel_grid.matrix.copy()  # shape (nx, ny, nz)

    # 3. Get integer indices of occupied voxels
    #    np.argwhere returns an array of shape (N, 3) with (i, j, k) coords
    indices = np.argwhere(occupancy)

    return occupancy, indices
