from argparse import ArgumentParser

import numpy as np
import trimesh

# Vibe-coded with ChatGPT!


def mesh_sdf_on_grid(mesh, resolution=(100, 100, 100), padding=0.1):
    """
    Compute the signed distance field of `mesh` on a regular 3D grid.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input triangular mesh.
    resolution : tuple of ints (nx, ny, nz)
        Number of samples along each axis.
    padding : float
        Fractional padding around the mesh’s bounding box
        to ensure the grid covers the object plus a margin.
        E.g. 0.1 = 10% extra in each direction.

    Returns
    -------
    sdf_grid : np.ndarray of shape (nx, ny, nz)
        Signed distance values at each grid point.
        Negative inside, positive outside.
    grid_pts : np.ndarray of shape (nx*ny*nz, 3)
        The flattened coordinates of all grid points.
    grid_coords : tuple of three 1D arrays (xs, ys, zs)
        The coordinate vectors along x, y, z (lengths nx, ny, nz).
    """
    # 1. Compute the padded bounding box of the mesh
    bounds = mesh.bounds  # shape (2, 3): [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    min_bound, max_bound = bounds
    size = max_bound - min_bound
    min_bound = min_bound - size * padding
    max_bound = max_bound + size * padding

    # 2. Build 1D coordinate vectors along each axis
    nx, ny, nz = resolution
    xs = np.linspace(min_bound[0], max_bound[0], nx)
    ys = np.linspace(min_bound[1], max_bound[1], ny)
    zs = np.linspace(min_bound[2], max_bound[2], nz)

    # 3. Create a 3D grid of points (flatten into (N,3) for querying)
    #    np.meshgrid with indexing='ij' so that xs vary along axis 0, ys along axis 1, etc.
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # shape (nx*ny*nz, 3)

    # 4. Query signed distances
    # Option A: use the direct function
    sdf_flat = trimesh.proximity.signed_distance(mesh, grid_pts)
    # Option B (equivalent):
    # pq = trimesh.proximity.ProximityQuery(mesh)
    # sdf_flat = pq.signed_distance(grid_pts)

    # 5. Reshape back into a (nx, ny, nz) volume
    sdf_grid = sdf_flat.reshape((nx, ny, nz))

    return sdf_grid, grid_pts, (xs, ys, zs)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("mesh", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--res", type=int, default=32)
    args = parser.parse_args()

    # 1. Load your mesh (replace with your file)
    mesh = trimesh.load(args.mesh, force="mesh")

    # 2. Choose a resolution—for instance
    res = [args.res] * 3

    # 3. Compute the SDF
    sdf_vol, pts, (xs, ys, zs) = mesh_sdf_on_grid(mesh, resolution=res, padding=0.05)

    # 4. Save it
    np.save(args.output, sdf_vol)
