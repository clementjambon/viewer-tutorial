import numpy as np
import polyscope as ps

# Thanks Anh Truong for the inspiration!

SDF_PATH = "data/bunny_sdf.npy"

ps.init()

# Load the densely sampled SDF with shape (res, res, res)
sdf_data = np.load(SDF_PATH)

# Specify the bounds of the SDF volume
dims = tuple(sdf_data.shape)
bound_low = [-1.0] * 3
bound_high = [1.0] * 3

# Instantiate the volume
ps_grid = ps.register_volume_grid("sample grid", dims, bound_low, bound_high)

# Simple SDF grid
# ps_grid.add_scalar_quantity("sdf", sdf_data, defined_on="nodes", enabled=True)

# SDF grid with zero-levelset mesh extraction
ps_grid.add_scalar_quantity(
    "mesh",
    sdf_data,
    defined_on="nodes",
    enable_isosurface_viz=True,  # Gives us the isosurface
    isosurface_level=0.0,
    slice_planes_affect_isosurface=False,  # Prevents the slicer below from slicing the mesh (i.e., only the volume)
    enabled=True,
    isolines_enabled=True,  # Show isolines
)

# Create a slice plane to see what's going on inside
slice_plane = ps.add_scene_slice_plane()
slice_plane.set_draw_widget(True)
# First tuple is position, second is normal of the gizmo
slice_plane.set_pose((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0))

ps.show()
