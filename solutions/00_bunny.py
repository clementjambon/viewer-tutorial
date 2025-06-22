import polyscope as ps
import trimesh

MESH_PATH = "data/bunny.obj"

# Initialize Polyscope
ps.init()

# Load our favorite bunny
mesh = trimesh.load(MESH_PATH)

# Display it in Polyscope
ps.register_surface_mesh(
    "bunny",
    mesh.vertices,
    mesh.faces,
)

# Start Polyscope
ps.show()
