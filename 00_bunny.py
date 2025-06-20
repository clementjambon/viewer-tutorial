import polyscope as ps
import trimesh

# Initialize Polyscope
ps.init()

# Load our favorite bunny
mesh = trimesh.load("data/bunny.obj")

# Display it in Polyscope
ps.register_surface_mesh(
    "bunny",
    mesh.vertices,
    mesh.faces,
)

# Start Polyscope
ps.show()
