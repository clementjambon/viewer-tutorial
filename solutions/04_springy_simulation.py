import polyscope as ps
import polyscope.imgui as psim

import numpy as np
import trimesh
import gpytoolbox

from ps_utils.viewer.base_viewer import BaseViewer
from ps_utils.ui.save_utils import check_extension
from ps_utils.structures.voxel_set import VoxelSet

BASIC_MESH_EXTENSIONS = {".ply", ".obj", ".stl"}

from utils.voxel_spring_simulator import VoxelSpringSimulator
from utils.voxelize import mesh_to_voxel_grid_indices

VOXEL_RES = 20


class SpringySimulation(BaseViewer):
    """
    Demo viewer showcasing a springy simulation
    """

    def post_init(self, **kwargs):
        # Simulation parameters
        self.dt = 0.02
        self.stiffness = 500.0
        self.dampening = 0.1

        # Load a mesh when starting the viewer
        self.load_mesh("data/bunny.obj")

    def gui(self):
        # Just calling super to get FPS
        super().gui()

        reinitialize_simulation = False

        # ========================
        # SIMULATION PARAMETERS
        # ========================

        psim.SeparatorText("Simulation Parameters")

        _, self.dt = psim.SliderFloat("dt", self.dt, v_min=0.005, v_max=0.02)
        _, self.stiffness = psim.SliderFloat(
            "stiffness", self.stiffness, v_min=10.0, v_max=1000.0
        )
        reinitialize_simulation |= psim.IsItemDeactivatedAfterEdit()
        _, self.dampening = psim.SliderFloat(
            "dampening", self.dampening, v_min=0.0, v_max=1.0
        )
        reinitialize_simulation |= psim.IsItemDeactivatedAfterEdit()

        if reinitialize_simulation:
            self.init_simulation(
                coords=self.voxel_set.coords,
                selection_mask=self.voxel_set.selection_mask,
            )

        # ========================
        # VOXELSET
        # ========================

        if self.voxel_set.gui():
            self.init_simulation(
                coords=self.voxel_set.coords,
                selection_mask=self.voxel_set.selection_mask,
                keep_voxelset=True,
            )

    def step(self):
        # Step the simulation
        self.simulation_step()

    def simulation_step(self):
        # Step the simulation
        self.sim.step(self.dt)

        # Update the point cloud and edges
        self.ps_pointcloud = ps.register_point_cloud(
            "points",
            self.sim.x,
        )
        self.ps_edges = ps.register_curve_network("springs", self.sim.x, self.sim.edges)

    def init_simulation(
        self,
        coords: np.ndarray,
        selection_mask: np.ndarray | None = None,
        keep_voxelset: bool = False,
    ):
        """
        Initialize the simulation
        """
        # Slightly offset initial positions to create jiggly patterns
        init_pos = coords.copy().astype(float)
        init_pos[:, 0] *= 1.2

        # Pin the top-most elements (for initialization only)
        if selection_mask is None:
            selection_mask = coords[:, 1] == coords[:, 1].max()
        fixed_ids = np.nonzero(selection_mask)[0]

        # Create the simulator
        self.sim = VoxelSpringSimulator(
            coords=coords,
            init_positions=init_pos,
            stiffness=self.stiffness,
            mass=1.0,
            damping=self.dampening,
            gravity=[0, -9.81, 0],
            fixed=fixed_ids,
        )

        # This creates a selectable voxel set, useful to manually select voxels
        if not keep_voxelset:
            self.voxel_set = VoxelSet(
                coords=coords,
                voxel_res=VOXEL_RES,
                bbox_min=0,
                bbox_max=VOXEL_RES,
                selection_mask=selection_mask,
                offset=np.array([1.5 * VOXEL_RES, 0.0, 0.0]),
                name="Boundary Conditions",
            )

    def load_mesh(self, input_path: str):
        """
        Loads a mesh with Trimesh. Voxelizes it.
        And initializes the simulation.
        """
        # Load the new grid coordinates
        mesh = trimesh.load(input_path)
        # Converts the mesh to voxels coordinates (i.e., integer coordinates of shape(n_coordinates, 3))
        grid_coords = mesh_to_voxel_grid_indices(mesh, VOXEL_RES)[1]
        # Initialize the simulation with them
        self.init_simulation(grid_coords)

    def ps_drop_callback(self, input_path: str):
        """
        Callback that automatically loads a mesh when drag-n-dropped (same as 01_laplacian_smoothing)
        """
        # Check valid mesh extensions
        if check_extension(input_path, BASIC_MESH_EXTENSIONS):
            try:
                self.load_mesh(input_path)
            except Exception as e:
                print(f"Couldn't load mesh at: {input_path}")
        else:
            print(f"Can only load meshes with extensions: {BASIC_MESH_EXTENSIONS}")


if __name__ == "__main__":
    SpringySimulation()
