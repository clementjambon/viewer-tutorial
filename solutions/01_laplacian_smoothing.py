import polyscope as ps
import polyscope.imgui as psim

import trimesh
import gpytoolbox

from ps_utils.viewer.base_viewer import BaseViewer
from ps_utils.ui.save_utils import check_extension
from ps_utils.ui.buttons import state_button
from ps_utils.ui.sliders import exp_slider

BASIC_MESH_EXTENSIONS = {".ply", ".obj", ".stl"}


class LaplacianSmoothing(BaseViewer):
    """
    Laplacian Smoothing with mesh drag-n-drops
    """

    def post_init(self, **kwargs):
        self.vertices, self.faces = None, None
        self.step_size = 0.1
        self.taubin_ratio = 0.95
        self.smooth = False
        self.taubin = False

    def gui(self):
        # Just calling super to get FPS
        super().gui()

        if self.vertices is not None:

            # Buttons
            _, self.smooth = state_button(self.smooth, "Stop", "Smooth")
            psim.SameLine()
            if psim.Button("Step"):
                self.smoothing_step()

            # Smoothing parameters
            _, self.step_size = psim.SliderFloat(
                "Step size", self.step_size, v_min=0.01, v_max=1.0
            )
            _, self.taubin = psim.Checkbox("Taubin smoothing", self.taubin)

    def step(self):
        if self.smooth:
            self.smoothing_step()

    def smoothing_step(self):
        assert self.vertices is not None
        assert self.faces is not None

        # Compute the Laplacian
        laplacian = gpytoolbox.cotangent_laplacian(self.vertices, self.faces)

        # Apply one step of Laplacian smoothing
        self.vertices -= self.step_size * laplacian @ self.vertices

        # For Taubin smoothing, repeat the operation but in the opposite direction
        if self.taubin:
            laplacian = gpytoolbox.cotangent_laplacian(self.vertices, self.faces)
            self.vertices += (
                self.taubin_ratio * self.step_size * laplacian @ self.vertices
            )

        # Update the mesh
        # NOTE: since a `mesh` with the same name was already created below, it will automatically be overwritten!
        self.ps_mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)

    def load_mesh(self, input_path: str):
        # Load mesh
        mesh = trimesh.load(input_path)
        self.vertices, self.faces = mesh.vertices, mesh.faces
        self.ps_mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
        self.smooth = False

    def ps_drop_callback(self, input_path: str):
        # Check valid mesh extensions
        if check_extension(input_path, BASIC_MESH_EXTENSIONS):
            try:
                self.load_mesh(input_path)
            except Exception as e:
                print(f"Couldn't load mesh at: {input_path}")
        else:
            print(f"Can only load meshes with extensions: {BASIC_MESH_EXTENSIONS}")


if __name__ == "__main__":
    LaplacianSmoothing()
