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
    Laplacian Smoothing with drag-n-dropped mesh
    """

    def post_init(self, **kwargs):
        self.vertices, self.faces = None, None

        # State variable. When set to True, smoothing will be applied between every frame.
        self.smooth = False

        # Smoothing parameters
        self.step_size = 0.1
        self.taubin = False  # See: https://dl.acm.org/doi/10.1145/218380.218473
        self.taubin_ratio = 0.95  # Ratio between smoothing/inflation steps

    def gui(self):
        # Just calling super to get FPS
        super().gui()

        if self.vertices is not None and self.faces is not None:

            # ============================================================
            # TODO: Add buttons for control and sliders for parameters
            # ============================================================
            pass

    def step(self):
        # If smoothing is enabled, smooth every frame
        if self.smooth:
            self.smoothing_step()

    def smoothing_step(self):
        """
        Performs one step of Laplacian smoothing.
        BONUS: Taubin smoothing (https://dl.acm.org/doi/10.1145/218380.218473)
        """
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

        # ============================================================
        # TODO: Update the Polyscope mesh
        # ============================================================

    def load_mesh(self, input_path: str):
        """
        Loads a mesh with Trimesh and display it with Polyscope
        """

        # Don't forget to stop smoothing if it was running
        self.smooth = False

        # ============================================================
        # TODO: Load a mesh with Trimesh and display it with Polyscope
        # ============================================================

    def ps_drop_callback(self, input_path: str):
        """
        Callback that automatically loads a mesh when drag-n-dropped
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
    LaplacianSmoothing()
