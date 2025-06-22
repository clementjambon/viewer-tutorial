import numpy as np
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

import polyscope as ps
import polyscope.imgui as psim

from ps_utils.viewer.base_viewer import BaseViewer
from utils.buffer import Buffer

# =====================
# HARDCODED PARAMETERS
# =====================

SCENE_PATH = "data/cbox.xml"
RENDER_SIZE = 512
WINDOW_SIZE = 768
COLOR_PICKER_FLAGS = (
    psim.ImGuiColorEditFlags_Float
    | psim.ImGuiColorEditFlags_NoAlpha
    | psim.ImGuiColorEditFlags_NoSidePreview
    | psim.ImGuiColorEditFlags_NoInputs
    | psim.ImGuiColorEditFlags_PickerHueWheel
)


class CornellBox(BaseViewer):
    """
    Render and edit a Cornell Box with Mistuba
    """

    def post_init(self, **kwargs):
        """
        Our usual initialization.
        """
        # Override resolution to square-shaped
        ps.set_window_size(WINDOW_SIZE, WINDOW_SIZE)
        # Create a Polyscope render buffer to display the result of optimization
        self.init_render_buffer()

        # Initialize a custom buffer to accumulate renderings every frame
        self.buffer = Buffer()
        # Initialize renderer and scene
        self.init_scene()

    def init_render_buffer(self):
        """
        Placeholder code to initialize a Polyscope render buffer.
        I always use this very same snippet in every project.

        The render buffer is simply initialized as a (H, W, 4) array.
        NOTE: it will always be resized to fit the window (without preserving the aspect ratio)!

        NOTE: only RGBA buffer work properly i.e., 4 channels!
        """
        self.render_buffer_quantity = ps.add_raw_color_alpha_render_image_quantity(
            "render_buffer",
            np.ones((RENDER_SIZE, RENDER_SIZE), dtype=float),
            np.ones((RENDER_SIZE, RENDER_SIZE, 4), dtype=float),
            enabled=True,
            allow_fullscreen_compositing=True,
        )

        self.render_buffer = ps.get_quantity_buffer("render_buffer", "colors")

    def gui(self):

        # Just calling super to get FPS
        super().gui()

        update = False

        # =================================================================
        # TODO: Add color pickers to control `left_color` and `right_color`
        # NOTE: It's better to update the scen after edits are complete
        # =================================================================

        if update:
            self.update_scene()

    def draw(self):
        """
        Ray traces the scene once.
        Accumulates the new samples in a buffer.
        Passes the rendered image to the Polyscope render buffer.
        """
        # =====
        # TODO
        # =====

    # ======
    # SCENE
    # ======

    def init_scene(self):
        """
        Initialize the Cornell Box scene (with Mitsuba)
        NOTE: You don't need to pay too much attention to this ;)
        """
        self.scene = mi.load_file(SCENE_PATH)
        self.params = mi.traverse(self.scene)
        self.left_color = self.params["left.reflectance.value"].numpy().flatten()
        self.right_color = self.params["right.reflectance.value"].numpy().flatten()
        self.buffer.reset()

    def update_scene(self):
        """
        Update the scene when the parameters are changed
        NOTE: You don't need to pay too much attention to this ;)
        """
        self.params["left.reflectance.value"] = mi.Color3f(self.left_color)
        self.params["right.reflectance.value"] = mi.Color3f(self.right_color)
        self.params.update()
        self.buffer.reset()


if __name__ == "__main__":
    CornellBox()
