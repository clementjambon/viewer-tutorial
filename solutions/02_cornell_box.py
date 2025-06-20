import numpy as np
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

import polyscope as ps
import polyscope.imgui as psim

from ps_utils.viewer.base_viewer import BaseViewer
from utils.buffer import Buffer

SCENE_PATH = "data/cbox.xml"
RENDER_SIZE = 512

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

    def init_scene(self):
        self.scene = mi.load_file(SCENE_PATH)
        self.params = mi.traverse(self.scene)
        self.left_color = self.params["left.reflectance.value"].numpy().flatten()
        self.right_color = self.params["right.reflectance.value"].numpy().flatten()
        self.buffer.reset()

    def init_render_buffer(self):
        self.render_buffer_quantity = ps.add_raw_color_alpha_render_image_quantity(
            "render_buffer",
            np.ones((RENDER_SIZE, RENDER_SIZE), dtype=float),
            np.ones((RENDER_SIZE, RENDER_SIZE, 4), dtype=float),
            enabled=True,
            allow_fullscreen_compositing=True,
        )

        self.render_buffer = ps.get_quantity_buffer("render_buffer", "colors")

    def post_init(self, **kwargs):
        # Override resolution to square-shaped
        ps.set_window_size(768, 768)
        # Create a render buffer to display the result of optimization
        self.init_render_buffer()

        # Initialize a buffer to accumulate renderings every frame
        self.buffer = Buffer()
        # Initialize renderer and scene
        self.init_scene()

    def gui(self):
        # Just calling super to get FPS
        super().gui()

        psim.PushItemWidth(150)
        update = False
        _, self.left_color = psim.ColorPicker3(
            "Left color",
            self.left_color,
            COLOR_PICKER_FLAGS,
        )
        update |= psim.IsItemDeactivatedAfterEdit()
        psim.SameLine()
        _, self.right_color = psim.ColorPicker3(
            "Right color",
            self.right_color,
            COLOR_PICKER_FLAGS,
        )
        update |= psim.IsItemDeactivatedAfterEdit()
        psim.PopItemWidth()

        if update:
            self.params["left.reflectance.value"] = mi.Color3f(self.left_color)
            self.params["right.reflectance.value"] = mi.Color3f(self.right_color)
            self.params.update()
            self.buffer.reset()

    def draw(self):
        # Render one frame
        image = mi.render(self.scene, spp=1, seed=self.buffer.count)
        # Add the corresponding image to the accumulation buffer
        self.buffer.add_frame(image)
        # Get the current accumulated image
        rendered_image = self.buffer.get_rgba() ** (1.0 / 2.2)
        # Update the Polyscope render buffer with it
        # NOTE: when using `update_data_from_host`, Polyscope expects (H*W,4) sizes
        self.render_buffer.update_data_from_host(rendered_image.reshape(-1, 4))


if __name__ == "__main__":
    CornellBox()
