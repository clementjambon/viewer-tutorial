from dataclasses import dataclass

import drjit as dr
import mitsuba as mi
import numpy as np


def pad_alpha(image: np.ndarray) -> np.ndarray:
    assert image.shape[-1] == 3
    return np.concatenate([image, np.ones((*image.shape[:-1], 1))], axis=-1)


@dataclass
class Buffer:

    buffer: np.ndarray
    count: int

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.buffer = None
        self.count = 0

    def add_frame(self, frame: mi.TensorXf) -> None:
        if self.buffer is None:
            self.buffer = frame.numpy()
        else:
            self.buffer += frame.numpy()
        self.count += 1

    def get_raw(self) -> np.ndarray:
        assert self.count > 0
        return self.buffer / self.count

    def get_rgba(self) -> np.ndarray:
        tmp = self.get_raw()
        return pad_alpha(tmp)
