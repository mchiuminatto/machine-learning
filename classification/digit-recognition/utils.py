import os
import logging
import matplotlib.pyplot as plt


DIRECTION_MAP: dict = {
    "UP":{"direction": -1, "axis":0},
    "DOWN":{"direction": 1, "axis":0},
    "LEFT":{"direction": -1, "axis":1},
    "RIGHT":{"direction": 1, "axis":1}
}
def shift_image(image_vector: np.array, direction: str, steps: int, resolution_x: int, resolution_y: int):

    direction_upper: str = direction.upper()

    if len(image_vector) != resolution_x*resolution_y:
        raise ValueError(f"Vector elements {len(image_vector)} differs from intended reshape {resolution_x*resolution_y}")

    M1 = image_vector.reshape(resolution_x, resolution_y)
    M1 = np.roll(M1, steps*DIRECTION_MAP[direction_upper]["direction"], DIRECTION_MAP[direction_upper]["axis"])

    return M1.reshape(-1)

def shift_digits(mnist: np.array):
    augmented_digits: list = []
    augmented_targets: list = []

    for position in range(len(mnist.data)):
        digit = mnist["data"][position]
        digit_up = shift_image(digit, "up", 1, 28,28)
        digit_down = shift_image(digit, "down", 1, 28,28)
        digit_left = shift_image(digit, "left", 1, 28,28)
        digit_right = shift_image(digit, "right", 1, 28,28)
        target = mnist["target"][position]
        augmented_digits += [digit_up, digit_down, digit_left, digit_right]
        augmented_targets += [target]*4

class Images:
    def __init__(self,
                 images_path: str,
                 tight_layout: bool = True,
                 figure_extension: str = "png",
                 resolution: int = 300):
        self._images_path: str = images_path
        self._tight_layout: bool = tight_layout
        self._figure_extension: str = figure_extension
        self._resolution: int = resolution

    def save_fig(self, fig_id):
        path = os.path.join(self._images_path, fig_id + "." + self._figure_extension)
        logging.info(f"Saving figure {fig_id}")
        if self._tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=self._figure_extension, dpi=self._resolution)
