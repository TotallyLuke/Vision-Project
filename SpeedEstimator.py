from math import dist
from typing import Tuple

import numpy as np
from numpy import typing as npt


class SpeedEstimator:
    def __init__(self, trk_history, view_transformer):
        self.trk_history = trk_history
        self.view_transformer = view_transformer

    def estimate_speed(self, id: int, fps: int = 30) -> Tuple[float, float]:
        start_source = self.trk_history[id][-1]
        end_source = self.trk_history[id][0]

        start, end = self.view_transformer.transform_points(points=np.array([start_source, end_source]))

        distance = dist(tuple(start), tuple(end))
        time = len(self.trk_history[id]) / fps
        speed = distance / time * 3.6

        error_dist_start = self.get_pixel_discretization_error(start_source)
        error_dist_end = self.get_pixel_discretization_error(end_source)
        error = (error_dist_start + error_dist_end) / time * 3.6

        return speed, error

    def get_pixel_discretization_error(self, source_point: npt.ArrayLike) -> float:
        [target_point] = self.view_transformer.transform_points(np.array(source_point))

        neighborhood = self.view_transformer.transform_points(np.asarray([
            [source_point[0] - 0.5, source_point[1] + 0.5],
            [source_point[0] + 0.5, source_point[1] + 0.5],
            [source_point[0] + 0.5, source_point[1] - 0.5],
            [source_point[0] - 0.5, source_point[1] - 0.5]
        ]))

        return max([dist(tuple(target_point), tuple(p))
                    for p in neighborhood])
