#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["PikaCameraConfig", "PikaCameraSource", "ColorMode", "Cv2Rotation"]


class PikaCameraSource(str, Enum):
    FISHEYE = "fisheye"
    REALSENSE_COLOR = "realsense_color"


@CameraConfig.register_subclass("pika")
@dataclass
class PikaCameraConfig(CameraConfig):
    source: PikaCameraSource = PikaCameraSource.FISHEYE
    port: str | None = None
    fisheye_index: int | None = None
    realsense_serial_number: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: float = 1.0
    fisheye_thread_fps: int = 100

    def __post_init__(self) -> None:
        self.source = PikaCameraSource(self.source)
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if self.width is None or self.height is None or self.fps is None:
            raise ValueError("Pika cameras require explicit `width`, `height`, and `fps`.")
