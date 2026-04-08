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

import math
from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig
from lerobot.cameras.pika.configuration_pika import PikaCameraConfig, PikaCameraSource

from ..config import RobotConfig


class URPikaControlMode(str, Enum):
    JOINT = "joint"
    TCP = "tcp"


def _default_cameras() -> dict[str, CameraConfig]:
    return {
        # Default to a Pi0-friendly single wrist camera setup. Additional views such
        # as `base_0_rgb` can still be added by overriding `robot.cameras`.
        "right_wrist_0_rgb": PikaCameraConfig(
            source=PikaCameraSource.REALSENSE_COLOR,
            width=640,
            height=480,
            fps=30,
        ),
    }


def _default_joint_limits() -> dict[str, tuple[float, float]]:
    return {f"joint_{index}": (-2 * math.pi, 2 * math.pi) for index in range(1, 7)}


def _default_tcp_relative_target() -> dict[str, float]:
    return {
        "tcp_x": 0.05,
        "tcp_y": 0.05,
        "tcp_z": 0.05,
        "tcp_rx": 0.2,
        "tcp_ry": 0.2,
        "tcp_rz": 0.2,
    }


@RobotConfig.register_subclass("ur_pika")
@dataclass
class URPikaConfig(RobotConfig):
    robot_ip: str
    gripper_port: str | None = None
    command_port: int = 30001
    state_port: int = 30012
    socket_timeout_s: float = 2.0
    state_poll_interval_s: float = 0.05
    control_mode: URPikaControlMode = URPikaControlMode.JOINT

    enable_gripper_on_connect: bool = True
    disable_gripper_on_disconnect: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)

    joint_limits_rad: dict[str, tuple[float, float]] = field(default_factory=_default_joint_limits)
    joint_max_relative_target_rad: float | dict[str, float] | None = 0.35
    movej_acceleration: float = 1.4
    movej_velocity: float = 1.05

    tcp_position_limits_m: dict[str, tuple[float, float]] | None = None
    tcp_rotation_limits_rad: dict[str, tuple[float, float]] | None = None
    tcp_max_relative_target: dict[str, float] | None = field(default_factory=_default_tcp_relative_target)
    movel_acceleration: float = 1.2
    movel_velocity: float = 0.25

    gripper_min_mm: float = 0.0
    gripper_max_mm: float = 90.0
    gripper_max_relative_target_mm: float | None = 10.0

    def __post_init__(self) -> None:
        self.control_mode = URPikaControlMode(self.control_mode)
        super().__post_init__()
