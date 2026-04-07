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

from __future__ import annotations

import logging
from functools import cached_property

from lerobot.cameras.pika.configuration_pika import PikaCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.pika.shared import (
    SharedPikaDevice,
    acquire_shared_pika_device,
    release_shared_pika_device,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_ur_pika import URPikaConfig, URPikaControlMode
from .ur_socket import URStateReader, build_movej, build_movel, send_urscript

logger = logging.getLogger(__name__)

UR_JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
TCP_KEYS = ("tcp_x.pos", "tcp_y.pos", "tcp_z.pos", "tcp_rx.pos", "tcp_ry.pos", "tcp_rz.pos")
GRIPPER_KEY = "gripper.pos"


class URPika(Robot):
    config_class = URPikaConfig
    name = "ur_pika"

    def __init__(self, config: URPikaConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = f"{self.name}_{self.config.control_mode.value}"

        self._attach_gripper_port_to_camera_configs()
        self.cameras = make_cameras_from_configs(config.cameras)

        self._pika_device: SharedPikaDevice | None = None
        self._state_reader: URStateReader | None = None

    def _attach_gripper_port_to_camera_configs(self) -> None:
        if not self.config.gripper_port:
            return

        for camera_config in self.config.cameras.values():
            if isinstance(camera_config, PikaCameraConfig) and camera_config.port is None:
                camera_config.port = self.config.gripper_port

    @property
    def _camera_features(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @property
    def _state_features(self) -> dict[str, type]:
        if self.config.control_mode == URPikaControlMode.JOINT:
            features = dict.fromkeys(UR_JOINT_KEYS, float)
        else:
            features = dict.fromkeys(TCP_KEYS, float)

        features[GRIPPER_KEY] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int | None, int | None, int]]:
        return {**self._state_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_features

    @property
    def is_connected(self) -> bool:
        return (
            self._state_reader is not None
            and self._state_reader.is_connected
            and self._pika_device is not None
            and self._pika_device.is_connected
            and all(camera.is_connected for camera in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        try:
            self._state_reader = URStateReader(
                host=self.config.robot_ip,
                port=self.config.state_port,
                poll_interval_s=self.config.state_poll_interval_s,
                socket_timeout_s=self.config.socket_timeout_s,
            )
            self._state_reader.start()
            if not self._state_reader.wait_for_first_state(timeout_s=max(self.config.socket_timeout_s * 2, 5.0)):
                raise TimeoutError("Timed out waiting for the first UR state packet.")

            self._pika_device = acquire_shared_pika_device(self.config.gripper_port)
            self.config.gripper_port = self._pika_device.port
            self._attach_gripper_port_to_runtime_cameras()

            if self.config.enable_gripper_on_connect:
                self._pika_device.enable_gripper()

            for camera in self.cameras.values():
                camera.connect()

            self.configure()
            logger.info("%s connected.", self)
        except Exception:
            self._disconnect_impl()
            raise

    def _attach_gripper_port_to_runtime_cameras(self) -> None:
        if self._pika_device is None:
            return

        for camera in self.cameras.values():
            camera_config = getattr(camera, "config", None)
            if isinstance(camera_config, PikaCameraConfig) and camera_config.port is None:
                camera_config.port = self._pika_device.port

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _require_state_reader(self) -> URStateReader:
        if self._state_reader is None:
            raise RuntimeError("UR state reader is not initialized.")
        return self._state_reader

    def _require_pika_device(self) -> SharedPikaDevice:
        if self._pika_device is None:
            raise RuntimeError("Pika device is not initialized.")
        return self._pika_device

    def _get_latest_state(self):
        return self._require_state_reader().get_latest_state(max_age_s=max(1.0, self.config.socket_timeout_s * 2))

    def _ensure_robot_state_is_safe(self) -> None:
        state = self._get_latest_state()
        if state.robot_mode.is_emergency_stopped:
            raise RuntimeError("UR robot is emergency stopped.")
        if state.robot_mode.is_protective_stopped:
            raise RuntimeError("UR robot is protective stopped.")

    def _joint_relative_limit_dict(self) -> float | dict[str, float] | None:
        limit = self.config.joint_max_relative_target_rad
        if limit is None or isinstance(limit, float):
            return limit
        return {f"{key}.pos": value for key, value in limit.items()}

    def _tcp_relative_limit_dict(self) -> dict[str, float] | None:
        limit = self.config.tcp_max_relative_target
        if limit is None:
            return None
        return {f"{key}.pos": value for key, value in limit.items()}

    @staticmethod
    def _clip_by_limits(
        action: dict[str, float], limits: dict[str, tuple[float, float]] | None
    ) -> dict[str, float]:
        if limits is None:
            return action

        clipped_action = dict(action)
        for key, (lower, upper) in limits.items():
            if key in clipped_action:
                clipped_action[key] = min(max(clipped_action[key], lower), upper)
        return clipped_action

    def _get_joint_observation(self) -> dict[str, float]:
        state = self._get_latest_state()
        if len(state.joint_positions_rad) != len(UR_JOINT_KEYS):
            raise RuntimeError("Latest UR state does not contain six joint positions.")
        return {key: float(state.joint_positions_rad[index]) for index, key in enumerate(UR_JOINT_KEYS)}

    def _get_tcp_observation(self) -> dict[str, float]:
        state = self._get_latest_state()
        if len(state.tcp_pose) != len(TCP_KEYS):
            raise RuntimeError("Latest UR state does not contain a full TCP pose.")
        return {key: float(state.tcp_pose[index]) for index, key in enumerate(TCP_KEYS)}

    def _clip_joint_action(self, action: RobotAction) -> dict[str, float]:
        goal = {key: float(action[key]) for key in UR_JOINT_KEYS}

        relative_limit = self._joint_relative_limit_dict()
        if relative_limit is not None:
            present = self._get_joint_observation()
            goal_present_pos = {key: (goal[key], present[key]) for key in UR_JOINT_KEYS}
            goal = ensure_safe_goal_position(goal_present_pos, relative_limit)

        absolute_limits = {f"{key}.pos": value for key, value in self.config.joint_limits_rad.items()}
        return self._clip_by_limits(goal, absolute_limits)

    def _clip_tcp_action(self, action: RobotAction) -> dict[str, float]:
        goal = {key: float(action[key]) for key in TCP_KEYS}

        relative_limit = self._tcp_relative_limit_dict()
        if relative_limit is not None:
            present = self._get_tcp_observation()
            goal_present_pos = {key: (goal[key], present[key]) for key in TCP_KEYS}
            goal = ensure_safe_goal_position(goal_present_pos, relative_limit)

        absolute_limits: dict[str, tuple[float, float]] = {}
        if self.config.tcp_position_limits_m is not None:
            absolute_limits.update({f"{key}.pos": value for key, value in self.config.tcp_position_limits_m.items()})
        if self.config.tcp_rotation_limits_rad is not None:
            absolute_limits.update({f"{key}.pos": value for key, value in self.config.tcp_rotation_limits_rad.items()})

        return self._clip_by_limits(goal, absolute_limits if absolute_limits else None)

    def _clip_gripper_target(self, target_mm: float) -> float:
        present_mm = self._require_pika_device().get_gripper_distance()
        clipped_target = min(max(target_mm, self.config.gripper_min_mm), self.config.gripper_max_mm)

        if self.config.gripper_max_relative_target_mm is not None:
            lower = present_mm - self.config.gripper_max_relative_target_mm
            upper = present_mm + self.config.gripper_max_relative_target_mm
            clipped_target = min(max(clipped_target, lower), upper)

        return clipped_target

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self.config.control_mode == URPikaControlMode.JOINT:
            observation: RobotObservation = self._get_joint_observation()
        else:
            observation = self._get_tcp_observation()

        observation[GRIPPER_KEY] = self._require_pika_device().get_gripper_distance()
        for camera_key, camera in self.cameras.items():
            observation[camera_key] = camera.read_latest()

        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._ensure_robot_state_is_safe()

        if GRIPPER_KEY not in action:
            raise KeyError(f"Action is missing required key '{GRIPPER_KEY}'.")

        if self.config.control_mode == URPikaControlMode.JOINT:
            arm_action = self._clip_joint_action(action)
            urscript = build_movej(
                joints_rad=[arm_action[key] for key in UR_JOINT_KEYS],
                a=self.config.movej_acceleration,
                v=self.config.movej_velocity,
            )
        else:
            arm_action = self._clip_tcp_action(action)
            urscript = build_movel(
                pose=[arm_action[key] for key in TCP_KEYS],
                a=self.config.movel_acceleration,
                v=self.config.movel_velocity,
            )

        gripper_target_mm = self._clip_gripper_target(float(action[GRIPPER_KEY]))

        send_urscript(
            script=urscript,
            host=self.config.robot_ip,
            port=self.config.command_port,
            timeout=self.config.socket_timeout_s,
        )
        self._require_pika_device().set_gripper_distance(gripper_target_mm)

        return {**arm_action, GRIPPER_KEY: gripper_target_mm}

    def _disconnect_impl(self) -> None:
        disconnect_errors: list[Exception] = []

        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:  # nosec B110
                disconnect_errors.append(exc)

        if self._pika_device is not None:
            try:
                if self.config.disable_gripper_on_disconnect:
                    self._pika_device.disable_gripper()
            except Exception as exc:  # nosec B110
                disconnect_errors.append(exc)
            finally:
                release_shared_pika_device(self._pika_device)
                self._pika_device = None

        if self._state_reader is not None:
            try:
                self._state_reader.stop()
            except Exception as exc:  # nosec B110
                disconnect_errors.append(exc)
            finally:
                self._state_reader = None

        if disconnect_errors:
            logger.warning("%s disconnected with cleanup errors: %s", self, disconnect_errors)
        else:
            logger.info("%s disconnected.", self)

    def disconnect(self) -> None:
        self._disconnect_impl()


# Backward-compatible alias for any local code already importing the previous class name.
URPikaRobot = URPika
