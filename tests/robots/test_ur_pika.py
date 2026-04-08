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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots import make_robot_from_config
from lerobot.robots.ur_pika import URPika, URPikaConfig, URPikaControlMode


def _make_config(*, calibration_dir, **kwargs) -> URPikaConfig:
    return URPikaConfig(robot_ip="192.168.0.2", calibration_dir=calibration_dir, **kwargs)


def _make_camera(name: str) -> MagicMock:
    camera = MagicMock(name=f"Camera:{name}")
    camera.is_connected = False
    camera.width = 640
    camera.height = 480

    def _connect():
        camera.is_connected = True

    def _disconnect():
        camera.is_connected = False

    camera.connect.side_effect = _connect
    camera.disconnect.side_effect = _disconnect
    camera.read_latest.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return camera


def _make_shared_pika_device() -> MagicMock:
    shared = MagicMock(name="SharedPikaDevice")
    shared.port = "/dev/ttyUSB81"
    shared.is_connected = True
    shared.enable_gripper.return_value = None
    shared.disable_gripper.return_value = None
    shared.get_gripper_distance.return_value = 12.5
    shared.set_gripper_distance.return_value = None
    return shared


def _make_state(
    *,
    joints: list[float] | None = None,
    tcp_pose: list[float] | None = None,
    protective_stop: bool = False,
    emergency_stop: bool = False,
):
    return SimpleNamespace(
        joint_positions_rad=joints or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        tcp_pose=tcp_pose or [0.11, 0.22, 0.33, 0.01, 0.02, 0.03],
        robot_mode=SimpleNamespace(
            is_emergency_stopped=emergency_stop,
            is_protective_stopped=protective_stop,
            is_program_running=True,
        ),
        received_at=0.0,
    )


def _make_state_reader(state) -> MagicMock:
    reader = MagicMock(name="URStateReader")
    reader.is_connected = False

    def _start():
        reader.is_connected = True

    def _stop():
        reader.is_connected = False

    reader.start.side_effect = _start
    reader.stop.side_effect = _stop
    reader.wait_for_first_state.return_value = True
    reader.get_latest_state.return_value = state
    return reader


@pytest.fixture
def patched_joint_robot(tmp_path):
    state = _make_state()
    reader = _make_state_reader(state)
    shared_device = _make_shared_pika_device()
    cameras = {
        "right_wrist_0_rgb": _make_camera("right_wrist_0_rgb"),
    }

    with (
        patch(
            "lerobot.robots.ur_pika.robot_ur_pika.make_cameras_from_configs",
            return_value=cameras,
        ),
        patch(
            "lerobot.robots.ur_pika.robot_ur_pika.acquire_shared_pika_device",
            return_value=shared_device,
        ),
        patch("lerobot.robots.ur_pika.robot_ur_pika.release_shared_pika_device"),
        patch("lerobot.robots.ur_pika.robot_ur_pika.URStateReader", return_value=reader),
        patch("lerobot.robots.ur_pika.robot_ur_pika.send_urscript") as send_urscript,
    ):
        robot = URPika(_make_config(calibration_dir=tmp_path))
        yield robot, reader, shared_device, cameras, send_urscript
        if robot.is_connected:
            robot.disconnect()


def test_factory_instantiates_ur_pika(tmp_path):
    robot = make_robot_from_config(_make_config(calibration_dir=tmp_path))
    assert isinstance(robot, URPika)


def test_connect_get_observation_and_disconnect(patched_joint_robot):
    robot, reader, shared_device, cameras, _send_urscript = patched_joint_robot

    robot.connect()
    assert robot.is_connected
    reader.start.assert_called_once()
    shared_device.enable_gripper.assert_called_once()

    observation = robot.get_observation()
    expected_keys = {
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "gripper.pos",
        "right_wrist_0_rgb",
    }
    assert set(observation) == expected_keys
    assert observation["gripper.pos"] == 12.5

    robot.disconnect()
    assert not robot.is_connected
    reader.stop.assert_called_once()
    shared_device.disable_gripper.assert_called_once()
    for camera in cameras.values():
        camera.disconnect.assert_called_once()


def test_send_action_joint_mode(patched_joint_robot):
    robot, _reader, shared_device, _cameras, send_urscript = patched_joint_robot
    robot.connect()

    action = {
        "joint_1.pos": 0.15,
        "joint_2.pos": 0.25,
        "joint_3.pos": 0.35,
        "joint_4.pos": 0.45,
        "joint_5.pos": 0.55,
        "joint_6.pos": 0.65,
        "gripper.pos": 18.0,
    }
    sent_action = robot.send_action(action)

    assert sent_action["gripper.pos"] == 18.0
    send_urscript.assert_called_once()
    assert "movej([" in send_urscript.call_args.kwargs["script"]
    shared_device.set_gripper_distance.assert_called_once_with(18.0)


def test_send_action_tcp_mode_uses_movel(tmp_path):
    state = _make_state()
    reader = _make_state_reader(state)
    shared_device = _make_shared_pika_device()
    cameras = {"right_wrist_0_rgb": _make_camera("right_wrist_0_rgb")}

    with (
        patch(
            "lerobot.robots.ur_pika.robot_ur_pika.make_cameras_from_configs",
            return_value=cameras,
        ),
        patch(
            "lerobot.robots.ur_pika.robot_ur_pika.acquire_shared_pika_device",
            return_value=shared_device,
        ),
        patch("lerobot.robots.ur_pika.robot_ur_pika.release_shared_pika_device"),
        patch("lerobot.robots.ur_pika.robot_ur_pika.URStateReader", return_value=reader),
        patch("lerobot.robots.ur_pika.robot_ur_pika.send_urscript") as send_urscript,
    ):
        robot = URPika(
            _make_config(
                calibration_dir=tmp_path,
                control_mode=URPikaControlMode.TCP,
            )
        )
        robot.connect()

        action = {
            "tcp_x.pos": 0.2,
            "tcp_y.pos": 0.3,
            "tcp_z.pos": 0.4,
            "tcp_rx.pos": 0.1,
            "tcp_ry.pos": 0.2,
            "tcp_rz.pos": 0.3,
            "gripper.pos": 20.0,
        }
        sent_action = robot.send_action(action)

        assert sent_action["gripper.pos"] == 20.0
        assert "movel(p[" in send_urscript.call_args.kwargs["script"]
        robot.disconnect()


def test_send_action_rejects_protective_stop(patched_joint_robot):
    robot, reader, _shared_device, _cameras, _send_urscript = patched_joint_robot
    reader.get_latest_state.return_value = _make_state(protective_stop=True)
    robot.connect()

    with pytest.raises(RuntimeError, match="protective stopped"):
        robot.send_action(
            {
                "joint_1.pos": 0.15,
                "joint_2.pos": 0.25,
                "joint_3.pos": 0.35,
                "joint_4.pos": 0.45,
                "joint_5.pos": 0.55,
                "joint_6.pos": 0.65,
                "gripper.pos": 18.0,
            }
        )
