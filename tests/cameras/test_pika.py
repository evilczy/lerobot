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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.pika import PikaCamera, PikaCameraConfig, PikaCameraSource
from lerobot.utils.errors import DeviceNotConnectedError


def _make_backend_camera(height: int, width: int):
    frame = np.full((height, width, 3), 123, dtype=np.uint8)
    backend = MagicMock(name="PikaBackendCamera")
    backend.get_frame.return_value = (True, frame)
    backend.get_color_frame.return_value = (True, frame)
    return backend


def _make_shared_device(config: PikaCameraConfig) -> MagicMock:
    shared = MagicMock(name="SharedPikaDevice")
    shared.port = config.port or "/dev/ttyUSB81"
    shared.is_connected = True
    shared.ensure_camera_params.return_value = None
    shared.get_fisheye_camera.return_value = _make_backend_camera(config.height, config.width)
    shared.get_realsense_camera.return_value = _make_backend_camera(config.height, config.width)
    return shared


@pytest.mark.parametrize(
    "source",
    [PikaCameraSource.FISHEYE, PikaCameraSource.REALSENSE_COLOR],
    ids=["fisheye", "realsense_color"],
)
def test_connect_read_and_disconnect(source):
    config = PikaCameraConfig(
        source=source,
        width=640,
        height=480,
        fps=30,
        warmup_s=0,
    )
    shared_device = _make_shared_device(config)

    with (
        patch(
            "lerobot.cameras.pika.camera_pika.acquire_shared_pika_device",
            return_value=shared_device,
        ),
        patch("lerobot.cameras.pika.camera_pika.release_shared_pika_device"),
    ):
        camera = PikaCamera(config)
        camera.connect()

        assert camera.is_connected
        frame = camera.read()
        latest = camera.read_latest(max_age_ms=1000)

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        np.testing.assert_array_equal(frame, latest)

        camera.disconnect()
        assert not camera.is_connected


def test_disconnect_before_connect():
    config = PikaCameraConfig(source=PikaCameraSource.FISHEYE, width=640, height=480, fps=30)
    camera = PikaCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_fisheye_connect_tolerates_slow_first_frame():
    config = PikaCameraConfig(
        source=PikaCameraSource.FISHEYE,
        width=640,
        height=480,
        fps=30,
        warmup_s=1.0,
    )
    shared_device = _make_shared_device(config)
    backend = _make_backend_camera(config.height, config.width)
    delayed_frames = [(False, None)] * 11 + [backend.get_frame.return_value]
    backend.get_frame.side_effect = delayed_frames + [backend.get_frame.return_value]
    shared_device.get_fisheye_camera.return_value = backend

    with (
        patch(
            "lerobot.cameras.pika.camera_pika.acquire_shared_pika_device",
            return_value=shared_device,
        ),
        patch("lerobot.cameras.pika.camera_pika.release_shared_pika_device"),
    ):
        camera = PikaCamera(config)
        camera.connect()

        latest = camera.read_latest(max_age_ms=1000)
        assert latest.shape == (480, 640, 3)

        camera.disconnect()
        assert not camera.is_connected


def test_find_cameras_delegates_to_shared_discovery():
    camera_infos = [{"id": "ttyUSB81:fisheye", "type": "Pika"}]
    with patch("lerobot.cameras.pika.camera_pika.find_pika_camera_infos", return_value=camera_infos):
        assert PikaCamera.find_cameras() == camera_infos
