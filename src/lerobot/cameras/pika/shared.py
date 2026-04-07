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

import glob
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from typing import Any

logger = logging.getLogger(__name__)

_SHARED_DEVICES_LOCK = Lock()
_SHARED_DEVICES: dict[str, "SharedPikaDevice"] = {}


def ensure_pika_sdk_importable() -> Path:
    """Add the vendored Pika SDK to `sys.path` if available."""
    pika_sdk_root = Path(__file__).resolve().parents[4] / "third_party" / "pika_sdk"
    if not pika_sdk_root.exists():
        raise ImportError(f"Pika SDK not found at {pika_sdk_root}")

    pika_sdk_root_str = str(pika_sdk_root)
    if pika_sdk_root_str not in sys.path:
        sys.path.insert(0, pika_sdk_root_str)

    return pika_sdk_root


def _import_gripper_class() -> type[Any]:
    ensure_pika_sdk_importable()
    from pika.gripper import Gripper

    return Gripper


def _run_command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _list_existing_devices(patterns: list[str]) -> list[str]:
    devices: list[str] = []
    for pattern in patterns:
        devices.extend(glob.glob(pattern))
    return sorted(set(devices))


def _extract_trailing_number(path_or_name: str) -> int | None:
    match = re.search(r"(\d+)$", path_or_name)
    return int(match.group(1)) if match else None


def _normalize_video_device(device: str | int) -> int:
    if isinstance(device, int):
        return device

    device_str = str(device).strip()
    if device_str.startswith("/dev/video"):
        index = _extract_trailing_number(device_str)
        if index is None:
            raise ValueError(f"Could not parse video index from '{device_str}'")
        return index

    return int(device_str)


def resolve_gripper_port(port: str | None = None) -> str:
    if port:
        return port

    env_port = os.getenv("PIKA_GRIPPER_PORT")
    if env_port:
        return env_port

    preferred_ports = [f"/dev/ttyUSB{index}" for index in range(81, 90)]
    preferred_ports += ["/dev/ttyUSB80", "/dev/ttyUSB0"]
    for candidate in preferred_ports:
        if os.path.exists(candidate):
            return candidate

    tty_devices = _list_existing_devices(["/dev/ttyUSB*", "/dev/ttyACM*"])
    if tty_devices:
        return tty_devices[0]

    raise RuntimeError(
        "Could not find a Pika gripper serial port. Set `PIKA_GRIPPER_PORT` or pass `port=` explicitly."
    )


def resolve_realsense_serial(serial_number: str | None = None) -> str:
    if serial_number:
        return serial_number

    env_serial = os.getenv("PIKA_REALSENSE_SERIAL")
    if env_serial:
        return env_serial.strip()

    serials: list[str] = []

    try:
        import pyrealsense2 as rs

        context = rs.context()
        for device in context.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            if serial and serial not in serials:
                serials.append(serial)
    except Exception:
        pass

    output = _run_command(["rs-enumerate-devices", "-s"])
    if output:
        for line in output.splitlines():
            match = re.search(r"Intel RealSense.*?(\d{6,})", line)
            if match:
                serial = match.group(1)
                if serial not in serials:
                    serials.append(serial)

    if not serials:
        raise RuntimeError(
            "Could not auto-detect a RealSense serial number. Set `PIKA_REALSENSE_SERIAL` or pass it explicitly."
        )

    return serials[0]


def _read_video_device_name(video_index: int) -> str:
    name_path = Path(f"/sys/class/video4linux/video{video_index}/name")
    try:
        return name_path.read_text().strip()
    except Exception:
        return ""


def resolve_fisheye_device(device: str | int | None = None, gripper_port: str | None = None) -> int:
    if device is not None:
        return _normalize_video_device(device)

    env_device = os.getenv("PIKA_FISHEYE_DEVICE")
    if env_device:
        return _normalize_video_device(env_device)

    if gripper_port is not None:
        port_index = _extract_trailing_number(gripper_port)
        if port_index is not None and port_index >= 80 and os.path.exists(f"/dev/video{port_index}"):
            return port_index

    video_devices = sorted(Path("/dev").glob("video*"), key=lambda path: path.name)
    if not video_devices:
        raise RuntimeError(
            "Could not auto-detect a Pika fisheye camera. Set `PIKA_FISHEYE_DEVICE` or pass `fisheye_index=` explicitly."
        )

    non_realsense_candidates = []
    for device_path in video_devices:
        index = _extract_trailing_number(device_path.name)
        if index is None:
            continue
        device_name = _read_video_device_name(index).lower()
        if "realsense" not in device_name:
            non_realsense_candidates.append(index)

    if non_realsense_candidates:
        return non_realsense_candidates[0]

    first_index = _extract_trailing_number(video_devices[0].name)
    if first_index is None:
        raise RuntimeError("Could not parse a usable fisheye camera index from /dev/video devices.")

    return first_index


def find_pika_camera_infos() -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []

    try:
        gripper_port = resolve_gripper_port()
    except Exception as exc:
        logger.debug("Skipping Pika camera discovery: %s", exc)
        return infos

    try:
        fisheye_index = resolve_fisheye_device(gripper_port=gripper_port)
        infos.append(
            {
                "name": "Pika Fisheye",
                "type": "Pika",
                "id": f"{gripper_port}:fisheye",
                "gripper_port": gripper_port,
                "source": "fisheye",
                "default_stream_profile": {"width": 640, "height": 480, "fps": 30, "device": fisheye_index},
            }
        )
    except Exception as exc:
        logger.debug("Fisheye detection failed: %s", exc)

    try:
        serial = resolve_realsense_serial()
        infos.append(
            {
                "name": "Pika RealSense Color",
                "type": "Pika",
                "id": f"{gripper_port}:realsense_color",
                "gripper_port": gripper_port,
                "source": "realsense_color",
                "default_stream_profile": {"width": 640, "height": 480, "fps": 30, "serial_number": serial},
            }
        )
    except Exception as exc:
        logger.debug("RealSense detection failed: %s", exc)

    return infos


@dataclass
class SharedPikaDevice:
    port: str
    gripper: Any
    ref_count: int = 0
    io_lock: RLock = field(default_factory=RLock)
    state_lock: Lock = field(default_factory=Lock)
    camera_params: tuple[int, int, int, int] | None = None
    fisheye_index: int | None = None
    realsense_serial: str | None = None

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self.gripper, "is_connected", False))

    def acquire(self) -> "SharedPikaDevice":
        with self.state_lock:
            if self.ref_count == 0:
                if not self.gripper.connect():
                    raise ConnectionError(f"Failed to connect to Pika device on {self.port}")
            self.ref_count += 1
        return self

    def release(self) -> int:
        with self.state_lock:
            if self.ref_count == 0:
                return 0

            self.ref_count -= 1
            remaining = self.ref_count

        if remaining == 0 and self.is_connected:
            self.gripper.disconnect()

        return remaining

    def ensure_camera_params(
        self,
        width: int,
        height: int,
        fps: int,
        fisheye_thread_fps: int = 100,
    ) -> None:
        requested = (width, height, fps, fisheye_thread_fps)
        if self.camera_params is not None and self.camera_params != requested:
            raise ValueError(
                "All Pika camera users attached to the same device must share width/height/fps/fisheye_thread_fps."
            )

        with self.io_lock:
            self.gripper.set_camera_param(width, height, fps, fisheye_thread_fps)
        self.camera_params = requested

    def get_fisheye_camera(self, fisheye_index: str | int | None = None) -> Any:
        resolved_index = resolve_fisheye_device(fisheye_index, self.port)
        if self.fisheye_index is not None and self.fisheye_index != resolved_index:
            raise ValueError(
                f"Shared Pika fisheye camera already bound to index {self.fisheye_index}, got {resolved_index}."
            )

        with self.io_lock:
            self.gripper.set_fisheye_camera_index(resolved_index)
            camera = self.gripper.get_fisheye_camera()

        if camera is None:
            raise ConnectionError(f"Failed to initialize Pika fisheye camera on index {resolved_index}")

        self.fisheye_index = resolved_index
        return camera

    def get_realsense_camera(self, serial_number: str | None = None) -> Any:
        resolved_serial = resolve_realsense_serial(serial_number)
        if self.realsense_serial is not None and self.realsense_serial != resolved_serial:
            raise ValueError(
                f"Shared Pika RealSense camera already bound to serial {self.realsense_serial}, got {resolved_serial}."
            )

        with self.io_lock:
            self.gripper.set_realsense_serial_number(resolved_serial)
            camera = self.gripper.get_realsense_camera()

        if camera is None:
            raise ConnectionError(f"Failed to initialize Pika RealSense camera with serial {resolved_serial}")

        self.realsense_serial = resolved_serial
        return camera

    def enable_gripper(self) -> None:
        with self.io_lock:
            if not self.gripper.enable():
                raise RuntimeError(f"Failed to enable Pika gripper on {self.port}")

    def disable_gripper(self) -> None:
        with self.io_lock:
            if self.is_connected:
                self.gripper.disable()

    def get_gripper_distance(self) -> float:
        return float(self.gripper.get_gripper_distance())

    def set_gripper_distance(self, target_mm: float) -> None:
        with self.io_lock:
            if not self.gripper.set_gripper_distance(target_mm):
                raise RuntimeError(f"Failed to command Pika gripper distance to {target_mm:.2f} mm")


def acquire_shared_pika_device(port: str | None = None) -> SharedPikaDevice:
    resolved_port = resolve_gripper_port(port)

    with _SHARED_DEVICES_LOCK:
        device = _SHARED_DEVICES.get(resolved_port)
        if device is None:
            gripper_cls = _import_gripper_class()
            device = SharedPikaDevice(port=resolved_port, gripper=gripper_cls(port=resolved_port))
            _SHARED_DEVICES[resolved_port] = device

    return device.acquire()


def release_shared_pika_device(device: SharedPikaDevice | None) -> None:
    if device is None:
        return

    remaining_refs = device.release()
    if remaining_refs != 0:
        return

    with _SHARED_DEVICES_LOCK:
        if _SHARED_DEVICES.get(device.port) is device:
            _SHARED_DEVICES.pop(device.port, None)
