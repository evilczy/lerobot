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

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

MESSAGE_TYPE_ROBOT_STATE = 16

ROBOT_STATE_PACKAGE_TYPE_ROBOT_MODE_DATA = 0
ROBOT_STATE_PACKAGE_TYPE_JOINT_DATA = 1
ROBOT_STATE_PACKAGE_TYPE_CARTESIAN_INFO = 4

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))


@dataclass
class URRobotModeData:
    is_emergency_stopped: bool = False
    is_protective_stopped: bool = False
    is_program_running: bool = False


@dataclass
class URRobotState:
    joint_positions_rad: list[float] = field(default_factory=list)
    tcp_pose: list[float] = field(default_factory=list)
    robot_mode: URRobotModeData = field(default_factory=URRobotModeData)
    received_at: float = 0.0


def _format_vector(values: list[float], digits: int = 6) -> str:
    return ", ".join(f"{value:.{digits}f}" for value in values)


def build_movej(joints_rad: list[float], a: float = 1.4, v: float = 1.05, t: float = 0.0, r: float = 0.0) -> str:
    return f"movej([{_format_vector(joints_rad)}], a={a}, v={v}, t={t}, r={r})"


def build_movel(pose: list[float], a: float = 1.2, v: float = 0.25, t: float = 0.0, r: float = 0.0) -> str:
    return f"movel(p[{_format_vector(pose)}], a={a}, v={v}, t={t}, r={r})"


def send_urscript(script: str, host: str, port: int, timeout: float = 2.0) -> None:
    if not script.endswith("\n"):
        script += "\n"

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(script.encode("utf-8"))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            raise ConnectionError("Socket closed by robot controller.")
        buffer.extend(chunk)
    return bytes(buffer)


def _read_message(sock: socket.socket) -> tuple[int, bytes]:
    header = _recv_exact(sock, 5)
    message_size, message_type = struct.unpack("!IB", header)
    payload = _recv_exact(sock, message_size - 5)
    return message_type, payload


def _parse_robot_mode_data(payload: bytes) -> URRobotModeData:
    (
        _timestamp,
        _is_real_robot_connected,
        _is_real_robot_enabled,
        _is_robot_power_on,
        is_emergency_stopped,
        is_protective_stopped,
        is_program_running,
        _is_program_paused,
        _robot_mode,
        _control_mode,
        _target_speed_fraction,
        _speed_scaling,
        _target_speed_fraction_limit,
        _reserved,
    ) = struct.unpack("!Q???????BBdddB", payload[:42])

    return URRobotModeData(
        is_emergency_stopped=is_emergency_stopped,
        is_protective_stopped=is_protective_stopped,
        is_program_running=is_program_running,
    )


def _parse_joint_data(payload: bytes) -> list[float]:
    joint_format = "!dddffffB"
    joint_size = struct.calcsize(joint_format)
    offset = 0
    joint_positions: list[float] = []

    for _joint_name in JOINT_NAMES:
        q_actual = struct.unpack_from(joint_format, payload, offset)[0]
        joint_positions.append(q_actual)
        offset += joint_size

    return joint_positions


def _parse_cartesian_info(payload: bytes) -> list[float]:
    return list(struct.unpack("!dddddd", payload[:48]))


def parse_robot_state_message(payload: bytes) -> URRobotState:
    state = URRobotState(received_at=time.perf_counter())
    offset = 0

    while offset + 5 <= len(payload):
        package_size = struct.unpack_from("!I", payload, offset)[0]
        if package_size < 5 or offset + package_size > len(payload):
            break

        package_type = payload[offset + 4]
        package_payload = payload[offset + 5 : offset + package_size]

        if package_type == ROBOT_STATE_PACKAGE_TYPE_ROBOT_MODE_DATA:
            state.robot_mode = _parse_robot_mode_data(package_payload)
        elif package_type == ROBOT_STATE_PACKAGE_TYPE_JOINT_DATA:
            state.joint_positions_rad = _parse_joint_data(package_payload)
        elif package_type == ROBOT_STATE_PACKAGE_TYPE_CARTESIAN_INFO:
            state.tcp_pose = _parse_cartesian_info(package_payload)

        offset += package_size

    return state


class URStateReader:
    def __init__(
        self,
        host: str,
        port: int,
        poll_interval_s: float = 0.05,
        socket_timeout_s: float = 2.0,
        reconnect_interval_s: float = 0.5,
    ):
        self.host = host
        self.port = port
        self.poll_interval_s = poll_interval_s
        self.socket_timeout_s = socket_timeout_s
        self.reconnect_interval_s = reconnect_interval_s

        self._latest_state: URRobotState | None = None
        self._latest_state_lock = threading.Lock()
        self._first_state_event = threading.Event()
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ur-state-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._connected.clear()

    def wait_for_first_state(self, timeout_s: float) -> bool:
        return self._first_state_event.wait(timeout_s)

    def get_latest_state(self, max_age_s: float = 1.0) -> URRobotState:
        with self._latest_state_lock:
            if self._latest_state is None:
                raise RuntimeError("No UR robot state available yet.")
            state = self._latest_state

        age_s = time.perf_counter() - state.received_at
        if age_s > max_age_s:
            raise TimeoutError(f"Latest UR robot state is too old: {age_s:.3f}s (max {max_age_s:.3f}s)")

        return state

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=self.socket_timeout_s) as sock:
                    sock.settimeout(self.socket_timeout_s)
                    self._connected.set()

                    while not self._stop_event.is_set():
                        message_type, payload = _read_message(sock)
                        if message_type != MESSAGE_TYPE_ROBOT_STATE:
                            continue

                        state = parse_robot_state_message(payload)
                        with self._latest_state_lock:
                            self._latest_state = state
                        self._first_state_event.set()

                        if self.poll_interval_s > 0 and self._stop_event.wait(self.poll_interval_s):
                            break

            except Exception as exc:
                self.last_error = exc
                self._connected.clear()
                if self._stop_event.wait(self.reconnect_interval_s):
                    break


def rad_to_deg(values: list[float]) -> list[float]:
    return [math.degrees(value) for value in values]
