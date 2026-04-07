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
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_pika import PikaCameraConfig, PikaCameraSource
from .shared import (
    SharedPikaDevice,
    acquire_shared_pika_device,
    find_pika_camera_infos,
    release_shared_pika_device,
)

logger = logging.getLogger(__name__)


class PikaCamera(Camera):
    config_class = PikaCameraConfig

    def __init__(self, config: PikaCameraConfig):
        super().__init__(config)
        self.config = config
        self.source = config.source
        self.color_mode = config.color_mode
        self.rotation = get_cv2_rotation(config.rotation)
        self.warmup_s = config.warmup_s

        self.shared_device: SharedPikaDevice | None = None
        self.backend_camera: Any | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event = Event()

    def __str__(self) -> str:
        port = self.shared_device.port if self.shared_device is not None else self.config.port
        return f"{self.__class__.__name__}({self.source}@{port})"

    @property
    def is_connected(self) -> bool:
        return (
            self.shared_device is not None
            and self.shared_device.is_connected
            and self.thread is not None
            and self.thread.is_alive()
        )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return find_pika_camera_infos()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        try:
            self.shared_device = acquire_shared_pika_device(self.config.port)
            self.shared_device.ensure_camera_params(
                width=int(self.config.width),
                height=int(self.config.height),
                fps=int(self.config.fps),
                fisheye_thread_fps=self.config.fisheye_thread_fps,
            )

            if self.source == PikaCameraSource.FISHEYE:
                self.backend_camera = self.shared_device.get_fisheye_camera(self.config.fisheye_index)
            elif self.source == PikaCameraSource.REALSENSE_COLOR:
                self.backend_camera = self.shared_device.get_realsense_camera(
                    self.config.realsense_serial_number
                )
            else:
                raise ValueError(f"Unsupported Pika camera source: {self.source}")

            self._start_read_thread()

            if warmup and self.warmup_s > 0:
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    self.async_read(timeout_ms=max(self.warmup_s * 1000, 250))
                    time.sleep(0.05)
                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")

            logger.info("%s connected.", self)
        except Exception:
            self._cleanup_failed_connect()
            raise

    def _cleanup_failed_connect(self) -> None:
        if self.thread is not None:
            self._stop_read_thread()
        self.backend_camera = None
        release_shared_pika_device(self.shared_device)
        self.shared_device = None

    def _read_from_hardware(self) -> NDArray[Any]:
        if self.backend_camera is None:
            raise DeviceNotConnectedError(f"{self} backend camera is not initialized")

        if self.source == PikaCameraSource.FISHEYE:
            success, frame = self.backend_camera.get_frame()
        elif self.source == PikaCameraSource.REALSENSE_COLOR:
            success, frame = self.backend_camera.get_color_frame()
        else:
            raise ValueError(f"Unsupported Pika camera source: {self.source}")

        if not success or frame is None:
            raise RuntimeError(f"{self} failed to read a frame from the Pika SDK.")

        return frame

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape
        if h != self.height or w != self.width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.width} or height={self.height}."
            )
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if self.color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self} stop_event is not initialized before starting read loop.")

        failure_count = 0
        has_captured_frame = False
        while not self.stop_event.is_set():
            try:
                frame = self._postprocess_image(self._read_from_hardware())
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0
                has_captured_frame = True

                if self.source == PikaCameraSource.FISHEYE and self.fps:
                    time.sleep(max(1.0 / float(self.fps), 0.001))

            except DeviceNotConnectedError:
                break
            except Exception as exc:
                failure_count += 1
                if failure_count <= 10 or not has_captured_frame:
                    logger.warning("Error reading frame in background thread for %s: %s", self, exc)
                    time.sleep(0.05)
                    continue
                if failure_count <= 20:
                    logger.warning("Error reading frame in background thread for %s: %s", self, exc)
                raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from exc

    def _start_read_thread(self) -> None:
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        if self.shared_device is None and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        self.backend_camera = None
        release_shared_pika_device(self.shared_device)
        self.shared_device = None
        logger.info("%s disconnected.", self)
