#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import socket
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.pika import PikaCameraConfig, PikaCameraSource
from lerobot.cameras.pika.shared import (
    acquire_shared_pika_device,
    release_shared_pika_device,
    resolve_gripper_port,
    resolve_realsense_serial,
)
from lerobot.robots.ur_pika import URPika, URPikaConfig

UR_COMMAND_PORT = 30001
UR_STATE_PORT = 30012
JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
GRIPPER_KEY = "gripper.pos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR + Pika hardware smoke test")
    parser.add_argument("--robot-ip", default=os.getenv("UR7E_HOST", "192.168.1.15"))
    parser.add_argument("--gripper-port", default=os.getenv("PIKA_GRIPPER_PORT"))
    parser.add_argument("--realsense-serial", default=os.getenv("PIKA_REALSENSE_SERIAL"))
    parser.add_argument("--camera-key", default="right_wrist_0_rgb")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--socket-timeout-s", type=float, default=2.0)
    parser.add_argument("--depth-attempts", type=int, default=20)
    parser.add_argument("--depth-sleep-s", type=float, default=0.2)
    parser.add_argument(
        "--send-noop-action",
        action="store_true",
        help="Send a no-op action using the current joint and gripper values to validate the command path.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("./calibration/ur_pika"),
    )
    return parser.parse_args()


def probe_tcp_port(host: str, port: int, timeout_s: float) -> float:
    start = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout_s):
        pass
    return time.perf_counter() - start


def format_ok(label: str, detail: str) -> None:
    print(f"[OK] {label}: {detail}")


def format_fail(label: str, detail: str) -> None:
    print(f"[FAIL] {label}: {detail}")


def validate_observation(obs: dict[str, object], camera_key: str, width: int, height: int) -> None:
    missing_state_keys = [key for key in (*JOINT_KEYS, GRIPPER_KEY) if key not in obs]
    if missing_state_keys:
        raise RuntimeError(f"Observation missing keys: {missing_state_keys}")

    for key in JOINT_KEYS:
        value = float(obs[key])
        if not math.isfinite(value):
            raise RuntimeError(f"{key} is not finite: {value}")

    gripper_mm = float(obs[GRIPPER_KEY])
    if not math.isfinite(gripper_mm):
        raise RuntimeError(f"{GRIPPER_KEY} is not finite: {gripper_mm}")

    if camera_key not in obs:
        raise RuntimeError(f"Observation missing camera key: {camera_key}")

    frame = obs[camera_key]
    if not isinstance(frame, np.ndarray):
        raise RuntimeError(f"{camera_key} is not a numpy array: {type(frame)}")
    if frame.shape != (height, width, 3):
        raise RuntimeError(f"{camera_key} shape mismatch: got {frame.shape}, expected {(height, width, 3)}")
    if frame.dtype != np.uint8:
        raise RuntimeError(f"{camera_key} dtype mismatch: got {frame.dtype}, expected uint8")


def probe_realsense_depth(gripper_port: str, realsense_serial: str, attempts: int, sleep_s: float) -> tuple[int, tuple[int, ...]]:
    shared = acquire_shared_pika_device(gripper_port)
    try:
        realsense_camera = shared.get_realsense_camera(realsense_serial)
        for index in range(attempts):
            ok_depth, depth = realsense_camera.get_depth_frame()
            if ok_depth and depth is not None:
                return index + 1, depth.shape
            time.sleep(sleep_s)
    finally:
        release_shared_pika_device(shared)

    raise RuntimeError(f"Failed to get a RealSense depth frame after {attempts} attempts")


def send_noop_action(robot: URPika, obs: dict[str, object]) -> None:
    action = {key: float(obs[key]) for key in JOINT_KEYS}
    action[GRIPPER_KEY] = float(obs[GRIPPER_KEY])
    robot.send_action(action)


def main() -> int:
    args = parse_args()

    gripper_port = resolve_gripper_port(args.gripper_port)
    realsense_serial = resolve_realsense_serial(args.realsense_serial)

    print("=== UR + Pika smoke test ===")
    print(f"robot_ip={args.robot_ip}")
    print(f"gripper_port={gripper_port}")
    print(f"realsense_serial={realsense_serial}")
    print(f"camera_key={args.camera_key}")

    command_latency_s = probe_tcp_port(args.robot_ip, UR_COMMAND_PORT, args.socket_timeout_s)
    format_ok("UR command socket", f"{args.robot_ip}:{UR_COMMAND_PORT} reachable in {command_latency_s * 1000:.1f} ms")

    state_latency_s = probe_tcp_port(args.robot_ip, UR_STATE_PORT, args.socket_timeout_s)
    format_ok("UR state socket", f"{args.robot_ip}:{UR_STATE_PORT} reachable in {state_latency_s * 1000:.1f} ms")

    robot = URPika(
        URPikaConfig(
            robot_ip=args.robot_ip,
            gripper_port=gripper_port,
            control_mode="joint",
            socket_timeout_s=args.socket_timeout_s,
            calibration_dir=args.calibration_dir,
            cameras={
                args.camera_key: PikaCameraConfig(
                    source=PikaCameraSource.REALSENSE_COLOR,
                    port=gripper_port,
                    realsense_serial_number=realsense_serial,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    warmup_s=args.warmup_s,
                )
            },
        )
    )

    try:
        connect_start = time.perf_counter()
        robot.connect()
        connect_elapsed_s = time.perf_counter() - connect_start
        format_ok("robot.connect", f"completed in {connect_elapsed_s:.2f} s")

        obs = robot.get_observation()
        validate_observation(obs, args.camera_key, args.width, args.height)
        format_ok(
            "robot.get_observation",
            f"joints OK, gripper={float(obs[GRIPPER_KEY]):.2f} mm, frame={obs[args.camera_key].shape}",
        )

        depth_attempt, depth_shape = probe_realsense_depth(
            gripper_port=gripper_port,
            realsense_serial=realsense_serial,
            attempts=args.depth_attempts,
            sleep_s=args.depth_sleep_s,
        )
        format_ok("RealSense depth", f"frame={depth_shape}, acquired on attempt {depth_attempt}")

        if args.send_noop_action:
            send_noop_action(robot, obs)
            format_ok("robot.send_action", "no-op action sent successfully")
        else:
            print("[SKIP] robot.send_action: not executed; pass --send-noop-action to validate the command path")

        print("=== smoke test passed ===")
        return 0

    except Exception as exc:
        format_fail("smoke test", str(exc))
        return 1

    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
                format_ok("robot.disconnect", "resources released")
        except Exception as exc:
            format_fail("robot.disconnect", str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
