#!/usr/bin/env python3
# 坐在背靠董事长办公室，正对机械臂方向，+x朝左侧（东），+y朝后侧（北），+z朝上
# 使用前：sudo chmod 666 /dev/ttyUSB0
# 
import glob
import math
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from device_dector import (
    normalize_video_device as detector_normalize_video_device,
    resolve_fisheye_device as detector_resolve_fisheye_device,
    resolve_gripper_port as detector_resolve_gripper_port,
    resolve_gripper_port_candidates as detector_resolve_gripper_port_candidates,
    resolve_realsense_serial as detector_resolve_realsense_serial,
)

PIKA_SDK_ROOT = Path("./third_party/pika_sdk")
if str(PIKA_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(PIKA_SDK_ROOT))

from pika.gripper import Gripper

UR7E_HOST = os.getenv("UR7E_HOST", "192.168.1.15")
UR7E_COMMAND_PORT = int(os.getenv("UR7E_COMMAND_PORT", "30001"))
UR7E_STATE_PORT = int(os.getenv("UR7E_STATE_PORT", "30012"))

# Pika 设备配置
# 优先级: 这里的显式配置 > 环境变量 > 自动探测
# 留空("")时会回退到自动探测。
DEFAULT_PIKA_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_PIKA_FISHEYE_DEVICE = "1"
# DEFAULT_PIKA_REALSENSE_SERIAL = "315122272459"(遥操作)
DEFAULT_PIKA_REALSENSE_SERIAL = "230322270988"

HOME_JOINTS_DEG = [2, -160, 65, -140, 15, 15]
HOME_JOINTS_RAD = [math.radians(value) for value in HOME_JOINTS_DEG]

LINEAR_STEP_M = 0.05
MOVEJ_A = 1.0
MOVEJ_V = 0.2
MOVEL_A = 0.2
MOVEL_V = 0.02

MESSAGE_TYPE_ROBOT_STATE = 16

ROBOT_STATE_PACKAGE_TYPE_ROBOT_MODE_DATA = 0
ROBOT_STATE_PACKAGE_TYPE_JOINT_DATA = 1
ROBOT_STATE_PACKAGE_TYPE_CARTESIAN_INFO = 4

JOINT_NAMES = (
    "base",
    "shoulder",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

WINDOW_FISHEYE = "Pika Fisheye"
WINDOW_COLOR = "Pika Color"
WINDOW_DEPTH = "Pika Depth"


@dataclass
class CameraConfig:
    width: int
    height: int
    fps: int
    fisheye_device: object
    realsense_serial: str


def _format_vector(values, digits=6):
    return ", ".join(f"{value:.{digits}f}" for value in values)


def _run_command(args):
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return ""

    output_parts = []
    if completed.stdout:
        output_parts.append(completed.stdout)
    if completed.stderr:
        output_parts.append(completed.stderr)
    return "".join(output_parts).strip()


def _device_sort_key(path):
    match = re.search(r"(\d+)$", path)
    if match:
        return int(match.group(1))
    return path


def _list_existing_devices(patterns):
    devices = []
    for pattern in patterns:
        devices.extend(glob.glob(pattern))
    return sorted(set(devices), key=_device_sort_key)


def _extract_trailing_number(value):
    match = re.search(r"(\d+)$", str(value))
    if match:
        return int(match.group(1))
    return None


def _normalize_video_device(value):
    return detector_normalize_video_device(value)


def _configured_value(explicit_value, env_name):
    value = explicit_value if explicit_value not in (None, "") else os.getenv(env_name, "")
    value = str(value).strip() if value is not None else ""
    return value or None


def _get_udev_properties(devnode):
    output = _run_command(["udevadm", "info", "-q", "property", "-n", devnode])
    properties = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key] = value
    return properties


def _extract_usb_topology(path_value):
    if not path_value:
        return ()

    match = re.search(r"usb-\d+:([0-9.]+):", path_value)
    if not match:
        return ()

    topology = []
    for part in match.group(1).split("."):
        if part.isdigit():
            topology.append(int(part))
    return tuple(topology)


def _common_prefix_length(left, right):
    length = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        length += 1
    return length


def _score_video_candidate(candidate):
    name = f"{candidate['product']} {candidate['model']}".lower()
    score = 0
    if "fisheye" in name:
        score += 50
    if "realsense" in name:
        score -= 100
    if "webcam" in name:
        score -= 20
    if "decxin" in name:
        score += 10
    if ":capture:" in candidate["capabilities"]:
        score += 5
    return score


def _list_video_candidates():
    candidates = []
    for video_dev in _list_existing_devices(["/dev/video*"]):
        props = _get_udev_properties(video_dev)
        if props.get("SUBSYSTEM") != "video4linux":
            continue
        candidate = {
            "devnode": video_dev,
            "index": _extract_trailing_number(video_dev),
            "product": props.get("ID_V4L_PRODUCT", ""),
            "model": props.get("ID_MODEL", ""),
            "capabilities": props.get("ID_V4L_CAPABILITIES", ""),
            "id_path": props.get("ID_PATH", ""),
            "topology": _extract_usb_topology(props.get("ID_PATH") or props.get("DEVPATH", "")),
        }
        candidates.append(candidate)
    return candidates


def _looks_like_realsense(candidate):
    text = f"{candidate['product']} {candidate['model']}".lower()
    return "realsense" in text


def _get_realsense_serials_from_pyrealsense():
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []

    serials = []
    try:
        context = rs.context()
        for device in context.devices:
            serial = device.get_info(rs.camera_info.serial_number)
            if serial:
                serials.append(serial)
    except Exception:
        return []

    return serials


def _get_realsense_serials_from_cli():
    output = _run_command(["rs-enumerate-devices", "-s"])
    if not output:
        return []

    serials = []
    for line in output.splitlines():
        match = re.search(r"Intel RealSense.*?(\d{6,})", line)
        if match:
            serials.append(match.group(1))
    return serials


def resolve_gripper_port():
    configured_port = _configured_value(DEFAULT_PIKA_GRIPPER_PORT, "PIKA_GRIPPER_PORT")
    if configured_port:
        return configured_port
    return detector_resolve_gripper_port()


def resolve_gripper_port_candidates():
    configured_port = _configured_value(DEFAULT_PIKA_GRIPPER_PORT, "PIKA_GRIPPER_PORT")
    if configured_port:
        return [configured_port]
    return detector_resolve_gripper_port_candidates()


def resolve_realsense_serial():
    configured_serial = _configured_value(DEFAULT_PIKA_REALSENSE_SERIAL, "PIKA_REALSENSE_SERIAL")
    if configured_serial:
        return configured_serial
    return detector_resolve_realsense_serial()


def resolve_fisheye_device(gripper_port):
    configured_device = _configured_value(DEFAULT_PIKA_FISHEYE_DEVICE, "PIKA_FISHEYE_DEVICE")
    if configured_device:
        return _normalize_video_device(configured_device)
    return detector_resolve_fisheye_device(gripper_port=gripper_port)


def build_camera_config(width=640, height=480, fps=30, gripper_port=None):
    gripper_port = gripper_port or resolve_gripper_port()
    fisheye_device = resolve_fisheye_device(gripper_port)
    realsense_serial = resolve_realsense_serial()
    return CameraConfig(
        width=width,
        height=height,
        fps=fps,
        fisheye_device=fisheye_device,
        realsense_serial=realsense_serial,
    )


def configure_camera_device(gripper, camera_config):
    gripper.set_camera_param(camera_config.width, camera_config.height, camera_config.fps)
    gripper.set_fisheye_camera_index(camera_config.fisheye_device)
    gripper.set_realsense_serial_number(camera_config.realsense_serial)


def build_program(lines, program_name="fake_policy"):
    body = "\n".join(f"  {line}" for line in lines)
    return f"def {program_name}():\n{body}\nend\n{program_name}()\n"


def build_movej(joints_rad, a=1.4, v=1.05, t=0.0, r=0.0):
    return f"movej([{_format_vector(joints_rad)}], a={a}, v={v}, t={t}, r={r})"


def build_movel(pose, a=1.2, v=0.25, t=0.0, r=0.0):
    return f"movel(p[{_format_vector(pose)}], a={a}, v={v}, t={t}, r={r})"


def send_urscript(script, host, port, timeout=2.0):
    if not script.endswith("\n"):
        script += "\n"
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(script.encode("utf-8"))


def _recv_exact(sock, size):
    buffer = bytearray()
    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))
        if not chunk:
            raise ConnectionError("Socket closed by robot controller.")
        buffer.extend(chunk)
    return bytes(buffer)


def _read_message(sock):
    header = _recv_exact(sock, 5)
    message_size, message_type = struct.unpack("!IB", header)
    payload = _recv_exact(sock, message_size - 5)
    return message_type, payload


def _parse_robot_mode_data(payload):
    (
        timestamp,
        is_real_robot_connected,
        is_real_robot_enabled,
        is_robot_power_on,
        is_emergency_stopped,
        is_protective_stopped,
        is_program_running,
        is_program_paused,
        robot_mode,
        control_mode,
        target_speed_fraction,
        speed_scaling,
        target_speed_fraction_limit,
        _reserved,
    ) = struct.unpack("!Q???????BBdddB", payload[:42])

    return {
        "timestamp_usec": timestamp,
        "is_real_robot_connected": is_real_robot_connected,
        "is_real_robot_enabled": is_real_robot_enabled,
        "is_robot_power_on": is_robot_power_on,
        "is_emergency_stopped": is_emergency_stopped,
        "is_protective_stopped": is_protective_stopped,
        "is_program_running": is_program_running,
        "is_program_paused": is_program_paused,
        "robot_mode": robot_mode,
        "control_mode": control_mode,
        "target_speed_fraction": target_speed_fraction,
        "speed_scaling": speed_scaling,
        "target_speed_fraction_limit": target_speed_fraction_limit,
    }


def _parse_joint_data(payload):
    joint_format = "!dddffffB"
    joint_size = struct.calcsize(joint_format)
    offset = 0
    joints = []

    for joint_name in JOINT_NAMES:
        (
            q_actual,
            q_target,
            qd_actual,
            i_actual,
            v_actual,
            t_motor,
            _reserved,
            joint_mode,
        ) = struct.unpack_from(joint_format, payload, offset)
        joints.append(
            {
                "name": joint_name,
                "q_actual_rad": q_actual,
                "q_actual_deg": math.degrees(q_actual),
                "q_target_rad": q_target,
                "qd_actual_rad_s": qd_actual,
                "i_actual_a": i_actual,
                "v_actual_v": v_actual,
                "t_motor_c": t_motor,
                "joint_mode": joint_mode,
            }
        )
        offset += joint_size

    return {
        "joints": joints,
        "q_actual_rad": [joint["q_actual_rad"] for joint in joints],
        "q_actual_deg": [joint["q_actual_deg"] for joint in joints],
    }


def _parse_cartesian_info(payload):
    (
        x,
        y,
        z,
        rx,
        ry,
        rz,
        tcp_offset_x,
        tcp_offset_y,
        tcp_offset_z,
        tcp_offset_rx,
        tcp_offset_ry,
        tcp_offset_rz,
    ) = struct.unpack("!dddddddddddd", payload[:96])

    return {
        "tcp_pose": {
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "rx_rad": rx,
            "ry_rad": ry,
            "rz_rad": rz,
        },
        "tcp_offset": {
            "x_m": tcp_offset_x,
            "y_m": tcp_offset_y,
            "z_m": tcp_offset_z,
            "rx_rad": tcp_offset_rx,
            "ry_rad": tcp_offset_ry,
            "rz_rad": tcp_offset_rz,
        },
    }


def parse_robot_state_message(payload):
    state = {
        "robot_mode": None,
        "joint_data": None,
        "cartesian_info": None,
    }
    offset = 0

    while offset + 5 <= len(payload):
        package_size = struct.unpack_from("!I", payload, offset)[0]
        if package_size < 5 or offset + package_size > len(payload):
            break

        package_type = payload[offset + 4]
        package_payload = payload[offset + 5 : offset + package_size]

        if package_type == ROBOT_STATE_PACKAGE_TYPE_ROBOT_MODE_DATA:
            state["robot_mode"] = _parse_robot_mode_data(package_payload)
        elif package_type == ROBOT_STATE_PACKAGE_TYPE_JOINT_DATA:
            state["joint_data"] = _parse_joint_data(package_payload)
        elif package_type == ROBOT_STATE_PACKAGE_TYPE_CARTESIAN_INFO:
            state["cartesian_info"] = _parse_cartesian_info(package_payload)

        offset += package_size

    return state


def format_tcp_pose(pose):
    return (
        "TCP pose: "
        f"x={pose[0] * 1000.0:.1f} mm, "
        f"y={pose[1] * 1000.0:.1f} mm, "
        f"z={pose[2] * 1000.0:.1f} mm, "
        f"rx={pose[3]:.4f} rad, "
        f"ry={pose[4]:.4f} rad, "
        f"rz={pose[5]:.4f} rad"
    )


def _sleep_with_stop(stop_event, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if stop_event.is_set():
            return False
        time.sleep(0.05)
    return True


class URStateReader(object):
    def __init__(self, host, port, poll_interval_s=0.1, socket_timeout_s=2.0):
        self.host = host
        self.port = port
        self.poll_interval_s = poll_interval_s
        self.socket_timeout_s = socket_timeout_s
        self.latest_state = None
        self.latest_state_lock = threading.Lock()
        self.first_state_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = None
        self.last_error = None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="ur-state-reader", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def wait_for_first_state(self, timeout_s):
        return self.first_state_event.wait(timeout_s)

    def get_latest_joint_positions_rad(self):
        state = self._get_latest_state()
        joint_data = state.get("joint_data")
        if joint_data is None:
            raise RuntimeError("Latest robot state has no joint_data.")
        return list(joint_data["q_actual_rad"])

    def get_latest_tcp_pose(self):
        state = self._get_latest_state()
        cartesian_info = state.get("cartesian_info")
        if cartesian_info is None:
            raise RuntimeError("Latest robot state has no cartesian_info.")
        tcp_pose = cartesian_info["tcp_pose"]
        return [
            tcp_pose["x_m"],
            tcp_pose["y_m"],
            tcp_pose["z_m"],
            tcp_pose["rx_rad"],
            tcp_pose["ry_rad"],
            tcp_pose["rz_rad"],
        ]

    def wait_for_joint_target(self, target_joints, tolerance_rad=0.03, stable_samples=3, timeout_s=25):
        deadline = time.time() + timeout_s
        stable_count = 0
        while time.time() < deadline:
            actual = self.get_latest_joint_positions_rad()
            if all(abs(current - target) <= tolerance_rad for current, target in zip(actual, target_joints)):
                stable_count += 1
                if stable_count >= stable_samples:
                    return True
            else:
                stable_count = 0
            if not _sleep_with_stop(self.stop_event, self.poll_interval_s):
                break
        raise TimeoutError("Timed out waiting for UR7e joint target.")

    def wait_for_tcp_target(self, target_pose, tolerance_m=0.003, stable_samples=3, timeout_s=15):
        deadline = time.time() + timeout_s
        stable_count = 0
        while time.time() < deadline:
            actual = self.get_latest_tcp_pose()
            pos_ok = all(abs(current - target) <= tolerance_m for current, target in zip(actual[:3], target_pose[:3]))
            rot_ok = all(abs(current - target) <= 0.05 for current, target in zip(actual[3:], target_pose[3:]))
            if pos_ok and rot_ok:
                stable_count += 1
                if stable_count >= stable_samples:
                    return True
            else:
                stable_count = 0
            if not _sleep_with_stop(self.stop_event, self.poll_interval_s):
                break
        raise TimeoutError("Timed out waiting for UR7e TCP target.")

    def _get_latest_state(self):
        with self.latest_state_lock:
            if self.latest_state is None:
                raise RuntimeError("No robot state available yet.")
            return self.latest_state

    def _run(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=self.socket_timeout_s) as sock:
                sock.settimeout(self.socket_timeout_s)
                while not self.stop_event.is_set():
                    message_type, payload = _read_message(sock)
                    if message_type != MESSAGE_TYPE_ROBOT_STATE:
                        continue
                    state = parse_robot_state_message(payload)
                    with self.latest_state_lock:
                        self.latest_state = state
                    self.first_state_event.set()
                    if not _sleep_with_stop(self.stop_event, self.poll_interval_s):
                        break
        except Exception as exc:
            self.last_error = exc


class PikaRig(object):
    def __init__(self):
        self.port_candidates = resolve_gripper_port_candidates()
        if not self.port_candidates:
            raise RuntimeError(
                "未找到任何夹爪串口候选项，请先检查 devices_info.conf、setup.bash 或设置 PIKA_GRIPPER_PORT。"
            )
        self.port = self.port_candidates[0]
        self.gripper = None
        self.connected = False
        self.enabled = False
        self.camera_config = None

    def connect(self):
        attempts = []
        for candidate_port in self.port_candidates:
            attempts.append(candidate_port)
            print(f"尝试夹爪串口: {candidate_port}")
            gripper = Gripper(port=candidate_port)
            if not gripper.connect():
                gripper.disconnect()
                continue
            self.port = candidate_port
            self.gripper = gripper
            self.connected = True
            print(f"使用夹爪串口: {self.port}")
            return

        raise RuntimeError(
            "Failed to connect to Pika gripper. Tried ports: "
            + ", ".join(attempts)
            + ". 如果这些端口都存在但仍然报 Permission denied，请先执行 third_party/pika_sdk/setup.bash "
            + "或检查当前用户是否具备串口访问权限。"
        )

    def enable_gripper(self):
        if not self.gripper.enable():
            raise RuntimeError("Failed to enable Pika gripper motor.")
        self.enabled = True

    def set_gripper_distance_and_wait(self, target_mm, tolerance_mm=2.0, timeout_s=5.0):
        if not self.gripper.set_gripper_distance(target_mm):
            raise RuntimeError(f"Failed to command gripper distance to {target_mm:.1f} mm.")
        self.wait_for_gripper_distance(target_mm, tolerance_mm=tolerance_mm, timeout_s=timeout_s)

    def wait_for_gripper_distance(self, target_mm, tolerance_mm=2.0, timeout_s=5.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            current = self.get_gripper_distance()
            if abs(current - target_mm) <= tolerance_mm:
                return True
            time.sleep(0.1)
        raise TimeoutError(
            f"Timed out waiting for gripper distance {target_mm:.1f} mm, latest {self.get_gripper_distance():.1f} mm."
        )

    def init_cameras(self):
        camera_width = int(os.getenv("PIKA_CAMERA_WIDTH", "640"))
        camera_height = int(os.getenv("PIKA_CAMERA_HEIGHT", "480"))
        camera_fps = int(os.getenv("PIKA_CAMERA_FPS", "30"))
        self.camera_config = build_camera_config(camera_width, camera_height, camera_fps, gripper_port=self.port)

        print(f"使用鱼眼设备: {self.camera_config.fisheye_device}")
        print(f"使用 RealSense 序列号: {self.camera_config.realsense_serial}")

        configure_camera_device(self.gripper, self.camera_config)

        fisheye_camera = self.gripper.get_fisheye_camera()
        if fisheye_camera is None:
            raise RuntimeError("Failed to initialize Pika fisheye camera.")

        realsense_camera = self.gripper.get_realsense_camera()
        if realsense_camera is None:
            raise RuntimeError("Failed to initialize Pika RealSense camera.")

        return fisheye_camera, realsense_camera

    def get_gripper_distance(self):
        if self.gripper is None:
            raise RuntimeError("Pika gripper is not connected.")
        return self.gripper.get_gripper_distance()

    def cleanup(self):
        if self.enabled and self.gripper is not None:
            try:
                self.gripper.disable()
            finally:
                self.enabled = False
        if self.connected and self.gripper is not None:
            self.gripper.disconnect()
            self.connected = False


class CameraDisplayLoop(threading.Thread):
    def __init__(self, stop_event, fisheye_camera, realsense_camera):
        super().__init__(name="camera-display-loop", daemon=True)
        self.stop_event = stop_event
        self.fisheye_camera = fisheye_camera
        self.realsense_camera = realsense_camera
        self.error = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                fisheye_ok, fisheye_frame = self.fisheye_camera.get_frame()
                if fisheye_ok and fisheye_frame is not None:
                    cv2.imshow(WINDOW_FISHEYE, fisheye_frame)

                color_ok, color_frame = self.realsense_camera.get_color_frame()
                if color_ok and color_frame is not None:
                    cv2.imshow(WINDOW_COLOR, color_frame)

                depth_ok, depth_frame = self.realsense_camera.get_depth_frame()
                if depth_ok and depth_frame is not None:
                    depth_view = cv2.applyColorMap(
                        cv2.convertScaleAbs(np.asanyarray(depth_frame), alpha=0.03),
                        cv2.COLORMAP_JET,
                    )
                    cv2.imshow(WINDOW_DEPTH, depth_view)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.stop_event.set()
                    break

                time.sleep(0.01)
        except Exception as exc:
            self.error = exc
            self.stop_event.set()


def print_current_tcp_pose(state_reader, label):
    pose = state_reader.get_latest_tcp_pose()
    print(f"{label}: {format_tcp_pose(pose)}")
    return pose


def move_linear_base(state_reader, send_urscript_fn, host, port, dx=0.0, dy=0.0, dz=0.0):
    current_pose = state_reader.get_latest_tcp_pose()
    target_pose = list(current_pose)
    target_pose[0] += dx
    target_pose[1] += dy
    target_pose[2] += dz
    motion_line = build_movel(target_pose, a=MOVEL_A, v=MOVEL_V)
    motion_program = build_program([motion_line], program_name="fake_policy_movel")
    send_urscript_fn(motion_program, host=host, port=port)
    state_reader.wait_for_tcp_target(target_pose, tolerance_m=0.003, stable_samples=3, timeout_s=15)
    return target_pose


def _ensure_reader_ready(state_reader):
    if state_reader.wait_for_first_state(timeout_s=3.0):
        return
    if state_reader.last_error is not None:
        raise RuntimeError(
            f"Failed to receive initial UR7e state from {state_reader.host}:{state_reader.port}: {state_reader.last_error}"
        )
    raise RuntimeError(f"Timed out waiting for initial UR7e state from {state_reader.host}:{state_reader.port}.")


def _check_display_error(display_loop):
    if display_loop is not None and getattr(display_loop, "error", None) is not None:
        raise display_loop.error


def _stop_requested(stop_event):
    return stop_event is not None and stop_event.is_set()


def _cleanup_runtime(stop_event, display_loop, pika_rig, state_reader):
    stop_event.set()
    if display_loop is not None:
        display_loop.join(timeout=1.0)
    cv2.destroyAllWindows()
    if pika_rig is not None:
        pika_rig.cleanup()
    state_reader.stop()


def main(
    state_reader_cls=URStateReader,
    pika_rig_cls=PikaRig,
    display_loop_cls=CameraDisplayLoop,
    send_urscript_fn=None,
    stop_event=None,
):
    stop_event = stop_event or threading.Event()
    if send_urscript_fn is None:
        send_urscript_fn = send_urscript
    state_reader = state_reader_cls(UR7E_HOST, UR7E_STATE_PORT, poll_interval_s=0.1)
    pika_rig = None
    display_loop = None

    try:
        state_reader.start()
        _ensure_reader_ready(state_reader)

        pika_rig = pika_rig_cls()
        pika_rig.connect()
        pika_rig.enable_gripper()

        movej_line = build_movej(HOME_JOINTS_RAD, a=MOVEJ_A, v=MOVEJ_V)
        movej_program = build_program([movej_line], program_name="fake_policy_movej")
        send_urscript_fn(movej_program, host=UR7E_HOST, port=UR7E_COMMAND_PORT)
        state_reader.wait_for_joint_target(HOME_JOINTS_RAD, tolerance_rad=0.03, stable_samples=3, timeout_s=25)

        pika_rig.set_gripper_distance_and_wait(0.0)

        fisheye_camera, realsense_camera = pika_rig.init_cameras()
        display_loop = display_loop_cls(stop_event, fisheye_camera, realsense_camera)
        display_loop.start()

        print_current_tcp_pose(state_reader, "初始化 TCP")

        steps = [
            ("X", {"dx": LINEAR_STEP_M, "dy": 0.0, "dz": 0.0}, 20.0),
            ("Y", {"dx": 0.0, "dy": LINEAR_STEP_M, "dz": 0.0}, 0.0),
            ("Z", {"dx": 0.0, "dy": 0.0, "dz": LINEAR_STEP_M}, 20.0),
        ]

        for axis_name, delta, gripper_target in steps:
            if _stop_requested(stop_event):
                break
            move_linear_base(state_reader, send_urscript_fn, UR7E_HOST, UR7E_COMMAND_PORT, **delta)
            pika_rig.set_gripper_distance_and_wait(gripper_target)
            print_current_tcp_pose(state_reader, f"{axis_name} 步后 TCP")
            print(f"{axis_name} 步后夹爪距离: {pika_rig.get_gripper_distance():.1f} mm")
            _check_display_error(display_loop)
            if _stop_requested(stop_event):
                break

        _check_display_error(display_loop)
    except KeyboardInterrupt:
        _cleanup_runtime(stop_event, display_loop, pika_rig, state_reader)
        raise
    except Exception:
        _cleanup_runtime(stop_event, display_loop, pika_rig, state_reader)
        raise
    else:
        _cleanup_runtime(stop_event, display_loop, pika_rig, state_reader)


if __name__ == "__main__":
    main()
