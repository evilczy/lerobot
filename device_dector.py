#!/usr/bin/env python3

import argparse
import glob
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEVICE_INDEX = int(os.getenv("PIKA_DEVICE_INDEX", "1"))
DEFAULT_DEVICES_INFO_PATH = REPO_ROOT / "third_party" / "pika_sdk" / "devices_info.conf"
DEFAULT_SETUP_SCRIPT_PATH = REPO_ROOT / "third_party" / "pika_sdk" / "setup.bash"


@dataclass
class ConfiguredDevice:
    index: int
    serial: Optional[str] = None
    tty_path: Optional[str] = None
    video_path: Optional[str] = None


@dataclass
class DetectedDeviceConfig:
    device_index: int
    gripper_port: str
    gripper_candidates: List[str]
    fisheye_device: Optional[Union[int, str]]
    realsense_serial: Optional[str]
    devices_info_path: Optional[str]
    setup_script_path: Optional[str]


def _run_command(args: List[str]) -> str:
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


def _device_sort_key(path: str):
    match = re.search(r"(\d+)$", path)
    if match:
        return (0, int(match.group(1)))
    return (1, path)


def _list_existing_devices(patterns: List[str]) -> List[str]:
    devices = []
    for pattern in patterns:
        devices.extend(glob.glob(pattern))
    return sorted(set(devices), key=_device_sort_key)


def _extract_trailing_number(value) -> Optional[int]:
    match = re.search(r"(\d+)$", str(value))
    if match:
        return int(match.group(1))
    return None


def normalize_video_device(value) -> Union[int, str]:
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        raise ValueError("Empty fisheye device value.")

    if text.isdigit():
        return int(text)

    video_match = re.search(r"(?:^|/|\\)video(\d+)$", text)
    if video_match:
        return int(video_match.group(1))

    return text


def _get_udev_properties(devnode: str) -> Dict[str, str]:
    output = _run_command(["udevadm", "info", "-q", "property", "-n", devnode])
    properties = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key] = value
    return properties


def _extract_usb_topology(path_value: str) -> Tuple[int, ...]:
    if not path_value:
        return ()

    patterns = [
        r"usb-\d+:([0-9.]+):\d+\.\d+",
        r"(?:^|/)\d+-([0-9.]+):\d+\.\d+(?:/|$)",
        r"(?:^|/)\d+-([0-9.]+)(?:/|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path_value)
        if not match:
            continue
        topology = []
        for part in match.group(1).split("."):
            if part.isdigit():
                topology.append(int(part))
        if topology:
            return tuple(topology)
    return ()


def _common_prefix_length(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    length = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        length += 1
    return length


def _score_video_candidate(candidate: Dict[str, object]) -> int:
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
    if ":capture:" in str(candidate["capabilities"]):
        score += 5
    if bool(candidate["accessible"]):
        score += 2
    return score


def _looks_like_realsense(candidate: Dict[str, object]) -> bool:
    text = f"{candidate['product']} {candidate['model']}".lower()
    return "realsense" in text


def _get_realsense_serials_from_pyrealsense() -> List[str]:
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


def _get_realsense_serials_from_cli() -> List[str]:
    output = _run_command(["rs-enumerate-devices", "-s"])
    if not output:
        return []

    serials = []
    for line in output.splitlines():
        match = re.search(r"Intel RealSense.*?(\d{6,})", line)
        if match:
            serials.append(match.group(1))
    return serials


def _resolve_existing_path(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_configured_devices(devices_info_path: Optional[Path] = None) -> Tuple[Dict[int, ConfiguredDevice], Optional[Path]]:
    path = devices_info_path or _resolve_existing_path(
        [DEFAULT_DEVICES_INFO_PATH, REPO_ROOT / "devices_info.conf"]
    )
    if path is None:
        return {}, None

    configured_devices: Dict[int, ConfiguredDevice] = {}
    pattern = re.compile(r"DEVICE_(\d+)_(SERIAL|TTY_PATH|VIDEO_PATH)=(.*)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue

        index = int(match.group(1))
        field = match.group(2)
        raw_value = match.group(3).strip()
        value = None if raw_value in {"", "未检测到"} else raw_value

        device = configured_devices.setdefault(index, ConfiguredDevice(index=index))
        if field == "SERIAL":
            device.serial = value
        elif field == "TTY_PATH":
            device.tty_path = value
        elif field == "VIDEO_PATH":
            device.video_path = value

    return configured_devices, path


def _load_symlink_topologies(setup_script_path: Optional[Path] = None) -> Tuple[Dict[str, Tuple[int, ...]], Optional[Path]]:
    path = setup_script_path or _resolve_existing_path(
        [DEFAULT_SETUP_SCRIPT_PATH, REPO_ROOT / "setup.bash"]
    )
    if path is None:
        return {}, None

    mapping: Dict[str, Tuple[int, ...]] = {}
    pattern = re.compile(r'KERNELS==\\"([^"]+)\\".*SYMLINK\+=\\"([^"]+)\\"')
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        topology_value = match.group(1).strip()
        symlink_name = match.group(2).strip()
        mapping[symlink_name] = _extract_usb_topology(topology_value)
    return mapping, path


def _list_tty_candidates() -> List[Dict[str, object]]:
    candidates = []
    for devnode in _list_existing_devices(["/dev/ttyUSB*", "/dev/ttyACM*"]):
        props = _get_udev_properties(devnode)
        candidates.append(
            {
                "devnode": devnode,
                "basename": Path(devnode).name,
                "accessible": os.access(devnode, os.R_OK | os.W_OK),
                "topology": _extract_usb_topology(props.get("ID_PATH") or props.get("DEVPATH", "")),
                "vendor": props.get("ID_VENDOR", ""),
                "model": props.get("ID_MODEL", ""),
            }
        )
    return candidates


def _list_video_candidates() -> List[Dict[str, object]]:
    candidates = []
    for video_dev in _list_existing_devices(["/dev/video*"]):
        props = _get_udev_properties(video_dev)
        if props.get("SUBSYSTEM") != "video4linux":
            continue
        candidates.append(
            {
                "devnode": video_dev,
                "basename": Path(video_dev).name,
                "index": _extract_trailing_number(video_dev),
                "accessible": os.access(video_dev, os.R_OK),
                "product": props.get("ID_V4L_PRODUCT", ""),
                "model": props.get("ID_MODEL", ""),
                "capabilities": props.get("ID_V4L_CAPABILITIES", ""),
                "topology": _extract_usb_topology(props.get("ID_PATH") or props.get("DEVPATH", "")),
            }
        )
    return candidates


def _legacy_gripper_preferences() -> List[str]:
    return [f"/dev/ttyUSB{index}" for index in range(81, 90)] + ["/dev/ttyUSB80", "/dev/ttyUSB0"]


def _ordered_unique(values: List[str]) -> List[str]:
    ordered = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def resolve_gripper_port_candidates(device_index: int = DEFAULT_DEVICE_INDEX) -> List[str]:
    env_port = os.getenv("PIKA_GRIPPER_PORT")
    if env_port:
        return [env_port]

    configured_devices, _ = _load_configured_devices()
    symlink_topologies, _ = _load_symlink_topologies()
    configured_device = configured_devices.get(device_index)

    preferred_path = configured_device.tty_path if configured_device else None
    preferred_topology = ()
    if preferred_path:
        preferred_topology = symlink_topologies.get(Path(preferred_path).name, ())

    scored_candidates = []
    for candidate in _list_tty_candidates():
        score = 0
        if candidate["accessible"]:
            score += 100000
        if preferred_path and candidate["devnode"] == preferred_path:
            score += 50000
        if preferred_topology:
            prefix = _common_prefix_length(preferred_topology, candidate["topology"])
            if prefix:
                score += prefix * 10000
                if prefix == len(preferred_topology) and prefix == len(candidate["topology"]):
                    score += 5000
        basename = str(candidate["basename"])
        if basename.startswith("ttyUSB"):
            score += 100
        if basename in {"ttyUSB81", "ttyUSB82", "ttyUSB83", "ttyUSB84", "ttyUSB85", "ttyUSB86", "ttyUSB87", "ttyUSB88", "ttyUSB89"}:
            score += 1000
        if basename == "ttyUSB80":
            score += 900
        if basename == "ttyUSB0":
            score += 100
        scored_candidates.append((score, str(candidate["devnode"])))

    scored_candidates.sort(key=lambda item: (-item[0], _device_sort_key(item[1])))
    ordered = [path for _, path in scored_candidates]

    extra_candidates = []
    if preferred_path:
        extra_candidates.append(preferred_path)
    extra_candidates.extend([path for path in _legacy_gripper_preferences() if os.path.exists(path)])
    ordered.extend(extra_candidates)

    return _ordered_unique(ordered)


def resolve_gripper_port(device_index: int = DEFAULT_DEVICE_INDEX) -> str:
    candidates = resolve_gripper_port_candidates(device_index=device_index)
    if candidates:
        return candidates[0]
    raise RuntimeError(
        "未找到可用的 Pika gripper 串口，请检查 devices_info.conf、setup.bash 或显式设置 PIKA_GRIPPER_PORT。"
    )


def resolve_realsense_serial(device_index: int = DEFAULT_DEVICE_INDEX) -> str:
    env_serial = os.getenv("PIKA_REALSENSE_SERIAL")
    if env_serial:
        return env_serial.strip()

    configured_devices, _ = _load_configured_devices()
    configured_serial = None
    configured_device = configured_devices.get(device_index)
    if configured_device is not None:
        configured_serial = configured_device.serial

    live_serials = []
    for serial in _get_realsense_serials_from_pyrealsense() + _get_realsense_serials_from_cli():
        if serial and serial not in live_serials:
            live_serials.append(serial)

    if configured_serial:
        return configured_serial
    if live_serials:
        return live_serials[0]

    raise RuntimeError(
        "无法自动检测 RealSense 序列号，请先连接 D405，或通过 PIKA_REALSENSE_SERIAL 显式指定。"
    )


def resolve_fisheye_device(gripper_port: Optional[str] = None, device_index: int = DEFAULT_DEVICE_INDEX):
    env_device = os.getenv("PIKA_FISHEYE_DEVICE")
    if env_device:
        return normalize_video_device(env_device)

    configured_devices, _ = _load_configured_devices()
    symlink_topologies, _ = _load_symlink_topologies()
    configured_device = configured_devices.get(device_index)
    preferred_video_path = configured_device.video_path if configured_device else None

    if preferred_video_path and os.path.exists(preferred_video_path):
        return normalize_video_device(preferred_video_path)

    preferred_video_topology = ()
    if preferred_video_path:
        preferred_video_topology = symlink_topologies.get(Path(preferred_video_path).name, ())

    video_candidates = _list_video_candidates()
    if preferred_video_topology:
        best_candidate = None
        best_score = None
        for candidate in video_candidates:
            if ":capture:" not in str(candidate["capabilities"]):
                continue
            prefix = _common_prefix_length(preferred_video_topology, candidate["topology"])
            if prefix <= 0:
                continue
            total_score = prefix * 100 + _score_video_candidate(candidate)
            if best_candidate is None or total_score > best_score:
                best_candidate = candidate
                best_score = total_score
        if best_candidate is not None:
            return best_candidate["index"]

    gripper_port = gripper_port or resolve_gripper_port(device_index=device_index)
    port_index = _extract_trailing_number(gripper_port)
    if port_index is not None and port_index >= 80:
        guessed_video = f"/dev/video{port_index}"
        if os.path.exists(guessed_video):
            return port_index

    tty_props = _get_udev_properties(gripper_port)
    tty_topology = _extract_usb_topology(tty_props.get("ID_PATH") or tty_props.get("DEVPATH", ""))

    capture_candidates = [
        candidate
        for candidate in video_candidates
        if ":capture:" in str(candidate["capabilities"]) and not _looks_like_realsense(candidate)
    ]

    best_candidate = None
    best_score = None
    for candidate in capture_candidates:
        topology_score = _common_prefix_length(tty_topology, candidate["topology"])
        total_score = topology_score * 100 + _score_video_candidate(candidate)
        if best_candidate is None or total_score > best_score:
            best_candidate = candidate
            best_score = total_score

    if best_candidate is not None:
        return best_candidate["index"]

    if port_index is not None:
        guessed_video = f"/dev/video{port_index}"
        if os.path.exists(guessed_video):
            return port_index

    generic_capture_candidates = [
        candidate for candidate in video_candidates if ":capture:" in str(candidate["capabilities"])
    ]
    if generic_capture_candidates:
        generic_capture_candidates.sort(key=_score_video_candidate, reverse=True)
        return generic_capture_candidates[0]["index"]

    if preferred_video_path:
        return normalize_video_device(preferred_video_path)

    raise RuntimeError(
        "无法自动检测鱼眼相机设备，请通过 PIKA_FISHEYE_DEVICE 指定 /dev/videoX 或数字编号。"
    )


def detect_device_config(device_index: int = DEFAULT_DEVICE_INDEX) -> DetectedDeviceConfig:
    configured_devices, devices_info_path = _load_configured_devices()
    _, setup_script_path = _load_symlink_topologies()
    configured_device = configured_devices.get(device_index)

    gripper_candidates = resolve_gripper_port_candidates(device_index=device_index)
    if not gripper_candidates:
        raise RuntimeError(
            "未找到任何 gripper 串口候选项，请先运行 multi_device_detector.py 生成配置或检查当前 /dev/ttyUSB*。"
        )

    gripper_port = gripper_candidates[0]
    fisheye_device = resolve_fisheye_device(gripper_port=gripper_port, device_index=device_index)

    if configured_device is not None and configured_device.serial:
        realsense_serial = configured_device.serial
    else:
        realsense_serial = resolve_realsense_serial(device_index=device_index)

    return DetectedDeviceConfig(
        device_index=device_index,
        gripper_port=gripper_port,
        gripper_candidates=gripper_candidates,
        fisheye_device=fisheye_device,
        realsense_serial=realsense_serial,
        devices_info_path=str(devices_info_path) if devices_info_path else None,
        setup_script_path=str(setup_script_path) if setup_script_path else None,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自动探测 Pika gripper、鱼眼和 RealSense 设备映射。")
    parser.add_argument(
        "--device-index",
        type=int,
        default=DEFAULT_DEVICE_INDEX,
        help="读取 devices_info.conf 中的设备编号，默认取环境变量 PIKA_DEVICE_INDEX 或 1。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 形式输出探测结果。",
    )
    return parser


def main():
    parser = _build_cli_parser()
    args = parser.parse_args()

    config = detect_device_config(device_index=args.device_index)
    if args.json:
        print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
        return

    print(f"设备编号: {config.device_index}")
    print(f"串口候选: {', '.join(config.gripper_candidates)}")
    print(f"选中串口: {config.gripper_port}")
    print(f"鱼眼设备: {config.fisheye_device}")
    print(f"RealSense 序列号: {config.realsense_serial}")
    if config.devices_info_path:
        print(f"devices_info.conf: {config.devices_info_path}")
    if config.setup_script_path:
        print(f"setup.bash: {config.setup_script_path}")
    print()
    print(f"export PIKA_GRIPPER_PORT={config.gripper_port}")
    if config.fisheye_device is not None:
        print(f"export PIKA_FISHEYE_DEVICE={config.fisheye_device}")
    if config.realsense_serial:
        print(f"export PIKA_REALSENSE_SERIAL={config.realsense_serial}")


if __name__ == "__main__":
    main()
