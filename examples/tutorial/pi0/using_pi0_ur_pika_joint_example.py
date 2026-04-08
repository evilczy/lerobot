#!/usr/bin/env python3  # 使用当前环境中的 python3 解释器执行这个 joint 模式示例脚本。
# HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 uv run python examples/tutorial/pi0/using_pi0_ur_pika_joint_example.py   --device cuda   --dtype bfloat16   --tokenizer-path /home/czy/code/robot/lerobot/ckpt/paligemma_tokenizer
# 跑之前需要sudo chmod 666 /dev/ttyUSB0
from __future__ import annotations  # 延迟解析类型注解，避免前向引用时报错。

import argparse  # 导入命令行参数解析库，用于接收运行参数。
import time  # 导入 time，用于超时控制和循环节拍控制。
from pathlib import Path  # 导入 Path，统一处理本地路径。

import cv2  # 导入 OpenCV，用于前台显示夹爪相机画面。
import numpy as np  # 导入 NumPy，用于处理相机图像数组。
import torch  # 导入 PyTorch，用于设备、推理和张量计算。

from checkpoint_utils import load_pi0_policy_low_mem  # 导入低峰值 PI0 加载器，避免默认加载路径的内存峰值过高。
from checkpoint_utils import resolve_model_checkpoint  # 导入 ckpt 解析函数，负责下载或定位本地模型目录。
from checkpoint_utils import resolve_paligemma_tokenizer_path  # 导入 PaliGemma tokenizer 解析函数，优先走本地目录。
from checkpoint_utils import resolve_policy_dtype  # 导入 policy dtype 解析函数，自动为 CUDA 选择 bfloat16。
from lerobot.datasets.feature_utils import hw_to_dataset_features  # 导入硬件 schema 到 LeRobot dataset schema 的转换函数。
from lerobot.policies.factory import make_pre_post_processors  # 导入 policy 预处理和后处理流水线构造函数。
from lerobot.policies.utils import build_inference_frame  # 导入把原始观测组装成 policy 输入的工具函数。
from lerobot.policies.utils import make_robot_action  # 导入把动作张量还原成具名机器人动作的工具函数。
from lerobot.robots.ur_pika import URPika  # 导入 UR+Pika 机器人类。
from lerobot.robots.ur_pika import URPikaConfig  # 导入 UR+Pika 机器人配置类。
from lerobot.utils.control_utils import is_headless  # 导入是否无头环境的判断函数，避免没有桌面时强行显示窗口。

JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))  # 定义 6 个关节状态/动作键名，保持固定顺序。
GRIPPER_KEY = "gripper.pos"  # 定义夹爪状态/动作键名。

INIT_JOINT_TARGET_RAD = {  # 定义启动前需要到达的固定关节初始化姿态，单位是弧度。
    "joint_1.pos": 0.010472,  # 基座关节目标值。
    "joint_2.pos": -2.733186,  # 肩部关节目标值。
    "joint_3.pos": 1.233948,  # 肘部关节目标值。
    "joint_4.pos": -2.892011,  # 手腕 1 关节目标值。
    "joint_5.pos": 1.371829,  # 手腕 2 关节目标值。
    "joint_6.pos": -0.034907,  # 手腕 3 关节目标值。
}  # 完成初始化关节目标字典定义。

INIT_COMMAND_PERIOD_S = 0.5  # 初始化阶段重复下发目标动作的周期，单位秒。
INIT_TIMEOUT_S = 30.0  # 初始化阶段的最大等待时间，单位秒。
JOINT_TOLERANCE_RAD = 0.05  # 关节到位判定阈值，单位弧度。
GRIPPER_TOLERANCE_MM = 1.0  # 夹爪到位判定阈值，单位毫米。
POLICY_STEP_PERIOD_S = 0.1  # policy 主循环的目标步进周期，约等于 10Hz。
CAMERA_WINDOW_NAME = "UR Pika Wrist Camera (Sent To PI0)"  # 前台显示窗口标题，强调这是送给 PI0 的同一张图。
CAMERA_FRAME_TIMEOUT_MS = 1000  # 从相机等待 fresh frame 的超时时间，单位毫秒。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(  # 创建命令行参数解析器。
        description=(  # 给脚本写一个用途说明。
            "Run UR+Pika in joint mode with a fixed startup pose and fully open gripper before starting "  # 说明脚本会先完成初始化。
            "a pretrained PI0 policy."  # 说明后续会启动预训练 PI0。
        )  # 完成 description 字符串拼接。
    )  # 完成解析器初始化。
    parser.add_argument("--model-id", default="lerobot/pi0_base")  # 指定模型来源，可以是 Hugging Face repo id 或本地目录。
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/home/czy/code/robot/lerobot/ckpt"))  # 指定模型下载和本地读取目录。
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")  # 指定推理设备，默认优先使用 CUDA。
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default=None)  # 指定 policy 权重加载精度，不传则按设备自动推断。
    parser.add_argument("--robot-ip", default="192.168.1.15")  # 指定 UR 控制器 IP 地址。
    parser.add_argument("--gripper-port", default=None)  # 指定 Pika 夹爪串口；不传则允许运行时自动解析。
    parser.add_argument("--tokenizer-path", default=None)  # 指定本地 tokenizer 目录；不传则优先从 ckpt 子目录解析。
    parser.add_argument("--task", default="grasp")  # 指定传给 PI0 的文本任务描述。
    parser.add_argument("--steps", type=int, default=20)  # 指定主循环执行多少步。
    parser.add_argument(  # 开始添加校准目录参数。
        "--calibration-dir",  # 指定参数名。
        type=Path,  # 把命令行值解析成 Path。
        default=Path("./calibration/ur_pika"),  # 指定默认校准目录。
    )  # 完成校准目录参数定义。
    return parser.parse_args()  # 解析并返回命令行参数对象。


def print_named_values(  # 定义按固定顺序打印具名字典的辅助函数。
    title: str,  # 第一个参数是打印标题。
    values: dict[str, float],  # 第二个参数是需要打印的具名数值字典。
    keys: tuple[str, ...] | list[str] | None = None,  # 第三个参数是可选的打印顺序。
) -> None:  # 这个函数没有返回值。
    print(title, flush=True)  # 先打印标题，并立即刷新到终端。
    ordered_keys = keys if keys is not None else values.keys()  # 如果调用方给了顺序就按给定顺序，否则按字典顺序输出。
    for key in ordered_keys:  # 逐个遍历要打印的键名。
        print(f"  {key}: {float(values[key]):.6f}", flush=True)  # 打印每个键对应的浮点值，并格式化到 6 位小数。


def print_state_transition(  # 定义打印“当前状态 / 模型预测 / 机器人实际执行”三组信息的辅助函数。
    step: int,  # 当前 rollout 步数。
    current_state: dict[str, float],  # 当前真实机器人状态。
    predicted_state: dict[str, float],  # 模型预测出来的下一步目标状态。
    applied_state: dict[str, float],  # 机器人侧裁剪后的最终执行状态。
) -> None:  # 这个函数没有返回值。
    print(f"\nStep {step}", flush=True)  # 打印当前步编号。
    print_named_values("Current robot state:", current_state, [*JOINT_KEYS, GRIPPER_KEY])  # 打印当前关节和夹爪状态。
    print_named_values("Predicted next robot state:", predicted_state, [*JOINT_KEYS, GRIPPER_KEY])  # 打印模型预测的目标状态。
    print_named_values("Applied state after clipping:", applied_state, [*JOINT_KEYS, GRIPPER_KEY])  # 打印机器人侧裁剪后的真实执行目标。


def describe_camera_frame(frame: np.ndarray) -> str:  # 定义把相机图像摘要成可读字符串的辅助函数。
    return (  # 返回一条可直接打印的图像摘要字符串。
        f"shape={tuple(frame.shape)}, dtype={frame.dtype}, "  # 包含图像形状和 dtype。
        f"min={int(frame.min())}, max={int(frame.max())}, mean={float(frame.mean()):.2f}"  # 包含图像数值范围和平均亮度。
    )  # 完成图像摘要字符串构造。


def prepare_frame_for_display(frame: np.ndarray) -> np.ndarray:  # 定义把送模图像转换成 OpenCV 可显示格式的辅助函数。
    display_frame = np.asarray(frame)  # 先把输入图像显式视为 NumPy 数组。

    if display_frame.ndim != 3:  # 如果图像不是三维数组。
        raise ValueError(f"Expected a 3D image array, got shape={display_frame.shape}")  # 直接抛错，避免错误显示。

    if display_frame.shape[0] == 3 and display_frame.shape[-1] != 3:  # 如果图像是通道优先格式而不是通道最后格式。
        display_frame = np.transpose(display_frame, (1, 2, 0))  # 把图像从 CHW 转成 HWC。

    if np.issubdtype(display_frame.dtype, np.floating):  # 如果图像是浮点数图像。
        max_value = float(display_frame.max()) if display_frame.size else 0.0  # 先读取当前图像的最大值，用来判断量纲。
        if max_value <= 1.0:  # 如果图像处于 [0, 1] 量纲。
            display_frame = np.clip(display_frame * 255.0, 0.0, 255.0).astype(np.uint8)  # 先放大到 [0, 255] 再转 uint8。
        else:  # 如果图像本身已经接近 [0, 255] 量纲。
            display_frame = np.clip(display_frame, 0.0, 255.0).astype(np.uint8)  # 直接裁剪后转成 uint8。
    else:  # 如果图像本来就是整数类型。
        display_frame = np.clip(display_frame, 0, 255).astype(np.uint8, copy=False)  # 直接裁剪到 uint8 可显示范围。

    display_frame = np.ascontiguousarray(display_frame)  # 保证数组连续，避免 OpenCV 在某些情况下显示异常。
    return cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)  # 把 RGB 图像转成 OpenCV 需要的 BGR 格式并返回。


def show_camera_frame(frame: np.ndarray, display_enabled: bool) -> None:  # 定义前台显示当前送模图像的辅助函数。
    if not display_enabled:  # 如果当前环境是无头环境或者用户不允许显示。
        return  # 直接跳过显示逻辑。

    cv2.imshow(CAMERA_WINDOW_NAME, prepare_frame_for_display(frame))  # 把当前送模图像显示到命名窗口里。
    cv2.waitKey(1)  # 用最小 waitKey 触发窗口刷新。


def attach_fresh_camera_frame(  # 定义“显式抓一张 fresh frame 并回填到 observation” 的辅助函数。
    robot: URPika,  # 第一个参数是当前机器人对象。
    observation: dict[str, object],  # 第二个参数是刚读取到的 observation 字典。
    camera_key: str,  # 第三个参数是相机 key。
) -> np.ndarray:  # 返回真正回填进去的 fresh frame。
    frame = robot.cameras[camera_key].async_read(timeout_ms=CAMERA_FRAME_TIMEOUT_MS)  # 明确等待一张新的相机帧，而不是只用 read_latest 的旧帧。
    observation[camera_key] = frame  # 把 fresh frame 回填到 observation，确保显示和送模是同一张图。
    return frame  # 返回这张 fresh frame 供调用方显示和打印统计信息。


def build_initialization_action(robot: URPika) -> dict[str, float]:  # 定义构造初始化动作字典的辅助函数。
    return {  # 返回一个 joint + gripper 一体化动作字典。
        **INIT_JOINT_TARGET_RAD,  # 展开 6 个关节的固定目标姿态。
        GRIPPER_KEY: float(robot.config.gripper_max_mm),  # 把夹爪目标直接设成最大张开。
    }  # 完成初始化动作字典构造。


def max_joint_error_rad(observation: dict[str, object]) -> float:  # 定义计算当前关节与初始化目标之间最大误差的辅助函数。
    return max(abs(float(observation[key]) - INIT_JOINT_TARGET_RAD[key]) for key in JOINT_KEYS)  # 返回 6 个关节里最大的绝对误差。


def is_initialized(observation: dict[str, object], robot: URPika) -> bool:  # 定义判断机器人是否完成初始化的辅助函数。
    return (  # 同时要求关节和夹爪都到位。
        max_joint_error_rad(observation) <= JOINT_TOLERANCE_RAD  # 先要求 6 个关节都在阈值范围内。
        and float(observation[GRIPPER_KEY]) >= float(robot.config.gripper_max_mm) - GRIPPER_TOLERANCE_MM  # 再要求夹爪已经张到最大附近。
    )  # 返回联合判定结果。


def initialize_robot_pose(  # 定义启动前初始化关节姿态和夹爪开度的函数。
    robot: URPika,  # 第一个参数是当前机器人对象。
    camera_key: str,  # 第二个参数是当前要显示/送模的相机 key。
    display_enabled: bool,  # 第三个参数表示当前是否允许显示窗口。
) -> None:  # 这个函数没有返回值，只在超时时抛异常。
    init_action = build_initialization_action(robot)  # 先构造 joint + gripper 初始化动作字典。
    print_named_values("Initialization joint target:", init_action, JOINT_KEYS)  # 打印 6 个关节初始化目标。
    print(f"Initialization gripper target: {init_action[GRIPPER_KEY]:.2f} mm", flush=True)  # 打印夹爪初始化目标。

    deadline = time.monotonic() + INIT_TIMEOUT_S  # 计算初始化阶段的绝对超时时刻。
    attempt = 0  # 初始化重发次数从 0 开始计数。

    while True:  # 进入重复下发初始化动作的循环。
        if time.monotonic() > deadline:  # 如果已经超过最大等待时间。
            raise TimeoutError(  # 直接抛出超时异常，并说明当前阈值条件。
                "Timed out waiting for the startup pose. "  # 先给出超时原因。
                f"Joint tolerance={JOINT_TOLERANCE_RAD:.3f} rad, "  # 打印关节阈值。
                f"gripper target>={float(robot.config.gripper_max_mm) - GRIPPER_TOLERANCE_MM:.2f} mm"  # 打印夹爪阈值。
            )  # 完成异常构造。

        attempt += 1  # 记录又发送了一次初始化动作。
        robot.send_action(init_action)  # 真正向机器人下发一次初始化动作。
        observation = robot.get_observation()  # 读取当前机器人状态和相机观测。
        camera_frame = attach_fresh_camera_frame(robot, observation, camera_key)  # 显式抓一张 fresh frame 并覆盖到 observation。
        show_camera_frame(camera_frame, display_enabled)  # 把当前送模图像显示在前台窗口。
        joint_error = max_joint_error_rad(observation)  # 计算当前 6 个关节离目标还差多少。
        gripper_mm = float(observation[GRIPPER_KEY])  # 读取当前夹爪开度。

        print(  # 打印这一轮初始化观测到的状态摘要。
            f"[init {attempt}] max_joint_error={joint_error:.6f} rad | "  # 打印最大关节误差。
            f"gripper={gripper_mm:.2f} mm | "  # 打印当前夹爪开度。
            f"camera={describe_camera_frame(camera_frame)}",  # 打印当前送模图像摘要。
            flush=True,  # 立刻把日志刷新到终端。
        )  # 完成初始化状态摘要打印。

        if is_initialized(observation, robot):  # 如果当前已经满足“关节到位 + 夹爪张开”的联合条件。
            print("Initialization complete. Starting policy.", flush=True)  # 打印初始化完成提示。
            return  # 结束初始化流程，回到主函数继续启动 policy。

        time.sleep(INIT_COMMAND_PERIOD_S)  # 如果还没到位，就等待下一个初始化周期再重发。


def main() -> int:  # 定义脚本主函数，并约定返回整数退出码。
    args = parse_args()  # 先读取并解析命令行参数。
    device = torch.device(args.device)  # 把设备字符串转换成 torch.device。
    model_path = resolve_model_checkpoint(args.model_id, args.ckpt_dir)  # 解析模型目录，必要时先下载到本地。
    tokenizer_path = resolve_paligemma_tokenizer_path(args.ckpt_dir, args.tokenizer_path)  # 解析本地 tokenizer 目录，必要时尝试下载到 ckpt 子目录。
    policy_dtype = resolve_policy_dtype(device, args.dtype)  # 根据设备和命令行参数确定 policy 加载精度。
    display_enabled = not is_headless()  # 根据当前环境判断是否可以弹出 OpenCV 窗口。

    print(f"Resolved model path: {model_path}", flush=True)  # 打印最终使用的本地模型目录。
    print(f"Resolved tokenizer path: {tokenizer_path}", flush=True)  # 打印最终使用的本地 tokenizer 目录。
    print(f"Policy load device: {device}", flush=True)  # 打印最终使用的推理设备。
    print(f"Policy load dtype: {policy_dtype}", flush=True)  # 打印最终使用的权重加载精度。

    policy = load_pi0_policy_low_mem(model_path, device, policy_dtype)  # 在连接硬件之前，用低峰值方式把 PI0 加载到目标设备。
    policy.reset()  # 显式重置 policy 的内部 action queue，确保本次 rollout 从干净状态开始。

    preprocess, postprocess = make_pre_post_processors(  # 根据当前 policy 配置构造预处理和后处理流水线。
        policy.config,  # 使用刚刚加载出来的 policy 配置。
        str(model_path),  # 告诉处理器配置文件也从同一个本地模型目录读取。
        preprocessor_overrides={  # 开始覆盖预处理流水线里的局部参数。
            "device_processor": {"device": str(device)},  # 把预处理里的设备搬运目标改成当前推理设备。
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},  # 把 tokenizer 从 gated 远端 repo 改成本地目录。
        },  # 完成预处理覆盖参数构造。
    )  # 完成处理器流水线构造。

    robot = URPika(  # 开始构造 UR+Pika 机器人对象。
        URPikaConfig(  # 构造机器人配置。
            robot_ip=args.robot_ip,  # 写入 UR 控制器 IP。
            gripper_port=args.gripper_port,  # 写入夹爪串口。
            control_mode="joint",  # 明确使用 joint 控制模式。
            calibration_dir=args.calibration_dir,  # 写入校准目录。
        )  # 完成机器人配置对象构造。
    )  # 完成机器人对象实例化。

    camera_key = next(iter(robot.cameras))  # 读取机器人配置中的唯一相机 key，当前默认应为 right_wrist_0_rgb。

    robot.connect()  # 连接 UR 状态口、命令口、Pika 设备和相机。
    try:  # 进入 try/finally，确保后面无论成功失败都能断开机器人。
        if display_enabled:  # 如果当前环境允许弹窗显示。
            print(f"Displaying camera stream from: {camera_key}", flush=True)  # 打印当前前台显示的相机 key。
        else:  # 如果当前环境是无头环境。
            print("Camera display disabled because the environment is headless.", flush=True)  # 打印不显示窗口的原因。

        initialize_robot_pose(robot, camera_key, display_enabled)  # 在启动 policy 之前先让机器人进入固定 joint 姿态并张开夹爪。

        action_features = hw_to_dataset_features(robot.action_features, "action")  # 把机器人动作 schema 转成 LeRobot dataset action schema。
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=False)  # 把机器人观测 schema 转成 LeRobot dataset observation schema。
        dataset_features = {**action_features, **obs_features}  # 合并动作和观测 schema，形成完整 dataset feature 描述。

        print(f"Robot type: {robot.robot_type}", flush=True)  # 打印当前机器人类型，例如 ur_pika_joint。
        print(f"Policy expects image keys: {list(policy.config.image_features)}", flush=True)  # 打印当前 ckpt 期望的视觉输入 key。
        print(f"Robot action names: {dataset_features['action']['names']}", flush=True)  # 打印动作向量和具名动作之间的映射顺序。

        for step in range(args.steps):  # 按命令行指定的步数运行主循环。
            loop_start = time.perf_counter()  # 记录当前循环起始时间，用于维持固定控制节拍。
            observation = robot.get_observation()  # 从真实机器人读取当前关节状态、夹爪状态和默认相机帧。
            camera_frame = attach_fresh_camera_frame(robot, observation, camera_key)  # 再显式抓一张 fresh frame 并回填到 observation。
            show_camera_frame(camera_frame, display_enabled)  # 在前台显示这张真正送给 PI0 的图像。

            obs_frame = build_inference_frame(  # 把原始 observation 组装成 policy 推理需要的输入格式。
                observation=observation,  # 传入当前这一步的原始机器人观测。
                ds_features=dataset_features,  # 传入动作和观测的 schema 描述。
                device=device,  # 传入当前推理设备。
                task=args.task,  # 传入文本任务描述。
                robot_type=robot.robot_type,  # 传入机器人类型元数据。
            )  # 完成推理输入帧构造。

            present_image_keys = [key for key in policy.config.image_features if key in obs_frame]  # 统计当前 batch 里真正存在的图像 key。
            missing_image_keys = [key for key in policy.config.image_features if key not in obs_frame]  # 统计当前 batch 里缺失、将由 PI0 自动补空的图像 key。

            print(f"\nStep {step + 1} camera sent to PI0: {describe_camera_frame(camera_frame)}", flush=True)  # 打印当前送模图像摘要。
            print(f"Policy image inputs present: {present_image_keys}", flush=True)  # 打印当前真正存在的图像 key。
            print(f"Policy image inputs missing and padded by PI0: {missing_image_keys}", flush=True)  # 打印当前会被补空的图像 key。

            model_input = preprocess(obs_frame)  # 对推理输入做 tokenizer、归一化和设备搬运。

            with torch.inference_mode():  # 在纯推理模式下执行 policy，减少额外显存和 autograd 开销。
                action = policy.select_action(model_input)  # 调用 PI0 基于当前观测选择一个动作。

            action = postprocess(action)  # 对 policy 输出做反归一化和设备回迁。
            robot_action = make_robot_action(action, dataset_features)  # 把动作张量还原成具名关节/夹爪动作字典。
            applied_action = robot.send_action(robot_action)  # 真正把动作发送给机器人，并拿到机器人侧裁剪后的实际动作。

            current_state = {key: float(observation[key]) for key in JOINT_KEYS}  # 按固定顺序提取当前 6 个关节状态。
            current_state[GRIPPER_KEY] = float(observation[GRIPPER_KEY])  # 追加当前夹爪状态。
            predicted_state = {key: float(robot_action[key]) for key in [*JOINT_KEYS, GRIPPER_KEY]}  # 按固定顺序提取模型预测的目标状态。
            applied_state = {key: float(applied_action[key]) for key in [*JOINT_KEYS, GRIPPER_KEY]}  # 按固定顺序提取机器人真正执行的目标状态。

            print_state_transition(step + 1, current_state, predicted_state, applied_state)  # 把当前状态、模型预测和实际执行统一打印出来。

            remaining_sleep = POLICY_STEP_PERIOD_S - (time.perf_counter() - loop_start)  # 计算为了维持固定控制节拍还需要睡多久。
            if remaining_sleep > 0:  # 如果当前这一步的推理和控制耗时还没有超过目标周期。
                time.sleep(remaining_sleep)  # 睡一小段时间，把主循环节拍稳定在大约 10Hz。

        print("\nPolicy rollout finished.", flush=True)  # 所有步数执行完之后打印结束提示。
        return 0  # 正常结束时返回 0 作为成功退出码。
    finally:  # 无论前面是否报错，都会进入 finally 做收尾。
        if display_enabled:  # 如果当前环境允许弹窗显示。
            cv2.destroyAllWindows()  # 关闭所有 OpenCV 显示窗口。
        if robot.is_connected:  # 如果机器人当前仍然处于连接状态。
            robot.disconnect()  # 主动断开机器人、夹爪和相机连接。


if __name__ == "__main__":  # 只有直接运行这个文件时才执行 main()。
    raise SystemExit(main())  # 用 main() 的返回值作为脚本退出码退出进程。
