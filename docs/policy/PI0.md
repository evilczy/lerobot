# PI0 在 LeRobot 中的复现与部署指南

这份文档不是泛泛介绍 `pi0` 论文，而是面向“我要在这个仓库里把 `pi0` 跑起来，尤其是跑在 `ur_pika` 上”的复现说明。重点会围绕下面这条主线展开：

`机器人原始观测 -> LeRobot dataset/schema -> PI0 preprocessor -> PI0Policy.select_action() -> postprocessor -> 具名机器人动作 -> UR + Pika 执行`

本文特别关注你当前这套配置：

- 机器人：`ur_pika`
- 控制模式：`joint`
- 相机：单路腕部相机
- 目标：先用预训练 `pi0` 跑通真实硬件，再决定是否继续微调

---

## 1. 先看结论

如果你的目标是“在当前仓库里复现 `pi0` 到 `ur+pika` 上”，最低风险路径是：

1. 用 `joint` 模式，不要先上 `tcp`。
2. 环路先按 `10Hz` 跑，不要一开始追求高频。
3. 相机 key 直接命名成 `right_wrist_0_rgb`，不要继续沿用 `wrist`。
4. 先做硬件 smoke test，再做零样本推理，再决定是否采数据微调。
5. 对真实机器人，先把动作裁剪调小，例如：
   - `joint_max_relative_target_rad=0.05`
   - `gripper_max_relative_target_mm=2.0`

当前仓库已经把 `ur_pika` 的默认单相机改成了 `pi0` 友好的 `right_wrist_0_rgb`，所以只要你的本地分支是现在这份代码，很多最麻烦的命名对齐问题已经提前处理掉了。

---

## 2. 本文对应的关键代码位置

如果你要自己顺藤摸瓜看实现，优先看这些文件：

- `docs/source/pi0.mdx`
- `docs/source/policy_pi0_README.md`
- `docs/source/rename_map.mdx`
- `docs/source/ur_pika.mdx`
- `examples/tutorial/pi0/using_pi0_example.py`
- `src/lerobot/policies/pi0/configuration_pi0.py`
- `src/lerobot/policies/pi0/processor_pi0.py`
- `src/lerobot/policies/pi0/modeling_pi0.py`
- `src/lerobot/policies/factory.py`
- `src/lerobot/policies/utils.py`
- `src/lerobot/datasets/feature_utils.py`
- `src/lerobot/utils/control_utils.py`
- `src/lerobot/scripts/lerobot_record.py`
- `src/lerobot/async_inference/helpers.py`
- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/policy_server.py`
- `src/lerobot/robots/ur_pika/config_ur_pika.py`
- `src/lerobot/robots/ur_pika/robot_ur_pika.py`
- `tests/robots/test_ur_pika_my.py`

可以把这些文件分成四组理解：

1. `pi0` 本体：
   - `configuration_pi0.py`
   - `processor_pi0.py`
   - `modeling_pi0.py`
2. 控制与推理主链路：
   - `control_utils.py`
   - `policies/utils.py`
   - `lerobot_record.py`
3. 数据 schema 变换：
   - `feature_utils.py`
   - `factory.py`
4. 真实硬件与部署：
   - `ur_pika/*`
   - `async_inference/*`
   - `tests/robots/test_ur_pika_my.py`

---

## 3. `pi0` 在这个仓库里到底是什么

### 3.1 它是一个 VLA policy

在这个仓库里，`pi0` 被实现为一个 Vision-Language-Action policy，也就是：

- 输入：
  - 图像
  - 机器人状态
  - 文本任务描述 `task`
- 输出：
  - 一段未来动作序列 `action chunk`

`pi0` 不是“只吃图像”的模型，也不是“只出一步动作”的模型。

### 3.2 它在 LeRobot 中的核心配置

`src/lerobot/policies/pi0/configuration_pi0.py` 里最关键的默认值是：

- `n_obs_steps = 1`
- `chunk_size = 50`
- `n_action_steps = 50`
- `max_state_dim = 32`
- `max_action_dim = 32`
- `num_inference_steps = 10`
- `image_resolution = (224, 224)`
- `tokenizer_max_length = 48`
- `use_relative_actions = False`
- `relative_exclude_joints = ["gripper"]`

这几个值直接决定了复现时你应该怎么理解它：

1. `pi0` 一次看一个当前观测，不是堆多帧历史。
2. 它一次会预测 50 步动作。
3. 默认会把这 50 步都放入内部动作队列逐步消费。
4. 真实机器人状态和动作维度如果不足 32，会被 pad 到 32 维。
5. 输入图像最终会被 resize/pad 到 `224x224`。

### 3.3 `pi0` 默认是绝对动作，不是相对动作

`use_relative_actions=False` 是默认值。

这意味着：

- 默认训练和推理都在“绝对动作空间”里完成。
- 如果你想训练相对动作版本，需要重新计算相对动作统计量，并打开：

```bash
--policy.use_relative_actions=true
```

对于你当前“先把预训练模型跑起来”的目标，建议先不要改成 relative actions。先确保硬件链路、命名、控制模式和数据 schema 完全对齐，再考虑这个优化项。

---

## 4. 从 `ur_pika` 观测到 `pi0` 动作的完整数据流

这部分是整份文档最重要的内容。

### 4.1 第 1 层：机器人原始观测

`URPika.get_observation()` 返回的是“硬件原生 dict”，典型内容类似：

```python
{
    "joint_1.pos": ...,
    "joint_2.pos": ...,
    "joint_3.pos": ...,
    "joint_4.pos": ...,
    "joint_5.pos": ...,
    "joint_6.pos": ...,
    "gripper.pos": ...,
    "right_wrist_0_rgb": np.ndarray(H, W, 3),
}
```

这里要注意：

- 关节是散开的具名标量。
- 图像是 `numpy.ndarray(H, W, 3)`。
- 此时还不是 `pi0` 直接吃的格式。

### 4.2 第 2 层：LeRobot dataset/schema 表示

`src/lerobot/datasets/feature_utils.py` 里的 `hw_to_dataset_features()` 会把硬件 feature 转成 LeRobot 统一 schema：

- `observation.state`
- `observation.images.right_wrist_0_rgb`
- `action`

规则是：

1. 所有低维状态拼成一个向量，放到 `observation.state`。
2. 所有具名动作拼成一个向量，放到 `action`。
3. 所有图像统一挂到 `observation.images.*`。

这一步非常关键，因为后面 policy 根本不认识 `joint_1.pos` 这种散开的硬件 key，它只认规范化后的 schema。

### 4.3 第 3 层：进入推理前的 observation 组装

同步推理有两条常见路径。

#### 路径 A：直接 Python 调用

对应 `examples/tutorial/pi0/using_pi0_example.py`。

它的基本套路是：

1. `robot.get_observation()`
2. `hw_to_dataset_features(...)`
3. `build_inference_frame(...)`
4. `preprocess(...)`
5. `policy.select_action(...)`
6. `postprocess(...)`
7. `make_robot_action(...)`
8. `robot.send_action(...)`

#### 路径 B：`lerobot-record` 在线部署

对应 `src/lerobot/scripts/lerobot_record.py`。

这里实际执行链是：

1. `robot.get_observation()`
2. `predict_action()`
3. `prepare_observation_for_inference()`
4. `preprocessor(observation)`
5. `policy.select_action(observation)`
6. `postprocessor(action)`
7. `make_robot_action(action, dataset.features)`
8. `robot.send_action(...)`

### 4.4 第 4 层：preprocessor 做了什么

`src/lerobot/policies/pi0/processor_pi0.py` 里的预处理顺序是：

1. `RenameObservationsProcessorStep`
2. `AddBatchDimensionProcessorStep`
3. `Pi0NewLineProcessor`
4. `TokenizerProcessorStep`
5. `DeviceProcessorStep`
6. `RelativeActionsProcessorStep`
7. `NormalizerProcessorStep`

其中有几个细节很容易忽略：

#### 任务文本会自动补换行

`Pi0NewLineProcessor` 会确保 `task` 结尾带 `\n`。这是为了兼容 PaliGemma tokenizer。

也就是说，你传：

```python
task = "pick up the object"
```

进入 tokenizer 前会变成：

```python
"pick up the object\n"
```

#### tokenizer 是固定的

`pi0` 这里用的是：

```python
google/paligemma-3b-pt-224
```

并且 `max_length=48`。

#### 图像不会按原始分辨率直接喂给模型

在 `modeling_pi0.py` 的 `_preprocess_images()` 里，图像会经历：

1. 保证 float32
2. 通道检查
3. resize with pad 到 `224x224`
4. 从 `[0, 1]` 映射到 `[-1, 1]`

所以你的硬件相机可以是 `640x480`，但进入 `pi0` 的视觉编码器之前，都会被转换到模型分辨率。

#### 状态和动作都会 pad 到 32 维

`prepare_state()` 和 `prepare_action()` 会把低维向量 pad 到：

- `max_state_dim=32`
- `max_action_dim=32`

这意味着你当前 `ur_pika_joint` 的 7 维状态和 7 维动作是可以直接接入的。

### 4.5 第 5 层：`select_action()` 怎么工作

`src/lerobot/policies/pi0/modeling_pi0.py` 里，`PI0Policy.select_action()` 的逻辑不是“来一帧就只算一步动作”，而是：

1. 如果内部 action queue 为空：
   - 调 `predict_action_chunk()`
   - 一次生成 `chunk_size=50` 步动作
   - 取前 `n_action_steps=50` 步塞进队列
2. 每次 `select_action()` 只弹出队列里的一个动作

这对真实硬件的影响是：

- 每次真正触发完整模型采样的频率，不一定等于控制循环频率。
- 你需要非常小心控制环路频率和动作裁剪，不要一上来就大步执行长 chunk。

### 4.6 第 6 层：postprocessor 和机器人动作恢复

`pi0` 的后处理顺序是：

1. `UnnormalizerProcessorStep`
2. `AbsoluteActionsProcessorStep`
3. `DeviceProcessorStep(device="cpu")`

然后 `src/lerobot/policies/utils.py` 里的 `make_robot_action()` 会把动作向量重新映射成具名机器人命令，比如：

```python
{
    "joint_1.pos": ...,
    "joint_2.pos": ...,
    ...
    "gripper.pos": ...,
}
```

最后才交给 `URPika.send_action()` 真正执行。

---

## 5. 单相机为什么也能跑 `pi0`

这是你当前复现里最关键的问题。

### 5.1 `pi0` 并不要求所有相机都同时存在

`src/lerobot/policies/pi0/modeling_pi0.py` 的 `_preprocess_images()` 里有这样一套逻辑：

1. 先找出 policy config 里“期望的图像 key”。
2. 再找出当前 batch 里“实际存在的图像 key”。
3. 对存在的图像正常预处理。
4. 对缺失的图像，自动补：
   - 全 `-1` 的 dummy image
   - 全 `0` 的 image mask

也就是说：

- 只要至少有一路图像存在，`pi0` 就可以前向。
- 缺失的 camera slot 会被 mask 掉。

### 5.2 但“至少有一路存在”不等于“什么名字都行”

这里的关键不是“有没有图”，而是“图的 key 是否和 policy 期望的 key 对得上”。

仓库里和 `pi0` 相关的 tutorial 与测试都在使用下面这组命名：

- `base_0_rgb`
- `left_wrist_0_rgb`
- `right_wrist_0_rgb`

例如：

- `examples/tutorial/pi0/using_pi0_example.py`
- `tests/policies/pi0_pi05/test_pi0_original_vs_lerobot.py`

所以对于单腕部相机，最稳妥的做法就是：

- 直接把你的相机命名成 `right_wrist_0_rgb`

而不是：

- `wrist`
- `camera0`
- `front`

### 5.3 为什么不能只指望 `rename_map`

`rename_map` 本身是有用的，但你不能把它理解成“所有路径都能自动救场”。

`docs/source/rename_map.mdx` 的描述是正确的：它用于 observation key 重命名，并且对 `pi0` 是支持的。

但对你现在最关心的 `lerobot-record` 在线部署路径来说，有一个实现细节必须知道：

1. `lerobot-record` 会先根据机器人当前 feature 创建 dataset。
2. 然后调用：

```python
make_policy(cfg.policy, ds_meta=dataset.meta)
```

3. 之后才会去创建 preprocessor，并把：

```python
rename_observations_processor.rename_map = cfg.dataset.rename_map
```

塞进去。

这意味着：

- 如果机器人观测 key 是 `observation.images.wrist`
- 但 policy 期望的是 `observation.images.right_wrist_0_rgb`

那么很可能在 policy 还没真正开始用 rename_map 之前，feature 一致性检查就已经不满足了。

所以对真实机器人在线部署，我的建议非常明确：

### 最好直接让机器人输出 policy 期望的相机 key

也就是：

- 单腕部相机：`right_wrist_0_rgb`
- 如果以后加外部相机：`base_0_rgb`
- 如果以后有另一只手腕视角：`left_wrist_0_rgb`

### 5.4 `empty_cameras` 在这里不是首选方案

`PI0Config.empty_cameras` 是支持的，但它更适合“我想显式往 config 里再补几个空相机 feature”的场景。

你现在这类情况并不需要优先靠它解决，因为：

1. `pi0` 本身已经会对缺失的期望相机做 masked dummy 填充。
2. 你当前真正的瓶颈是“相机 key 对不对”，不是“能不能补空相机”。

所以：

- 先解决命名
- 再考虑 `empty_cameras`

---

## 6. `ur_pika` 这条控制链路里，哪些约束和 `pi0` 最相关

### 6.1 先用 `joint` 模式

`docs/source/ur_pika.mdx` 已经写得很明确：

- `joint`
- `tcp`

都支持，但 VLA/ACT 部署优先使用 `joint`。

对于 `pi0`，建议你只使用：

```bash
--robot.control_mode=joint
```

原因很简单：

1. 当前 `ur_pika` 文档和测试链路都主要围绕 joint schema。
2. `joint` 的动作语义更直接，也更容易和数据采集保持一致。
3. `tcp` 会引入另一套动作定义，容易把训练和部署 schema 搅混。

### 6.2 当前 `joint` schema 是 7 维

`ur_pika_joint` 的观测和动作字段是：

- `joint_1.pos`
- `joint_2.pos`
- `joint_3.pos`
- `joint_4.pos`
- `joint_5.pos`
- `joint_6.pos`
- `gripper.pos`

这正好对应：

- 6 个 UR 关节
- 1 个夹爪开合量

这套 schema 对 `pi0` 来说是完全可接入的，因为它最终只看到一个 7 维状态和 7 维动作向量。

### 6.3 当前 `ur_pika` 默认相机已经对齐到 `pi0`

`src/lerobot/robots/ur_pika/config_ur_pika.py` 当前默认是：

```python
"right_wrist_0_rgb": PikaCameraConfig(
    source=PikaCameraSource.REALSENSE_COLOR,
    width=640,
    height=480,
    fps=30,
)
```

这意味着：

- 默认就会输出 `right_wrist_0_rgb`
- 默认就是 RealSense color
- 默认分辨率 `640x480`

对于单相机 `pi0` 复现，这是一个很合适的起点。

### 6.4 环路先按 `10Hz` 跑

`docs/source/ur_pika.mdx` 对这一版集成的建议是：

- v1 integration is designed for stable deployment around `10Hz`

这句话很重要。

虽然相机自身可以 `30fps`，但这不等于整条：

`相机 -> Python -> preprocessor -> pi0 -> postprocessor -> UR socket -> 机械臂`

都适合直接跑到 30Hz。

所以建议是：

- 相机可以保持 `30fps`
- 机器人控制主环路先设 `10Hz`

换句话说：

- `camera fps` 和 `dataset/robot loop fps` 不是一回事

### 6.5 真机上一定先收紧动作裁剪

`ur_pika` 默认的关节相对目标裁剪是比较宽的。你在 `pi0` 零样本尝试阶段，不要直接用大步长。

建议起步参数：

```bash
--robot.joint_max_relative_target_rad=0.05
--robot.gripper_max_relative_target_mm=2.0
```

这样即使 policy 输出很激进，也能先把风险压下来。

---

## 7. 当前仓库里可以直接用的硬件 smoke test

### 7.1 脚本位置

已经有一份专门给 `ur+pika` 写的 smoke test：

```text
tests/robots/test_ur_pika_my.py
```

它会做这些检查：

1. 探测 UR 命令口 `30001`
2. 探测 UR 状态口 `30012`
3. `robot.connect()`
4. `robot.get_observation()`
5. 检查关节、夹爪和 RGB 图像
6. 通过共享 Pika 句柄额外读取 RealSense depth
7. 可选发送 no-op action，验证命令链路

### 7.2 默认参数

这个脚本默认假设：

- `robot_ip = 192.168.1.15`
- `camera_key = right_wrist_0_rgb`
- `control_mode = joint`

同时也支持通过环境变量或命令行覆盖：

- `PIKA_GRIPPER_PORT`
- `PIKA_REALSENSE_SERIAL`
- `UR7E_HOST`

### 7.3 建议先跑的命令

先只看观测链路：

```bash
uv run python tests/robots/test_ur_pika_my.py
```

再验证命令链路：

```bash
uv run python tests/robots/test_ur_pika_my.py --send-noop-action
```

### 7.4 当前仓库上的一次通过结果

这份脚本已经在当前环境里跑通过一次，检查点包括：

- UR command socket 可达
- UR state socket 可达
- `robot.connect()` 成功
- `robot.get_observation()` 成功
- `right_wrist_0_rgb` 返回 `(480, 640, 3)` 的 `uint8` 图像
- RealSense depth 可读
- no-op `robot.send_action()` 成功

这意味着你现在的基础硬件链路不是“未知状态”，而是已经有一条可复用的检查路径。

---

## 8. 复现 `pi0` 的推荐顺序

我建议你不要直接跳到“真实任务复现”，而是按下面四步推进。

### 8.1 第一步：安装依赖

如果你只跑 `pi0`：

```bash
uv pip install -e ".[pi]"
```

如果你后面还想跑异步部署：

```bash
uv pip install -e ".[pi,async]"
```

对应的 optional dependencies 定义在 `pyproject.toml` 里：

- `pi = ["lerobot[transformers-dep]", "lerobot[scipy-dep]"]`
- `async = ["lerobot[grpcio-dep]", "lerobot[matplotlib-dep]"]`

### 8.2 第二步：先通过 smoke test

这是必须先过的一步。

如果 smoke test 没过，就不要继续看 `pi0` 输出，因为那时你根本分不清是模型问题、命名问题还是硬件问题。

### 8.3 第三步：先做最小零样本推理

最小零样本推理有两条路。

#### 路线 A：直接 Python 最小闭环

这是最清晰、最适合 debug 的方式。

下面是一份适合你当前 `ur_pika + 单右腕相机` 的最小示例：

```python
from pathlib import Path

import torch

from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.ur_pika import URPika, URPikaConfig


device = torch.device("cuda")
model_id = "lerobot/pi0_base"
task = "pick up the object"

policy = PI0Policy.from_pretrained(model_id)
policy.to(device)

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

robot = URPika(
    URPikaConfig(
        robot_ip="192.168.1.15",
        gripper_port="/dev/ttyUSB0",
        control_mode="joint",
        calibration_dir=Path("./calibration/ur_pika"),
        joint_max_relative_target_rad=0.05,
        gripper_max_relative_target_mm=2.0,
    )
)

robot.connect()
try:
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=False)
    dataset_features = {**action_features, **obs_features}

    obs = robot.get_observation()
    obs_frame = build_inference_frame(
        observation=obs,
        device=device,
        ds_features=dataset_features,
        task=task,
        robot_type=robot.robot_type,
    )

    model_input = preprocess(obs_frame)
    action = policy.select_action(model_input)
    action = postprocess(action)
    robot_action = make_robot_action(action, dataset_features)

    print(robot_action)
    # 真机调试时建议先打印，不要立刻执行
    # robot.send_action(robot_action)
finally:
    robot.disconnect()
```

这段代码有三个优点：

1. 路径短，出错点少。
2. 你能直接看到 `robot_action` 的具名结果。
3. 你可以先不执行动作，只验证 observation 和 policy 输出。

#### 路线 B：用 `lerobot-record` 直接部署预训练 policy

如果你想快速做在线尝试，可以用：

```bash
uv run lerobot-record \
  --robot.type=ur_pika \
  --robot.robot_ip=192.168.1.15 \
  --robot.gripper_port=/dev/ttyUSB0 \
  --robot.control_mode=joint \
  --robot.joint_max_relative_target_rad=0.05 \
  --robot.gripper_max_relative_target_mm=2.0 \
  --dataset.repo_id=<hf_user>/eval_ur_pika_pi0_smoke \
  --dataset.single_task="pick up the object" \
  --dataset.fps=10 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=20 \
  --dataset.num_episodes=3 \
  --dataset.video=false \
  --dataset.push_to_hub=false \
  --policy.path=lerobot/pi0_base \
  --policy.device=cuda \
  --display_data=true
```

这条命令默认依赖当前 `ur_pika` 的默认相机 key 已经是 `right_wrist_0_rgb`。

如果你需要显式覆盖相机配置，可以再加：

```bash
--robot.cameras='{right_wrist_0_rgb: {type: pika, source: realsense_color, width: 640, height: 480, fps: 30, warmup_s: 2.0}}'
```

### 8.4 第四步：如果零样本效果不行，再进入微调

这点要有合理预期。

从仓库里的 `docs/source/pi0.mdx` 看，官方示例主要是把 `lerobot/pi0_base` 作为一个微调起点使用：

```bash
uv run lerobot-train \
  --dataset.repo_id=<your_dataset> \
  --policy.type=pi0 \
  --output_dir=./outputs/pi0_training \
  --job_name=ur_pika_pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.repo_id=<hf_user>/pi0-ur-pika \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=3000 \
  --policy.device=cuda \
  --batch_size=32
```

如果你的目标是真正复现某个具体抓取/操作任务，而不是只做零样本 smoke test，那么最后大概率还是会走到这一步。

---

## 9. `lerobot-record`、直接 Python、async inference 这三条路有什么区别

### 9.1 直接 Python

最适合：

- 理解数据流
- 看清楚 observation 和 action
- 排查命名/shape/设备问题

不适合：

- 一开始就长时间自动运行
- 一开始就做复杂在线评估

### 9.2 `lerobot-record`

最适合：

- 边部署边录制 eval episode
- 快速验证真实机器人在线跑 policy

要特别注意：

- 相机 key 最好直接对齐 policy
- `dataset.rename_map` 不能替代前面的 feature/schema 对齐

### 9.3 async inference

这一条链路由：

- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/policy_server.py`

构成。

它的思路是：

1. `robot_client` 挂在机器人侧取 observation
2. 通过 gRPC 发送到 `policy_server`
3. `policy_server` 加载 policy 和 pre/post processor
4. 返回一整个 action chunk

这里也有一个和你当前问题高度相关的点：

- `RemotePolicyConfig` 虽然支持 `rename_map`
- 但 `robot_client` 默认传给服务端的是机器人当前 schema 导出的 feature

所以对于真机 async 部署，依然推荐：

- 机器人原始相机 key 直接命名成 policy 期望的 key

如果以后你确实要上 async，可以先用：

```bash
uv run python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=10 \
  --inference_latency=0.1
```

再在另一个终端运行：

```bash
uv run python -m lerobot.async_inference.robot_client \
  --robot.type=ur_pika \
  --robot.robot_ip=192.168.1.15 \
  --robot.gripper_port=/dev/ttyUSB0 \
  --robot.control_mode=joint \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi0 \
  --pretrained_name_or_path=lerobot/pi0_base \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=8 \
  --fps=10
```

不过对你现在这个阶段，我不建议先从 async 开始。先把同步链路跑通更稳。

---

## 10. 复现时最容易踩的坑

### 10.1 相机 key 不对

这是第一大坑。

错误示例：

- `wrist`
- `front`
- `camera0`

当前更推荐的命名：

- `right_wrist_0_rgb`

### 10.2 以为 `rename_map` 能解决所有问题

不是。

`rename_map` 主要作用在 observation 重命名阶段，但很多路径在那之前就已经做了 feature/schema 推断。

### 10.3 所有图像都缺失

`pi0` 允许缺部分图像，但不允许“一个期望图像都没有”。如果 batch 中一个可识别的 image key 都不存在，会直接报错。

### 10.4 `joint` 和 `tcp` 混用

不要：

- 用 `joint` 采数据
- 再用 `tcp` 部署

也不要把不同 action schema 的数据混进同一次训练。

### 10.5 把相机帧率和控制环路频率混为一谈

相机 `30fps` 不等于机器人控制主环必须 `30Hz`。

当前 `ur_pika` 更稳妥的建议是：

- camera: `30fps`
- robot loop / dataset fps: `10Hz`

### 10.6 一开始就允许大动作

零样本阶段，先把动作裁剪收紧。

不然你很容易遇到的不是“复现失败”，而是“刚开始就动作过猛，无法安全调试”。

### 10.7 期待零样本立刻完成真实任务

这是预期管理问题。

在这套代码路径下，`lerobot/pi0_base` 完全可以先拿来做：

- 模型能否加载
- observation 是否能进入 policy
- action 是否能正常出来
- 真机是否能安全执行小步动作

但如果目标是稳定完成你自己的具体任务，通常还是要准备针对你这套硬件和视角的数据，再继续微调。

---

## 11. 推荐的实际操作清单

如果你今天就要开始复现，我建议按这个 checklist 执行：

1. 安装依赖：

```bash
uv pip install -e ".[pi]"
```

2. 跑硬件 smoke test：

```bash
uv run python tests/robots/test_ur_pika_my.py --send-noop-action
```

3. 确认 observation 里出现：

- `joint_1.pos` 到 `joint_6.pos`
- `gripper.pos`
- `right_wrist_0_rgb`

4. 运行最小 Python 推理闭环，但先只打印动作，不执行：

- 加载 `PI0Policy.from_pretrained("lerobot/pi0_base")`
- `build_inference_frame()`
- `preprocess()`
- `select_action()`
- `postprocess()`
- `make_robot_action()`
- `print(robot_action)`

5. 确认动作数值范围合理后，再尝试非常小幅度执行。

6. 如果在线同步部署没问题，再考虑：

- `lerobot-record` 录 eval
- async inference
- 采集数据微调

---

## 12. 对你当前这套 `ur+pika` 的最终建议

把它压缩成一句话就是：

### 先把 `pi0` 当作“真实硬件链路与 policy 接口的验证器”，不要一上来把它当成“任务效果保证器”。

你现在最有价值的事情不是立刻追求最终任务成功率，而是先确认下面四件事已经全部稳定：

1. `ur_pika` 的 `joint` schema 稳定。
2. 默认相机 key 已经是 `right_wrist_0_rgb`。
3. `pi0` 能稳定吃下当前 observation 并输出动作。
4. 真实机器人能在小动作裁剪下安全执行。

只要这四件事成立，你后面无论是：

- 继续做零样本尝试
- 录 `eval_` 数据
- 采集任务数据微调 `lerobot/pi0_base`

都会顺很多。

如果这四件事里有一件没稳，后面越往前走越容易混淆问题来源。

