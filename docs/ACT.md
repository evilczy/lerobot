# ACT 在 LeRobot 中如何从相机输入走到机械臂动作执行

这份文档面向后续需要在本仓库里接入新硬件的人，目标不是泛泛介绍 ACT 论文，而是把当前仓库里与 ACT 和真实机械臂控制直接相关的代码路径讲清楚，尤其是下面这条主线：

`相机/关节观测 -> observation 规范化 -> ACT 输出 action chunk -> 取出单步 action -> 变成具名关节命令 -> 机器人执行`

本文重点是同步真实机器人执行链路，也会补一节 async inference 的差异说明。最后会给出在当前代码结构下接入 `pika` 夹爪 + `UR7e` 的落点蓝图。

---

## 1. 先看结论：当前同步主链路的总图

在真实机器人上，当前最典型的 ACT 执行路径不是 `lerobot-eval`，而是 `lerobot-record` 中的 `record_loop()`。当你给 `lerobot-record` 同时传入机器人和 policy 时，核心流程如下：

```text
robot.get_observation()
    -> 返回硬件原始观测 dict
    -> build_dataset_frame()
    -> predict_action()
        -> prepare_observation_for_inference()
        -> ACT preprocessor
        -> ACTPolicy.select_action()
            -> 若 action queue 为空，则先 predict_action_chunk()
            -> 从 chunk 中取出一个单步 action
        -> ACT postprocessor
    -> make_robot_action()
    -> robot.send_action()
    -> precise_sleep() 控频
```

如果把关键函数名带上，代码主路径大致是：

1. `src/lerobot/scripts/lerobot_record.py:record_loop`
2. `src/lerobot/utils/control_utils.py:predict_action`
3. `src/lerobot/policies/act/processor_act.py:make_act_pre_post_processors`
4. `src/lerobot/policies/act/modeling_act.py:ACTPolicy.select_action`
5. `src/lerobot/policies/utils.py:make_robot_action`
6. 具体机器人类的 `send_action()`，例如：
   - `src/lerobot/robots/so_follower/so_follower.py`
   - `src/lerobot/robots/openarm_follower/openarm_follower.py`
   - `src/lerobot/robots/reachy2/robot_reachy2.py`

理解这条链路时，最重要的不是某一个神经网络层，而是三套“数据表示”之间的转换。

---

## 2. 三套数据表示：这是理解整套代码的核心

LeRobot 里同一个时刻的数据，通常同时存在三种表示：

### 2.1 机器人硬件表示

这是 `Robot.get_observation()` 和 `Robot.send_action()` 直接面对的格式。

典型例子：

```python
{
    "shoulder_pan.pos": 12.3,
    "shoulder_lift.pos": -8.1,
    "elbow_flex.pos": 45.0,
    "gripper.pos": 30.0,
    "front": np.ndarray(H, W, 3),
    "wrist": np.ndarray(H, W, 3),
}
```

这里的 key 是“硬件侧命名”。对机器人类来说，这就是原生接口。

### 2.2 数据集表示

LeRobot 录制和训练时，会把硬件数据整理成标准字段名：

- `observation.state`
- `observation.images.front`
- `observation.images.wrist`
- `action`

其中：

- `observation.state` 是一个向量，不再是多个散开的 `*.pos` 字段。
- `action` 也是一个向量，不再是多个散开的 `*.pos` 字段。
- 图像字段统一挂在 `observation.images.*` 下。

### 2.3 policy 表示

policy 侧进一步把数据解释为 `PolicyFeature`：

- `FeatureType.STATE`
- `FeatureType.VISUAL`
- `FeatureType.ACTION`
- `FeatureType.ENV`

同时附带 shape。例如：

- `observation.state -> shape=(7,)`
- `observation.images.front -> shape=(3, 480, 640)`
- `action -> shape=(7,)`

ACT 看到的不是原始硬件 dict，而是已经转换为 policy 表示、并且经过设备搬运和归一化之后的 tensor。

### 2.4 为什么这三层必须分开理解

因为新增硬件时，最容易出错的不是“模型会不会跑”，而是：

1. 你在机器人层定义的关节顺序，是否和数据集中的 `action.names` 一致。
2. 你在数据集中记录的 camera key，是否和 policy config 期望的 key 一致。
3. 你在运行时发送给机器人的是 joint-space 命令还是别的语义，而训练数据中的 `action` 是否也是同一语义。

这三个地方只要有一个不一致，就会出现“模型输出看起来正常，但执行到机器人上完全错位”的问题。

---

## 3. 背景与术语

下面这些术语在代码里反复出现。

### 3.1 `observation.state`

这是机器人低维状态向量。对真实机械臂来说，通常由关节位置组成，也可能包含速度、力矩或其他标量状态。

在当前多数 follower 机器人实现里，`observation.state` 最终通常来自多个 `*.pos` 字段按固定顺序拼成的向量。

### 3.2 `observation.images.*`

这是视觉输入。ACT 支持多相机，多路图像字段会以：

- `observation.images.front`
- `observation.images.side`
- `observation.images.wrist`

这样的 key 出现。

注意两点：

1. camera key 必须稳定。
2. 对 ACT 来说，目前默认假设所有图像 shape 相同。

### 3.3 `action`

在数据集和 policy 侧，`action` 是一个连续向量。

但在机器人硬件侧，`send_action()` 接收的往往是具名 dict，例如：

```python
{
    "shoulder_pan.pos": 10.0,
    "shoulder_lift.pos": -5.0,
    "gripper.pos": 20.0,
}
```

这两者之间的映射由 `make_robot_action()` 完成。

### 3.4 `chunk_size`

ACT 一次不是预测一个动作，而是预测一段未来动作序列。这个序列长度就是 `chunk_size`。

例如：

- `chunk_size = 100`

表示一次前向会输出未来 100 个时刻的动作。

### 3.5 `n_action_steps`

ACT 并不一定把一个 chunk 的所有动作都执行掉。`n_action_steps` 表示每次从新预测的 chunk 中实际消费多少步。

例如：

- `chunk_size = 100`
- `n_action_steps = 20`

表示：

1. 模型预测 100 步未来动作。
2. 当前只把前 20 步放进执行队列。
3. 执行完后，再重新基于最新观测预测下一段。

### 3.6 `action_is_pad`

训练时需要从数据集里取一个未来动作窗口。如果当前帧已经接近 episode 尾部，未来窗口超出边界的那部分会被 padding，并通过 `action_is_pad` 掩码标记出来。

ACT 训练时会：

- 用它屏蔽 L1 reconstruction loss 中不合法的未来动作位置；
- 在启用 VAE 的训练路径里，把这些 padded action token 作为 transformer key padding mask 的一部分。

### 3.7 `rename_map`

`rename_map` 用于在 observation 输入侧做字段名重映射，常见用途是“数据集/机器人上的 camera 名称”和“policy config 中保存的 camera 名称”不一致时做适配。

例如：

```python
{
    "observation.images.front": "observation.images.camera1"
}
```

它只改 observation key，不改 action 语义和 action 顺序。

### 3.8 dataset features 与 policy features

可以把二者理解为：

- dataset features：录制/加载数据时的 schema
- policy features：模型看到的 schema

二者的桥梁是 `dataset_to_policy_features()`。

其中一个特别重要的规则是：图像在 dataset feature 里通常是 `(H, W, C)`，进入 policy feature 后会转成 `(C, H, W)`。

---

## 4. ACT 本体在仓库里是怎么实现的

ACT 相关代码主要在：

- `src/lerobot/policies/act/configuration_act.py`
- `src/lerobot/policies/act/modeling_act.py`
- `src/lerobot/policies/act/processor_act.py`

### 4.1 `ACTConfig` 规定了什么

`ACTConfig` 是整个 ACT 行为的约束中心。最重要的参数有：

- `n_obs_steps`
- `chunk_size`
- `n_action_steps`
- `input_features`
- `output_features`
- `vision_backbone`
- `use_vae`
- `latent_dim`
- `temporal_ensemble_coeff`

当前实现里有几个非常重要的约束：

1. `n_obs_steps` 目前只能是 1。
   - 也就是说 ACT 在这个仓库里默认只吃“当前时刻观测”，不是多帧历史堆叠。
2. `n_action_steps <= chunk_size`
3. 如果启用 temporal ensembling，则 `n_action_steps` 必须是 1。
4. 至少要有一个视觉输入，或者有 `observation.environment_state`。
5. `action` 必须是输出。

### 4.2 ACT 输入输出的实际语义

对 ACT 来说，最典型的输入是：

- 多路 RGB 图像
- 当前机器人状态 `observation.state`
- 可选的 `observation.environment_state`

输出是：

- 一个长度为 `chunk_size` 的未来动作序列，shape 为 `(B, chunk_size, action_dim)`

这里的 `action_dim` 不是由 ACT 自己决定，而是来自 `config.output_features["action"].shape[0]`。

所以 ACT 不关心“这是 6 轴 UR 机械臂还是 7 轴 OpenArm”，它只关心你告诉它 action 向量有几维。

### 4.3 VAE 只在训练路径里真正工作

`ACT` 类内部实现里，训练和推理有一个关键分叉：

#### 训练时

如果：

- `config.use_vae = True`
- `self.training = True`
- batch 中带有 `action`

那么模型会走 VAE encoder 路径：

1. 把 `[CLS, robot_state, action_sequence]` 编成 token 序列。
2. 通过 `vae_encoder` 得到 latent 分布参数 `mu` 和 `log_sigma_x2`。
3. 用 reparameterization trick 采样 latent。
4. 再把这个 latent 作为主 transformer encoder 的一个输入 token。

#### 推理时

推理阶段没有未来 `action` 作为条件，因此不会走 VAE encoder。

当前实现直接把 latent 设为全零向量。

这意味着：

- VAE 是训练时的建模手段；
- 推理时 ACT 的执行路径更像“固定零 latent 条件下的条件动作生成器”。

### 4.4 图像是怎么进入 transformer 的

如果有图像输入，每路相机会：

1. 先通过 ResNet backbone 提取 feature map。
2. 再通过 `1x1 conv` 投影到 `dim_model`。
3. 把 feature map flatten 成 token 序列。
4. 配上二维正弦位置编码。

然后所有相机的 spatial token 会被串接到 encoder token 序列后面。

encoder 的 token 顺序大致是：

1. latent token
2. `observation.state` token（如果有）
3. `observation.environment_state` token（如果有）
4. 所有 camera 的图像 token

这点非常重要：多相机不是“先融合成一张图”，而是“每个相机各自产生一串 token，再拼到一起交给 transformer encoder”。

### 4.5 decoder 为什么输出的是一段动作

ACT 的 decoder 不是自回归地一个个预测动作，而是像 DETR 一样，为长度为 `chunk_size` 的每个未来时刻放一个 learnable query embedding。

于是 decoder 输出天然就是一个长度为 `chunk_size` 的序列，再经过 `action_head` 线性层得到：

```python
(B, chunk_size, action_dim)
```

这就是 action chunk。

### 4.6 `predict_action_chunk()` 和 `select_action()` 的区别

这是 ACT 在工程上最关键的一组接口。

#### `predict_action_chunk()`

只负责：

1. 准备 batch
2. 调用 ACT 主模型
3. 返回完整 action chunk

它不会决定“当前到底执行哪一步”。

#### `select_action()`

这是执行接口。它会把 action chunk 变成单步 action。

当前实现有两种模式。

##### 模式 A：普通 action queue

当 `temporal_ensemble_coeff is None` 时：

1. 如果内部 `_action_queue` 为空，就调用 `predict_action_chunk()`。
2. 从返回的 chunk 里取前 `n_action_steps`。
3. 把它们按时间顺序放进 deque。
4. 每次 `select_action()` 只 `popleft()` 一个单步动作返回。

所以“模型一次预测多步，但控制循环每次只执行一步”的核心，是 `ACTPolicy` 内部这个 action queue。

##### 模式 B：temporal ensembling

当 `temporal_ensemble_coeff` 非空时：

1. 每个控制周期都重新预测一个完整 chunk。
2. `ACTTemporalEnsembler` 在线融合重叠时间步上的多个预测。
3. 只取融合后的当前一步 action。

这个模式下要求 `n_action_steps = 1`，因为每一步都要重新推理。

### 4.7 训练损失是怎么构造的

在 `ACTPolicy.forward()` 里：

1. 先拿到 `actions_hat`
2. 用 `F.l1_loss(..., reduction="none")`
3. 用 `~batch["action_is_pad"]` 屏蔽 padding 位置
4. 对有效位置求均值，得到 reconstruction loss
5. 若开启 VAE，再加上 `KL * kl_weight`

也就是说当前 LeRobot 里的 ACT 训练损失本质上是：

```text
L = masked_L1(action_hat, action) + kl_weight * KL
```

---

## 5. 训练侧最小必要上下文：ACT 为什么需要未来动作窗口

要理解推理时为什么会有 action chunk，必须先理解训练时数据是怎么组织的。

### 5.1 `action_delta_indices` 决定未来动作窗口

在 `ACTConfig` 里：

- `observation_delta_indices -> None`
- `action_delta_indices -> list(range(chunk_size))`

这意味着：

- observation 只取当前帧
- action 会取当前时刻开始、长度为 `chunk_size` 的未来动作窗口

在训练脚本中，这些 index 会被转换成 `delta_timestamps`。

例如：

```python
delta_timestamps = {
    "action": [0/fps, 1/fps, 2/fps, ..., (chunk_size-1)/fps]
}
```

然后 `LeRobotDataset` / `DatasetReader` 会根据这个窗口，把单帧样本变成：

- `observation.state -> 当前时刻`
- `observation.images.* -> 当前时刻`
- `action -> 未来 chunk_size 步`

### 5.2 `action_is_pad` 处理 episode 边界

如果当前样本已经靠近 episode 尾部，未来动作窗口会越界。

`DatasetReader._get_query_indices()` 会：

1. 把越界索引 clamp 到 episode 边界内；
2. 同时生成对应的 padding mask：

```python
"action_is_pad": BoolTensor([...])
```

这样训练时：

- 数据 tensor shape 始终合法；
- 但 loss 不会把越界 padding 当成真实监督。

### 5.3 dataset stats 如何驱动归一化

LeRobot 在训练和推理里都使用 processor pipeline。

对 ACT 来说，默认 pre/post processor 是：

- preprocessor：
  - `RenameObservationsProcessorStep`
  - `AddBatchDimensionProcessorStep`
  - `DeviceProcessorStep`
  - `NormalizerProcessorStep`
- postprocessor：
  - `UnnormalizerProcessorStep`
  - `DeviceProcessorStep(device="cpu")`

其中 `NormalizerProcessorStep` 和 `UnnormalizerProcessorStep` 依赖 dataset stats。

默认归一化映射是：

- `VISUAL -> MEAN_STD`
- `STATE -> MEAN_STD`
- `ACTION -> MEAN_STD`

所以训练和推理都不是把原始数值直接送进模型，而是先按数据集统计量归一化。

这也是为什么“换一台新机器人重新录数据”后，如果 action 或 state 语义变了，你不应该直接复用旧 stats。

---

## 6. 同步真实机器人主链路：代码是怎样一层层走下去的

下面按真实执行顺序展开。

### 6.1 入口：`lerobot-record` 的 `record_loop()`

真实机器人同步执行 ACT 的主入口在：

- `src/lerobot/scripts/lerobot_record.py:record_loop`

这个循环每一帧大致做以下事情：

1. `robot.get_observation()`
2. `robot_observation_processor(obs)`，默认是 identity
3. `build_dataset_frame(...)` 生成标准 observation frame
4. `predict_action(...)`
5. `make_robot_action(...)`
6. `robot_action_processor(...)`，默认是 identity
7. `robot.send_action(...)`
8. `dataset.add_frame(...)`（如果在录数据）
9. `precise_sleep(...)` 保持目标 FPS

也就是说，ACT 控制真实机器人时，运行时骨架其实非常朴素：读取、变换、推理、下发、控频。

### 6.2 第一步：`robot.get_observation()` 读取硬件原始观测

这是机器人类最底层的同步读接口。

以 `SOFollower.get_observation()` 为例，它会：

1. 从电机总线读当前位置；
2. 把结果变成：

```python
{
    "shoulder_pan.pos": ...,
    "shoulder_lift.pos": ...,
    ...
}
```

3. 再逐个相机调用 `cam.read_latest()`，把图像放进同一个 dict：

```python
{
    "shoulder_pan.pos": ...,
    ...
    "front": np.ndarray(H, W, 3),
    "wrist": np.ndarray(H, W, 3),
}
```

OpenArm 和 Reachy2 也是一样的模式，只是底层总线和 SDK 不同。

当前 LeRobot 对机器人层没有强制总线类型要求。你可以是：

- 串口总线
- CAN 总线
- TCP/UDP/RTDE
- 厂商 Python SDK

只要你能实现 `Robot` 抽象类约定的同步方法即可。

### 6.3 第二步：`build_dataset_frame()` 把硬件观测转成标准 observation 字段

这一层非常重要。

在 `record_loop()` 里，代码会先得到：

```python
observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
```

`build_dataset_frame()` 做的事不是神经网络处理，而是 schema 映射。

它会根据 `dataset.features` 里的定义：

1. 把散开的关节值按固定顺序拼成 `observation.state`
2. 把原始 camera key 映射成 `observation.images.<camera>`

例如如果 `dataset.features` 中有：

```python
"observation.state": {
    "names": [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
}
```

那么 `build_dataset_frame()` 会严格按这个 `names` 顺序生成 `observation.state` 向量。

这说明一个极其关键的事实：

> 运行时 observation/state 的维度顺序，不是从模型里“推断”出来的，而是由数据集 schema 明确规定的。

同理，图像字段也是由 dataset feature 决定的：

- `front -> observation.images.front`
- `wrist -> observation.images.wrist`

如果 `Robot.get_observation()` 返回了额外字段，但 dataset features 里没有，它们会在这里被丢掉。

### 6.4 第三步：`predict_action()` 进入 policy 推理

`predict_action()` 在：

- `src/lerobot/utils/control_utils.py`

它内部又分成几步。

### 6.4.1 `prepare_observation_for_inference()`

这是第一次真正把 numpy 数据变成 policy 可吃的 tensor。

它会对 observation dict 做：

1. `numpy -> torch`
2. 如果 key 里包含 `"image"`：
   - 转成 `float32`
   - 除以 255，变成 `[0, 1]`
   - 从 `(H, W, C)` 变成 `(C, H, W)`
3. 给所有 tensor 增加 batch 维
4. 把 tensor 移到目标 device
5. 附加：
   - `task`
   - `robot_type`

这一步之后，单帧 observation 已经从“原始硬件格式”进入了“模型输入前格式”。

注意一个容易忽略的点：

> 同步链路下，这里不会自动做图像 resize。

所以当前同步真实机器人使用 ACT 时，最安全的做法是：

1. 相机 key 与训练时一致；
2. 相机分辨率与训练时一致；
3. 如果不一致，就自己加 processor 或额外适配逻辑。

这也是 `examples/tutorial/act/act_using_example.py` 明确提醒“camera keys 和 resolutions 必须与训练一致”的原因。

### 6.4.2 ACT preprocessor

ACT 默认 preprocessor 来自：

- `src/lerobot/policies/act/processor_act.py`

顺序是：

1. `RenameObservationsProcessorStep`
2. `AddBatchDimensionProcessorStep`
3. `DeviceProcessorStep`
4. `NormalizerProcessorStep`

其中：

#### `RenameObservationsProcessorStep`

用 `rename_map` 做 observation key 重命名。

这个步骤最常用于 camera key 适配。

#### `AddBatchDimensionProcessorStep`

在同步推理路径里，`prepare_observation_for_inference()` 已经加过 batch 维了，所以这一步很多时候不会真正改 shape。

它更像一个防御性步骤，用来保证：

- 如果某些输入还没 batched，就补上；
- 同一份 preprocessor 在训练和推理都可复用。

#### `DeviceProcessorStep`

确保 tensor 在正确设备上。

#### `NormalizerProcessorStep`

根据 dataset stats 对 state / image 做归一化。

到这里为止，ACT 看到的输入已经不是原始关节角和原始像素，而是规范化后的 policy tensor。

### 6.5 第四步：`ACTPolicy.select_action()` 从 chunk 中拿出一个动作

这是整个 ACT 执行语义最关键的函数。

### 6.5.1 先组装多相机输入

如果 ACT config 里声明了多路图像，`select_action()` 内部最终会走到：

```python
predict_action_chunk(batch)
```

而 `predict_action_chunk()` 会把：

```python
batch["observation.images.front"]
batch["observation.images.wrist"]
```

按 `self.config.image_features` 的顺序整理成：

```python
batch["observation.images"] = [img1, img2, ...]
```

所以多相机顺序也是有语义的。它通常来自 `input_features` 的顺序，而 `input_features` 又通常来自数据集 metadata。

### 6.5.2 ACT 主模型返回的是 action chunk

主模型 `self.model(batch)` 返回：

```python
actions: (B, chunk_size, action_dim)
```

这不是“当前动作”，而是一个未来动作序列。

### 6.5.3 `select_action()` 为什么只返回一个动作

因为 `select_action()` 在普通模式下维护了内部 `_action_queue`：

1. 当队列为空时，重新调用 `predict_action_chunk()`
2. 只取 chunk 的前 `n_action_steps`
3. 按时间顺序塞进队列
4. 每次调用只取一个 `popleft()`

因此：

- 控制循环每 tick 执行一个单步动作；
- 但模型不必每 tick 都重新推理。

这正是 ACT 的 action chunking 在工程上的落地方式。

### 6.5.4 如果用了 temporal ensembling

当 `temporal_ensemble_coeff` 非空时，不再使用普通队列，而是：

1. 每步重算一个 chunk
2. 对重叠时刻的动作做指数加权在线融合
3. 取当前一步动作

这个模式更接近“每步都重规划，但保留 chunk 预测的平滑性”。

### 6.6 第五步：ACT postprocessor 把输出动作还原到执行空间

ACT postprocessor 只有两步：

1. `UnnormalizerProcessorStep`
2. `DeviceProcessorStep(device="cpu")`

它做的事情是：

1. 把模型输出的归一化 action 还原回数据集动作空间；
2. 把 tensor 拉回 CPU。

所以 `policy.select_action()` 返回的只是“模型动作”，而 postprocessor 返回的才是“可解释为真实动作语义的 action tensor”。

### 6.7 第六步：`make_robot_action()` 把 action tensor 变成具名关节命令

这一步在：

- `src/lerobot/policies/utils.py:make_robot_action`

它会：

1. 去掉 batch 维
2. 遍历 `dataset.features["action"]["names"]`
3. 按名字把 action tensor 的每一维映射成具名 dict

例如：

```python
action tensor = [10.0, -5.0, 30.0]
names = ["joint_1.pos", "joint_2.pos", "gripper.pos"]
```

会变成：

```python
{
    "joint_1.pos": 10.0,
    "joint_2.pos": -5.0,
    "gripper.pos": 30.0,
}
```

这又一次说明：

> 同步链路里的 action 维度顺序，最终由 `dataset.features["action"]["names"]` 决定。

如果这个顺序错了，模型输出的每一维就会被送到错误的关节上。

### 6.8 第七步：`robot.send_action()` 把具名命令送到硬件

最终控制由具体机器人类实现。

### SOFollower

`SOFollower.send_action()` 会：

1. 从 `{"joint.pos": value}` 提取出目标关节位置；
2. 可选地根据 `max_relative_target` 做相对位移安全裁剪；
3. 调用 Feetech 总线的 `sync_write("Goal_Position", ...)`。

### OpenArmFollower

`OpenArmFollower.send_action()` 会：

1. 提取目标关节位置；
2. 先按配置好的 joint limits 裁剪；
3. 可选地做 `max_relative_target` 裁剪；
4. 组装 MIT control 批量命令；
5. 通过 CAN 总线下发。

### Reachy2Robot

`Reachy2Robot.send_action()` 会：

1. 把 LeRobot key 映射到厂商 SDK 的 joint key；
2. 可选地做 `max_relative_target` 限幅；
3. 调用 Reachy SDK 设置 `goal_position` 并发送。

这里可以看出 LeRobot 的设计取向：

- policy 层不理解底层硬件协议；
- 机器人类负责把统一 action schema 翻译成厂商接口。

### 6.9 第八步：`precise_sleep()` 控制循环频率

同步链路最后会用：

- `src/lerobot/utils/robot_utils.py:precise_sleep`

来维持目标 FPS。

如果某一帧总耗时超过 `1/fps`，日志会警告：

- 相机跟不上
- policy 推理太慢
- CPU 饥饿

这对 ACT 很重要，因为 action chunking 虽然减少了“每步都推理”的成本，但并不意味着推理无限便宜。实际部署依然受控频影响。

---

## 7. 特征名、顺序与维度绑定：这是后续接新硬件最容易踩坑的地方

这一节单独强调，因为它几乎决定了你后续接入新设备会不会一次成功。

### 7.1 数据集 schema 是同步链路里的真源头

在同步链路中，下面几件事都受 dataset metadata 控制：

1. `observation.state` 的维度顺序
2. `action` 的维度顺序
3. `observation.images.*` 的 camera key
4. 归一化 stats 的 key

更具体地说：

- `build_dataset_frame()` 用 dataset features 决定如何把 raw robot observation 拼成 `observation.state`
- `make_robot_action()` 用 dataset features 决定如何把 action tensor 还原成具名关节命令

所以对同步 ACT 部署来说，dataset metadata 不是附属文件，而是 runtime schema。

### 7.2 `action.names` 的顺序就是动作维度顺序

LeRobot 生成 dataset features 时，会把机器人 `action_features` 里的键按字典插入顺序写进：

```python
dataset.features["action"]["names"]
```

而后续运行时又按这个顺序反向映射回去。

因此：

1. 你在 `Robot.action_features` 里声明关节的顺序要稳定；
2. 你录数据、训练、推理时都必须保持这个顺序一致；
3. 一旦顺序改了，旧数据和旧模型就不能直接混用。

### 7.3 `observation.state.names` 同样有顺序语义

和 action 一样，`observation.state` 也是按 `names` 顺序拼出来的向量。

因此如果你后续在 `UR7e + pika` 里决定：

- 只放 6 个 arm joints
- 还是放 6 个 arm joints + 1 个 gripper state
- 还是放 joint position + velocity

这必须在数据层先拍板，而不是等模型训练完再调整。

### 7.4 camera key 必须与 policy 期望一致

ACT 多相机输入依赖 camera key。

例如模型可能保存的是：

- `observation.images.front`
- `observation.images.wrist`

那你运行时要么：

1. 机器人本来就返回 `front` / `wrist`
2. 要么用 `rename_map` 把当前 key 映射到这两个名字

如果 key 完全不对，模型会直接拿不到对应输入。

### 7.5 同步链路默认不帮你 resize 图像

这是一个经常被忽略的限制。

同步 `predict_action()` 路径里，图像只做：

- `uint8 -> float32`
- `/255`
- `HWC -> CHW`

不会自动 resize 到 policy shape。

所以：

- 如果训练时是 `640x480`
- 运行时却接成 `1280x720`

当前同步链路并不会自动替你对齐。

如果你想放宽这个约束，需要自己加 processor 或单独改前处理逻辑。

### 7.6 `rename_map` 只解决“名字不一致”，不解决“语义不一致”

`rename_map` 能做的是：

- `front -> camera1`

不能做的是：

- 把 wrist camera 当成 front camera 的语义替代
- 把 6 维 joint action 变成 7 维 joint+gripper action
- 把 degree 单位变成 radian 单位

所以不要把 `rename_map` 当成通用适配层。它只是 observation key rename。

---

## 8. 机器人抽象层：LeRobot 是怎么把不同机械臂放进同一个框架里的

当前机器人抽象基类在：

- `src/lerobot/robots/robot.py`

它要求所有机器人实现同一组接口。

### 8.1 `Robot` 抽象类真正规定了什么

你必须实现：

- `observation_features`
- `action_features`
- `is_connected`
- `connect()`
- `is_calibrated`
- `calibrate()`
- `configure()`
- `get_observation()`
- `send_action()`
- `disconnect()`

这些接口背后有几个重要约束：

1. `observation_features` / `action_features` 在机器人未连接时也必须可调用。
2. `get_observation()` 和 `send_action()` 应该是同步接口。
3. `send_action()` 最好返回“实际发送出去的动作”，因为它可能经过安全裁剪。

### 8.2 `RobotConfig` 的作用

`RobotConfig` 是机器人配置入口，所有机器人配置都通过 `@RobotConfig.register_subclass(...)` 注册。

如果机器人有相机，`RobotConfig.__post_init__()` 会要求每路相机至少提供：

- `width`
- `height`
- `fps`

也就是说 camera config 是机器人 schema 的一部分，不是临时参数。

### 8.3 `observation_features` / `action_features` 是 schema 声明，不是运行逻辑

这两个属性的核心价值是：

- 描述机器人对外暴露的观测/动作接口
- 为数据集 schema 和上层工具提供结构信息

例如 `SOFollower`：

- observation features：各关节 `.pos` + 各相机图像
- action features：各关节 `.pos`

例如 `OpenArmFollower`：

- observation features：`.pos`、`.vel`、`.torque` + 图像
- action features：仍以具名位置命令为主

例如 `Reachy2Robot`：

- observation/action key 先是 LeRobot 自己的命名
- 再通过内部映射表翻译到 Reachy SDK 的 key

这说明 LeRobot 并不要求底层厂商接口统一，只要求你在 Robot 层把接口整理成统一 schema。

### 8.4 现有三个实现的对比

### SOFollower

特点：

- 串口 + Feetech 总线
- 观测/动作都比较纯粹，基本是 joint position
- `send_action()` 中带有相对位移安全裁剪

它代表了当前代码库里最“标准”的 low-cost arm 路线。

### OpenArmFollower

特点：

- CAN + Damiao 电机
- 观测包含 `pos/vel/torque`
- `send_action()` 里除了安全裁剪，还会做 joint limit 限幅和 MIT control 组包

它说明 LeRobot 的 `send_action()` 完全可以容纳更复杂的控制实现。

### Reachy2Robot

特点：

- 网络 SDK 机器人
- 内部有 LeRobot key 到 SDK key 的映射表
- 支持更多部件和移动底座

它说明 LeRobot 的 Robot 抽象并不依赖某种特定总线，只依赖“把厂商接口包成统一同步接口”。

### 8.5 当前默认 processor 基本是 identity

`make_default_processors()` 默认返回：

- teleop action processor：identity
- robot action processor：identity
- robot observation processor：identity

这意味着当前很多真实机器人链路其实很薄：

- 原始观测基本直接进 dataset frame / policy
- policy 输出也基本直接变成 robot action

这既是优点，也是风险点：

- 优点：链路简单、容易追踪
- 风险：一旦你需要加单位变换、裁剪、滤波、额外状态组合，就得明确决定放在哪一层

---

## 9. Async inference 和同步链路有什么本质不同

当前 async 代码主要在：

- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/policy_server.py`
- `src/lerobot/async_inference/helpers.py`

要点是：

> async 不是另一种 ACT 模型，而是另一种“观测发送、推理、动作执行”的系统架构。

ACT 预测 action chunk 的语义本身没有变。

### 9.1 同步链路 vs async 链路

### 同步链路

同一台机器本地顺序执行：

1. 读观测
2. 推理
3. 得到 action
4. 发给机器人

### async 链路

拆成 client/server：

- `RobotClient`：
  - 负责读机器人观测
  - 维护本地 action queue
  - 执行动作
  - 把观测流式发送到 server
- `PolicyServer`：
  - 负责加载 ACT / 其他 policy
  - 跑 preprocess / inference / postprocess
  - 把 action chunk 返回给 client

所以 async 的价值是：

- 机器人执行当前 action queue 时，server 已经在并行算下一段 chunk
- 尽量减少“机器人空等推理结果”的时间

### 9.2 async 的观测处理方式

client 侧会先把 `robot.observation_features` 转成 LeRobot 风格特征描述：

```python
map_robot_keys_to_lerobot_features(robot)
```

server 侧收到 raw observation 后，会走：

```python
raw_observation_to_observation()
```

它会：

1. 先用 `build_dataset_frame()` 把 raw robot obs 变成 LeRobot observation 风格；
2. 抽出 `observation.state`；
3. 找出所有图像；
4. 按 policy 保存的 image shape 对图像做 resize；
5. 把图像变成 `(B, C, H, W)`；
6. 再送进 preprocessor。

这和同步链路有一个显著区别：

> async helper 里显式做了图像 resize，而同步 `predict_action()` 默认没有。

### 9.3 async 的动作执行方式

server 侧：

1. 跑 preprocessor
2. `policy.predict_action_chunk()`
3. postprocessor
4. 给每个 action 附时间戳，打成 `TimedAction` 列表

client 侧：

1. 接收 action chunk
2. 用聚合函数和现有队列做合并
3. 从队列里逐个取单步 action 执行

也就是说：

- 同步链路里，ACT 内部自己维护 `_action_queue`
- async 链路里，client 额外维护了系统级 action queue，并可聚合重叠 chunk

但动作 chunk 的来源仍然是同一个 ACT。

### 9.4 async 的一个关键 caveat：动作维度映射来源不同

这是理解后续新硬件接入时非常重要的地方。

同步链路下：

- tensor action -> 具名 dict
- 用的是 `dataset.features["action"]["names"]`

async 链路下，`RobotClient` 执行动作时会调用：

```python
_action_tensor_to_action_dict()
```

它直接按：

```python
self.robot.action_features
```

的字典顺序来解释 action tensor 的每一维。

这意味着 async 部署时有一个额外要求：

> `robot.action_features` 的顺序必须与 policy 输出动作维度顺序一致。

当前 async 握手里传的是 observation feature，不传 action rename/mapping，所以这里更依赖你在机器人类里把 action schema 定义正确。

---

## 10. `pika + UR7e` 接入蓝图：在当前仓库里应该落在哪些层

下面不是最终实现代码，而是按当前代码结构给出的接入设计蓝图。

### 10.1 优先建议：先走 joint-space 语义

当前代码库里现成的机械臂接入范式几乎都偏 joint-space：

- `observation.state` 以关节量为主
- `action` 以关节目标量为主
- `send_action()` 负责把关节目标翻译到底层驱动

因此如果你要新增 `UR7e + pika`，最稳妥的第一版方案通常是：

1. `observation.state` 明确为 UR7e 关节状态 + gripper 状态
2. `action` 明确为 UR7e 关节目标 + gripper 目标

不是说不能走笛卡尔末端位姿 action，而是那会让数据语义、模型监督、运行时转换都变复杂。当前仓库现有实现对 joint-space 路径支持最好。

### 10.2 机器人层：新增 `RobotConfig` + `Robot` 子类

如果放在仓库内，建议最少有一组类似下面的结构：

```text
src/lerobot/robots/ur7e_pika/
    config_ur7e_pika.py
    robot_ur7e_pika.py
    __init__.py
```

你需要：

1. 定义配置类并注册：
   - `@RobotConfig.register_subclass("ur7e_pika")`
2. 定义机器人类：
   - `config_class = UR7ePikaConfig`
   - `name = "ur7e_pika"`

配置里至少要提前想清楚：

- UR 控制地址/端口
- gripper 通讯地址或 SDK 句柄参数
- 相机配置
- 控制频率
- 安全限幅参数
- 单位约定

### 10.3 先定义 observation/action schema，再写代码

新增硬件时，第一件事不是写 `send_action()`，而是把 schema 定下来。

你至少要明确：

### observation side

`observation_features` 里是否包含：

- `joint_1.pos` ... `joint_6.pos`
- `gripper.pos`
- 是否还包含 velocity / current / force
- 哪些 camera key

### action side

`action_features` 里是否只包含：

- `joint_1.pos` ... `joint_6.pos`
- `gripper.pos`

或者还要不要包含：

- gripper speed
- gripper force
- 其他低层控制量

建议第一版尽量简单：

- observation：joint position + gripper position + cameras
- action：joint target + gripper target

因为这样最符合当前 ACT + LeRobot 的主流使用方式。

### 10.4 `get_observation()` 应该承担什么

你的 `UR7ePikaRobot.get_observation()` 应该：

1. 从 UR7e SDK/RTDE 读当前关节状态
2. 从 pika 接口读当前 gripper 状态
3. 把结果组织成统一 dict
4. 再附上相机图像

需要特别注意的不是“能不能读到”，而是“你返回的数值单位是什么”。

必须提前决定：

- 关节量是 degree 还是 radian
- gripper 是 opening width、百分比，还是 vendor 原始刻度

这个决定必须与录数据、训练、推理、发送命令保持一致。

如果这里不一致，模型哪怕能训练，也只是在学习错误单位。

### 10.5 `send_action()` 应该承担什么

你的 `send_action()` 至少要负责：

1. 把 LeRobot action dict 拆成 arm 与 gripper 两部分
2. 把 LeRobot 约定单位转换到底层 SDK 单位
3. 施加安全约束
4. 发送到底层控制接口
5. 返回实际发送出去的动作

安全约束通常至少包括：

- 关节范围限制
- 单步最大相对位移限制
- 速度/加速度上限
- gripper 开口范围限制
- 通讯失败或状态异常时的保护策略

如果 UR7e 自身 SDK 已经有一部分安全策略，也建议在 `send_action()` 再包一层本地保护，这样上层 policy 不会直接把异常值原样打到硬件上。

### 10.6 相机层：优先复用现有 camera backend

如果你的相机能通过现有 backend 读取，优先复用：

- `opencv`
- `realsense`
- `zmq`

只要把 camera key 设计好，机器人类里继续用：

```python
self.cameras = make_cameras_from_configs(config.cameras)
```

即可。

关键不是“哪种相机”，而是：

1. key 是否稳定
2. 分辨率是否与你训练数据一致
3. 这些 key 是否会写入 dataset metadata 并最终变成 policy input feature

### 10.7 数据层：真正的兼容性是在数据里锁定的

如果你打算后续训练 ACT 来控制 `UR7e + pika`，那你录数据时就必须确保：

1. `observation.state` 维度顺序固定
2. `action` 维度顺序固定
3. `observation.images.*` camera key 固定
4. 运行时使用的 schema 与训练数据 schema 完全同构

这里“完全同构”至少包括：

- 维度数一致
- 各维含义一致
- 顺序一致
- 单位一致

最常见的错误是：

- 训练时 `action = [q1, q2, q3, q4, q5, q6, gripper]`
- 推理时却按 `[gripper, q1, q2, q3, q4, q5, q6]` 去解释

从模型视角看它没错，从执行视角看它会完全错位。

### 10.8 sync 和 async 下分别要注意什么

### 如果先走同步链路

要重点保证：

1. `dataset.features["action"]["names"]` 顺序正确
2. camera key 和分辨率与训练一致
3. `make_robot_action()` 生成的具名 dict 能被 `send_action()` 直接解释

### 如果后面要走 async

还要额外保证：

1. `robot.action_features` 的顺序与 policy 输出动作顺序一致
2. client 的 `_action_tensor_to_action_dict()` 不会把 action 维度错映射到错误关节

所以对 `UR7e + pika` 来说，同步版本先打通通常是更稳妥的路径。因为同步路径里动作维度映射显式依赖 dataset metadata，更容易检查。

### 10.9 插件方式也是可行的

如果你不想直接改主仓库，也可以走插件机制。

当前代码会通过：

- `register_third_party_plugins()`

自动发现已安装的第三方包，命名约定包括：

- `lerobot_robot_*`
- `lerobot_camera_*`
- `lerobot_teleoperator_*`
- `lerobot_policy_*`

也就是说你可以把 `UR7e + pika` 做成外部插件包，例如：

```text
lerobot_robot_ur7e_pika
```

然后：

1. 在插件内定义并注册 `RobotConfig` 子类
2. 实现对应 `Robot` 类
3. 保证模块导入路径满足 `make_device_from_device_class()` 的约定

这种方式的优点是：

- 不污染主仓库
- 更适合厂商私有 SDK 或实验性硬件

---

## 11. 一个实用的实现顺序建议

如果你下一步真的要做 `pika + UR7e`，建议按下面顺序推进。

### 第一步：先把 schema 写死

先明确：

1. `observation_features`
2. `action_features`
3. camera key
4. 所有量的单位

不先锁 schema，后面录数据和接 policy 都会反复返工。

### 第二步：先打通最小同步机器人闭环

目标不是马上接 ACT，而是先做到：

1. `robot.connect()`
2. `robot.get_observation()`
3. `robot.send_action()`

可以用一个假的 action dict 人工下发，确认 arm 和 gripper 都能正常响应。

### 第三步：录一小段数据并检查 metadata

录完之后优先检查：

- `dataset.meta.features`
- `dataset.meta.stats`
- `dataset.features["observation.state"]["names"]`
- `dataset.features["action"]["names"]`

这一步其实是在验证你的 schema 是否真的被 LeRobot 正确记录了。

### 第四步：用最小 ACT 流程跑通推理

可以参考：

- `examples/tutorial/act/act_using_example.py`

先做最小同步推理闭环，确认：

1. observation 能进模型
2. 模型输出 shape 对
3. `make_robot_action()` 的维度映射对
4. `send_action()` 能吃这个 dict

### 第五步：再考虑 async 或更复杂控制

当同步版已经稳定后，再考虑：

- async inference
- 更强的安全层
- 更复杂的 observation
- 额外 processor

这样排查问题会容易得多。

---

## 12. 最后总结

当前仓库里的 ACT 控制真实机械臂，核心不是“模型直接看图然后控制电机”，而是下面这套分层系统：

1. `Robot` 层负责和真实硬件交互，暴露统一的原始 observation/action 接口
2. dataset feature 层把原始硬件字段整理成标准 schema
3. policy preprocessor 把标准 schema 变成 ACT 可吃的归一化 tensor
4. ACT 一次输出一段 future action chunk
5. `select_action()` 把 chunk 变成单步执行动作
6. postprocessor 把动作还原回真实动作空间
7. `make_robot_action()` 再把动作向量还原成具名关节命令
8. `robot.send_action()` 最终把统一 action schema 翻译到底层总线或 SDK

对后续新增 `pika + UR7e` 支持来说，最关键的结论是：

- 真正的“控制语义”要先在数据层和机器人 schema 层定清楚；
- ACT 本身只关心连续向量，不替你决定每一维代表什么；
- 一旦 observation/action 的命名、顺序、单位没锁住，后面训练和推理都会出问题。

如果把这一点抓牢，那么新增硬件支持的工作本质上就会收敛成三件事：

1. 定义稳定的 schema
2. 实现稳定的 `Robot` 适配层
3. 确保数据、模型、运行时三者使用同一套语义

这也是当前仓库中 ACT 与真实机器人控制逻辑的真正接缝所在。
