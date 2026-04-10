# PI0 Policy 在 LeRobot 中的实现分析报告

本文基于对以下四个文件的完整阅读与交叉理解而写：

- `src/lerobot/policies/pi0/__init__.py`
- `src/lerobot/policies/pi0/configuration_pi0.py`
- `src/lerobot/policies/pi0/processor_pi0.py`
- `src/lerobot/policies/pi0/modeling_pi0.py`

为了把实现讲透，文中还参考了少量直接依赖模块，包括：

- `src/lerobot/policies/pi_gemma.py`
- `src/lerobot/policies/rtc/*`
- `src/lerobot/processor/*`
- `src/lerobot/configs/policies.py`
- `src/lerobot/policies/factory.py`

但分析的主体始终聚焦在 `pi0` 目录本身，也就是你要求“完整理解”的那部分代码。

---

## 1. 先看结论

LeRobot 里的 `pi0` 不是一个“普通的图像到动作网络”，而是一个带有明显 OpenPI 风格的 Vision-Language-Action policy。它的实现核心可以概括成下面四句话：

1. 它把图像和语言作为前缀 token，把机器人状态、带噪动作序列和时间步作为后缀 token。
2. 它不是直接回归动作，而是用 flow matching 学一个速度场 `v_t`，训练目标是从噪声轨迹反推干净动作序列。
3. 它的 Transformer 主体不是单塔，而是一个“PaliGemma 前缀分支 + Gemma action expert 后缀分支”的联合结构。
4. 它在推理时先把视觉语言前缀做一次 cache/prefill，再对动作后缀做多步去噪；可选地用 RTC 把前一个 chunk 的剩余动作并入当前去噪过程。

从工程结构上看，`pi0` 目录里的四个文件分工非常清楚：

- `__init__.py`: 暴露公共 API。
- `configuration_pi0.py`: 定义所有超参数、默认特征形状、优化器/调度器预设。
- `processor_pi0.py`: 定义 preprocessor/postprocessor，负责文本换行、tokenize、归一化、相对动作变换等输入输出适配。
- `modeling_pi0.py`: 定义真正的模型、训练损失、推理流程、权重加载、action queue、RTC 接口。

如果只记一条主线，可以记成：

```text
原始 batch
-> pi0 preprocessor
-> 图像/语言/状态/动作整理
-> 联合 Transformer 预测 flow velocity
-> 训练时做 MSE(flow target, predicted velocity)
-> 推理时从噪声开始逐步 Euler 反积分得到动作 chunk
-> postprocessor 反归一化/转回绝对动作
```

---

## 2. 整体架构与数据流

### 2.1 模块级职责总览

| 模块 | 核心职责 | 你应该如何理解它 |
| --- | --- | --- |
| `__init__.py` | 导出 `PI0Config`、`PI0Policy`、`make_pi0_pre_post_processors` | 对外入口 |
| `configuration_pi0.py` | 把 OpenPI/LeRobot 所需超参数收拢成一个 dataclass | “这套 policy 的合同” |
| `processor_pi0.py` | 组织输入输出处理链 | “进入模型前、离开模型后该做什么” |
| `modeling_pi0.py` | 模型结构、训练公式、采样器、RTC、权重加载 | “PI0 真正怎么工作” |

### 2.2 训练时的数据流

训练阶段，`PI0Policy.forward()` 的主链路可以按下面顺序理解：

```text
原始 batch
-> preprocessor:
   - 加 batch 维
   - task 末尾补 '\n'
   - tokenizer 生成 language tokens / attention mask
   - 设备搬运
   - 可选相对动作变换
   - 状态/动作归一化
-> PI0Policy.forward():
   - 图像 resize/pad 到 224x224, 并映射到 [-1, 1]
   - state/action pad 到 max_state_dim/max_action_dim
-> PI0Pytorch.forward():
   - 采样 noise 和 time
   - 构造 flow matching 轨迹 x_t 与目标 u_t
   - embed_prefix(图像+语言)
   - embed_suffix(状态+带噪动作+时间)
   - 进入联合 Transformer
   - 输出预测速度 v_t
   - 计算 MSE(u_t, v_t)
```

### 2.3 推理时的数据流

推理阶段，`PI0Policy.predict_action_chunk()` 的逻辑与训练不同，关键在于“先缓存前缀，再反积分去噪”：

```text
原始 observation batch
-> preprocessor:
   - task 换行
   - tokenize
   - device
   - 可选 relative state/action 处理
   - state 归一化
-> PI0Policy.predict_action_chunk():
   - 图像预处理
   - state pad
-> PI0Pytorch.sample_actions():
   - 从高斯噪声 x_1 开始
   - prefix(图像+语言) 只跑一次，生成 past_key_values
   - 对 suffix(状态+动作 tokens) 做 num_inference_steps 次 denoise_step
   - 用 Euler 反积分更新 x_t
   - 可选 RTC 纠正 velocity
-> 得到 action chunk
-> postprocessor:
   - 动作反归一化
   - 可选从相对动作转回绝对动作
   - 搬回 CPU
```

### 2.4 这份实现最重要的三个设计点

#### 设计点 1：训练目标不是直接动作回归，而是 flow matching

这套实现学的是：

- 给定带噪动作 `x_t`
- 预测速度场 `v_t`
- 让 `v_t` 去拟合 `u_t = noise - action`

这样推理时可以从噪声出发，逐步“流”回动作。

#### 设计点 2：前缀和后缀不是两个完全独立的模型

代码里虽然有 `paligemma` 和 `gemma_expert` 两条分支，但训练时并不是“前缀先编码完再把结果拼接给后缀这么简单”，而是在每一层里手动把两边的 Q/K/V 拼到一起做联合 self-attention，再把输出拆回两条分支各自做 MLP。这是整份实现最关键、也最容易看漏的地方。

#### 设计点 3：相对动作、归一化、文本 tokenization 都不在模型主体里做

这些逻辑都被有意识地放到了 processor 层。这意味着：

- `modeling_pi0.py` 只关注“已经对齐好的张量”。
- `processor_pi0.py` 负责把 LeRobot 的 batch 变成模型想吃的格式。
- 这也是 `pi0` 可以和 LeRobot 的统一训练/部署基础设施对齐的原因。

---

## 3. 算法核心：PI0 在这里到底学了什么

### 3.1 Flow matching 训练目标

`PI0Pytorch.forward()` 里的核心公式是：

```text
noise ~ N(0, I)
time ~ scaled Beta(alpha, beta)
x_t = t * noise + (1 - t) * actions
u_t = noise - actions
loss = MSE(v_t, u_t)
```

其中：

- `actions` 是真实动作序列。
- `noise` 是高斯噪声。
- `x_t` 是时间 `t` 上的中间轨迹点。
- `u_t` 是这条线性轨迹的真速度。
- 模型输出 `v_t`，目标是逼近 `u_t`。

这意味着模型学到的不是“动作值本身”，而是“从当前带噪轨迹点往哪里流动”。

### 3.2 为什么推理时可以从噪声恢复动作

如果连续时间轨迹定义为：

```text
x_t = t * noise + (1 - t) * action
```

那么：

```text
dx_t / dt = noise - action = u_t
```

推理时代码从 `t = 1` 开始，此时 `x_1 = noise`。随后不断执行：

```text
x_t <- x_t + dt * v_t
```

其中 `dt = -1 / num_steps`，也就是向 `t = 0` 方向积分。只要 `v_t` 预测得足够准，最终 `x_0` 就会逼近真实动作。

### 3.3 前缀 token 和后缀 token 是怎么构成的

#### 前缀 prefix

由两部分组成：

- 图像 embedding
- 语言 token embedding

图像由 PaliGemma 的视觉塔编码，语言由 PaliGemma 的 token embedding 编码。

#### 后缀 suffix

由两部分组成：

- 一个 `state token`
- `chunk_size` 个动作 token

每个动作 token 不是纯动作，而是：

- `action_in_proj(noisy_action)`
- 与 `sin/cos timestep embedding` 拼接
- 过一个 MLP 融合

所以 suffix 本质上表示的是：

```text
当前机器人状态 + 当前时间 t 下的带噪未来动作序列
```

### 3.4 注意力掩码的真实语义

`make_att_2d_masks()` 不是普通的 causal mask。它用的是一个“分块累计掩码”的技巧：

- `prefix` 的 `att_mask` 全是 `0`
- `state token` 的 `att_mask` 是 `1`
- 第一个 action token 的 `att_mask` 是 `1`
- 后续 action token 的 `att_mask` 是 `0`

做 cumulative sum 后会形成三个 block：

1. 图像和语言都在 block 0
2. 状态 token 在 block 1
3. 全部动作 token 在 block 2

再通过：

```text
cumsum[:, None, :] <= cumsum[:, :, None]
```

生成二维 mask 后，含义就变成：

- block 0 里的前缀 token 只能看 block 0
- block 1 的状态 token 可以看 block 0 和 block 1
- block 2 的动作 token 可以看 block 0、block 1 和 block 2

这非常关键，因为它保证了：

- 图像语言前缀不会被动作 token 污染
- 状态可以读取前缀上下文
- 全部动作 token 之间是同块双向可见的，不是标准因果解码

这也解释了为什么 `pi0` 更像“对整段动作轨迹做条件去噪”，而不是一步一步自回归预测动作。

### 3.5 RTC 在这里扮演什么角色

RTC 只影响推理，不影响训练。它的作用是：

- 已经执行了一部分前一个 action chunk
- 还有一部分 leftover 没执行完
- 当前 chunk 生成时，希望与前一个 leftover 更平滑衔接

RTC 的代码不是写在 `pi0` 目录里，但 `PI0Pytorch.sample_actions()` 把它接进了 denoise 循环。具体做法是：

1. 先算普通的 `v_t`
2. 估计一步到 `t=0` 的 `x1_t`
3. 用 `prev_chunk_left_over - x1_t` 构造前缀误差
4. 通过 autograd 对 `x_t` 求修正方向
5. 得到修正后的 `v_t`

所以 RTC 本质上是“对 denoise velocity 施加一项基于前一 chunk leftover 的引导”。

---

## 4. 模块一：`src/lerobot/policies/pi0/__init__.py`

这个文件非常简单，但它定义了 `pi0` 对外的公共接口。

### 模块作用

- 从 `configuration_pi0.py` 导出 `PI0Config`
- 从 `modeling_pi0.py` 导出 `PI0Policy`
- 从 `processor_pi0.py` 导出 `make_pi0_pre_post_processors`
- 通过 `__all__` 声明这三个名字是稳定公共 API

### 为什么它重要

外部代码通常不会直接写：

```python
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
```

而是会写：

```python
from lerobot.policies.pi0 import PI0Policy
```

因此 `__init__.py` 决定了这个目录在包层面暴露给外部的“正式入口”。

---

## 5. 模块二：`configuration_pi0.py`

这个模块的本质是：把 OpenPI 的超参数和 LeRobot 的 policy 配置体系接起来。

### 5.1 常量：`DEFAULT_IMAGE_SIZE`

值是 `224`。

它定义了默认图像分辨率，和 PaliGemma 224 版模型对齐。后面的 `image_resolution` 默认就是 `(224, 224)`。

### 5.2 类：`PI0Config`

`PI0Config` 继承自 `PreTrainedConfig`，并通过：

```python
@PreTrainedConfig.register_subclass("pi0")
```

注册成 LeRobot 的一个 policy 类型。这使得外部可以通过工厂函数用 `"pi0"` 字符串创建它。

#### 关键字段按功能分组理解

##### 1. 基础模型结构

- `paligemma_variant="gemma_2b"`
- `action_expert_variant="gemma_300m"`
- `dtype="float32"`

默认设计是“大一点的视觉语言前缀 + 小一点的动作 expert”。

##### 2. 序列长度与动作 chunk

- `n_obs_steps=1`
- `chunk_size=50`
- `n_action_steps=50`

这表示：

- 当前实现默认只看 1 个观测时刻
- 一次预测 50 步动作
- 默认会把这 50 步全部作为本次 chunk 的可执行动作

##### 3. 状态/动作维度上限

- `max_state_dim=32`
- `max_action_dim=32`

这里不是“任意维度都能自动支持”的意思，而是：

- 如果真实维度更小，会 pad 到 32
- 如果真实维度更大，`pad_vector()` 不会帮你截断，后面的线性层会直接维度不匹配

所以这两个值本质上是模型内部固定维度，必须至少覆盖真实 state/action 维数。

##### 4. Flow matching 与时间采样

- `num_inference_steps=10`
- `time_sampling_beta_alpha=1.5`
- `time_sampling_beta_beta=1.0`
- `time_sampling_scale=0.999`
- `time_sampling_offset=0.001`
- `min_period=4e-3`
- `max_period=4.0`

这些参数分别控制：

- 推理时做多少次 denoise
- 训练时 `t` 怎么采样
- 时间 embedding 的频率范围

##### 5. 相对动作

- `use_relative_actions=False`
- `relative_exclude_joints=["gripper"]`
- `action_feature_names=None`

这三者是配套使用的：

- 若 `use_relative_actions=False`，则整个 relative/absolute 逻辑失效
- 若 `use_relative_actions=True`，则 preprocessor 会把动作转成相对状态偏移
- `relative_exclude_joints` 用来让某些维度保持绝对量
- 但这个排除机制依赖 `action_feature_names`

`action_feature_names` 默认是 `None`，通常由 `make_policy` 在读取数据集元信息时填充。如果用户手动启用相对动作却没有提供这些名字，那么“排除 gripper”这件事实际上不会生效，所有维度都会被视为相对动作维度。

##### 6. RTC

- `rtc_config: RTCConfig | None = None`

只要不传，`pi0` 就按普通 chunking 推理；一旦给出配置，就能在推理中接入 RTC。

##### 7. 图像与空相机

- `image_resolution=(224, 224)`
- `empty_cameras=0`

`empty_cameras` 的作用是：如果模型结构上期望更多相机，但当前环境没有全部提供，就在 feature 层面补出“空相机槽位”，后续再用 `_preprocess_images()` 生成全黑/全无效 mask 的输入。

##### 8. 归一化

- `VISUAL -> IDENTITY`
- `STATE -> MEAN_STD`
- `ACTION -> MEAN_STD`

这点很重要：图像不会在 processor 里做统计归一化，图像真正的数值映射是在 `_preprocess_images()` 里从 `[0, 1]` 转成 `[-1, 1]`。

##### 9. 训练工程参数

- `gradient_checkpointing`
- `compile_model`
- `compile_mode`
- `device`

它们不改变算法本身，但决定显存/速度权衡。

##### 10. 微调控制

- `freeze_vision_encoder`
- `train_expert_only`

分别表示：

- 只冻结视觉塔
- 直接冻结整个 PaliGemma，只训练 expert 与投影层

##### 11. 优化器与调度器预设

- `optimizer_*`
- `scheduler_*`

这些值被 `get_optimizer_preset()` 和 `get_scheduler_preset()` 读取，生成真正的配置对象。

##### 12. tokenizer 长度

- `tokenizer_max_length=48`

这与原始 OpenPI 的 PI0 默认 prompt 长度保持一致。

### 5.3 `PI0Config.__post_init__`

职责有两层：

1. 先调用父类 `PreTrainedConfig.__post_init__()` 自动选择/校验 device。
2. 再做 `pi0` 自己的合法性检查。

它检查：

- `n_action_steps <= chunk_size`
- `paligemma_variant` 只能是 `"gemma_300m"` 或 `"gemma_2b"`
- `action_expert_variant` 只能是这两者之一
- `dtype` 只能是 `"bfloat16"` 或 `"float32"`

这一步的作用是尽早失败，避免模型初始化到一半才因为配置非法崩掉。

### 5.4 `PI0Config.validate_features()`

这是 `pi0` 配置层里非常重要的方法。

它会在需要时自动补足三类 feature：

#### 1. 空相机 feature

对每个 `empty_camera_i`，注册一个：

- key: `observation.images.empty_camera_i`
- type: `VISUAL`
- shape: `(3, H, W)`

#### 2. 状态 feature

若 `observation.state` 不存在，则自动补：

- type: `STATE`
- shape: `(max_state_dim,)`

#### 3. 动作 feature

若 `action` 不存在，则自动补：

- type: `ACTION`
- shape: `(max_action_dim,)`

这意味着 `pi0` 对“输入/输出 schema 不完整”的情况是有兜底能力的，但它的兜底是按内部 pad 维度来的，不是按真实机器人维度推断出来的。

### 5.5 `PI0Config.get_optimizer_preset()`

返回一个 `AdamWConfig`，把配置中的：

- `lr`
- `betas`
- `eps`
- `weight_decay`
- `grad_clip_norm`

打包起来。它只是“预设生成器”，不直接创建 PyTorch 优化器。

### 5.6 `PI0Config.get_scheduler_preset()`

返回一个 `CosineDecayWithWarmupSchedulerConfig`，包括：

- `peak_lr`
- `decay_lr`
- `num_warmup_steps`
- `num_decay_steps`

这让 `pi0` 可以复用 LeRobot 统一的 scheduler 创建路径。

### 5.7 `PI0Config.observation_delta_indices`

返回 `None`。

含义是：当前 `pi0` 不使用“多个观测时间差分”的接口描述，它默认只看当前观测。

### 5.8 `PI0Config.action_delta_indices`

返回 `list(range(self.chunk_size))`。

含义是：policy 输出的是一个未来动作序列，时间索引覆盖：

```text
0, 1, 2, ..., chunk_size - 1
```

这和 action chunk 的语义保持一致。

### 5.9 `PI0Config.reward_delta_indices`

返回 `None`。

说明 `pi0` 不是 reward model，也不建模 reward 序列。

---

## 6. 模块三：`processor_pi0.py`

这个模块的职责不是“做模型运算”，而是把 LeRobot 规范 batch 变成 PI0 真正能消费的输入，并在输出后把动作恢复到可执行语义。

### 6.1 类：`Pi0NewLineProcessor`

这个 step 很小，但它是 PaliGemma 对接里一个容易忽略的关键点。

#### 模块意图

PaliGemma tokenizer 对 prompt 结尾换行比较敏感，因此这里强制要求 `task` 最终以 `\n` 结尾。

#### `Pi0NewLineProcessor.complementary_data()`

处理逻辑是：

1. 如果没有 `task`，直接原样返回。
2. 如果 `task is None`，原样返回。
3. 如果 `task` 是字符串，且不以 `\n` 结尾，就补一个。
4. 如果 `task` 是字符串列表，就逐个补。
5. 若 `task` 既不是字符串也不是字符串列表，则保持不变。

这个实现很克制，不会对非标准输入做过度假设。

#### `Pi0NewLineProcessor.transform_features()`

直接返回原特征定义。

因为补换行不会改变 feature shape/type，它只是修改 complementary data 里的字符串内容。

### 6.2 函数：`make_pi0_pre_post_processors()`

这是整个 `pi0` 与 LeRobot processor 框架的接缝点。

它返回两个 pipeline：

- preprocessor: `dict[str, Any] -> dict[str, Any]`
- postprocessor: `PolicyAction -> PolicyAction`

#### preprocessor 的步骤顺序

实际顺序是：

1. `RenameObservationsProcessorStep(rename_map={})`
2. `AddBatchDimensionProcessorStep()`
3. `Pi0NewLineProcessor()`
4. `TokenizerProcessorStep(...)`
5. `DeviceProcessorStep(device=config.device)`
6. `RelativeActionsProcessorStep(...)`
7. `NormalizerProcessorStep(...)`

#### 为什么是这个顺序

##### 1. `RenameObservationsProcessorStep`

这里的 `rename_map` 是空字典，等于 no-op。它存在的意义不是改 key，而是为了保持与预训练处理器结构的一致性。

##### 2. `AddBatchDimensionProcessorStep`

若输入是单条样本，它会把：

- 状态从 `(D,)` 变 `(1, D)`
- 图像从 `(C, H, W)` 变 `(1, C, H, W)`
- `task` 从字符串变为长度 1 的列表

这样后面的 tokenizer 和模型就都可以按 batch 形式工作。

##### 3. `Pi0NewLineProcessor`

保证 `task` 结尾有换行。

##### 4. `TokenizerProcessorStep`

使用固定 tokenizer：

```text
google/paligemma-3b-pt-224
```

并设置：

- `max_length = config.tokenizer_max_length`
- `padding_side = "right"`
- `padding = "max_length"`

它会往 observation 里加入：

- `observation.language.tokens`
- `observation.language.attention_mask`

##### 5. `DeviceProcessorStep`

把 transition 内所有张量搬到 `config.device`。

把这一步放在 relative/normalize 之前，可以保证后续数学运算发生在正确设备上。

##### 6. `RelativeActionsProcessorStep`

如果 `config.use_relative_actions=True`，它会把：

```text
action <- action - state
```

但只在“允许转成相对量”的维度上做。它还会缓存最后一次看到的 state，供 postprocessor 把动作再转回绝对量。

##### 7. `NormalizerProcessorStep`

根据 `dataset_stats` 和 `normalization_mapping` 对：

- `observation.state`
- `action`

做归一化。图像因为 `VISUAL -> IDENTITY`，在这里不会被改动。

#### postprocessor 的步骤顺序

顺序是：

1. `UnnormalizerProcessorStep`
2. `AbsoluteActionsProcessorStep`
3. `DeviceProcessorStep(device="cpu")`

#### 为什么要先反归一化再转绝对动作

因为如果使用 relative actions，模型输出的是“归一化后的相对动作”。正确恢复顺序必须是：

```text
模型输出
-> 先反归一化，回到真实相对位移量纲
-> 再加回 state，得到绝对动作
```

否则数值会错。

#### `relative_step` 共享引用的意义

`make_pi0_pre_post_processors()` 里创建了一个 `relative_step`，同时把它：

- 放进 preprocessor
- 作为参数传给 postprocessor 里的 `AbsoluteActionsProcessorStep`

这样 postprocessor 才能读取 preprocessor 缓存的 `_last_state`。这也是为什么 LeRobot 在从磁盘反序列化 processor 后，还需要在 `factory.py` 里手动重新连接一次这两个 step。

---

## 7. 模块四：`modeling_pi0.py` 总览

这是 `pi0` 的核心模块。它可以再分成四层：

1. 顶层工具函数
2. `PaliGemmaWithExpertModel`
3. `PI0Pytorch`
4. `PI0Policy`

其中：

- `PaliGemmaWithExpertModel` 解决“联合前缀/后缀 Transformer 怎么跑”
- `PI0Pytorch` 解决“flow matching 怎么训练、怎么采样”
- `PI0Policy` 解决“如何适配 LeRobot 的 policy 接口、processor、action queue、权重加载”

下面分层讲。

---

## 8. `modeling_pi0.py` 顶层工具函数详解

### 8.1 `ActionSelectKwargs`

这是一个 `TypedDict`，定义推理时额外允许传入的关键字参数：

- `inference_delay`
- `prev_chunk_left_over`
- `execution_horizon`

它们都与 RTC 相关。

### 8.2 `get_safe_dtype(target_dtype, device_type)`

作用是“在不同设备上选择安全 dtype”。

实现逻辑：

- 若设备是 `mps` 且目标 dtype 为 `float64`，降为 `float32`
- 若设备是 `cpu` 且目标 dtype 为 `bfloat16`，改成 `float32`
- 若设备是 `cpu` 且目标 dtype 为 `float64`，保留 `float64`
- 其他情况直接返回原 dtype

它主要服务于时间 embedding 这类对数值稳定性更敏感、又需要兼容不同后端的函数。

### 8.3 `create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu")`

作用是为标量时间 `t` 构造正弦/余弦位置编码。

关键细节：

- `dimension` 必须是偶数
- `time` 必须是一维张量，形状 `(batch_size,)`
- 频率不是线性采样，而是在 `min_period` 到 `max_period` 间按指数尺度展开
- 输出形状是 `(batch_size, dimension)`

PI0 在 `embed_suffix()` 里用它对 denoise 时间步做编码，然后与动作 embedding 融合。

### 8.4 `sample_beta(alpha, beta, bsize, device)`

作用是从 Beta 分布采样时间 `t` 的基础随机数。

关键细节：

- 先在 CPU 上用 `torch.distributions.Beta` 采样
- 再搬到目标 device

这么写的原因是注释里说明：MPS 后端对底层 `_sample_dirichlet` 支持不完整。

### 8.5 `make_att_2d_masks(pad_masks, att_masks)`

这是整份实现里最值得认真理解的 mask 构造函数之一。

输入：

- `pad_masks`: 哪些 token 真存在，哪些是 padding
- `att_masks`: 用来编码“块级注意力结构”的 0/1 序列

输出：

- 二维布尔 mask，形状约为 `(B, N, N)`

核心步骤：

1. 对 `att_masks` 做 `cumsum`
2. 用 `<=` 比较构造块级可见性
3. 再与 `pad_masks` 结合，屏蔽 padding token

它不是标准 causal mask，而是一个更灵活的“块递进注意力”机制。

### 8.6 `pad_vector(vector, new_dim)`

作用是把最后一维 pad 到 `new_dim`，不足的部分补 0。

支持：

- `(B, D)`
- `(B, T, D)`

若 `vector.shape[-1] >= new_dim`，直接原样返回。

这也是为什么 `max_state_dim/max_action_dim` 必须足够大；代码不会自动截断高维输入。

### 8.7 `resize_with_pad_torch(images, height, width, mode="bilinear")`

作用是按保持纵横比的方式 resize，并用黑边 pad 到目标大小。

关键细节：

- 同时支持 channels-last 和 channels-first
- 若是单张图像，会自动补 batch 维
- `ratio = max(cur_width / width, cur_height / height)`，因此不会拉伸变形
- `uint8` 图像会 round/clamp 到 `[0, 255]`
- `float32` 图像会 clamp 到 `[0.0, 1.0]`
- 输出格式与输入格式保持一致

这个函数是 `_preprocess_images()` 的底层工具。

### 8.8 `compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert)`

这是整份 `pi0` 实现里最核心的函数之一。

它的真实职责是：

> 在“前缀分支 + 后缀分支联合训练”的情况下，手工执行第 `layer_idx` 层的完整跨分支注意力与各自后处理。

可以按步骤理解：

#### 第一步：分别对前缀和后缀做输入层归一化

对每个分支：

- 取该层 `input_layernorm`
- 调用 `layernorm_forward()`
- 得到归一化后的 hidden states 与 residual gate

#### 第二步：分别投影出 Q/K/V

对前缀与后缀各自的 hidden states：

- `q_proj`
- `k_proj`
- `v_proj`

再 reshape 成多头注意力需要的形状。

#### 第三步：把两边的 Q/K/V 沿序列维拼起来

这是关键点。拼起来之后，前缀 token 和后缀 token 会在同一个 attention 里交互，而不是各跑各的。

#### 第四步：共享 rotary embedding 与 attention

代码用的是 PaliGemma 语言模型那边的：

- `rotary_emb`
- `eager_attention_forward`

也就是说，两条分支共享一套 attention 计算逻辑。

#### 第五步：再把 attention 输出按原来的前缀/后缀长度切回去

切回去后：

- 前缀部分送回前缀分支自己的 `o_proj`
- 后缀部分送回后缀分支自己的 `o_proj`

#### 第六步：各自完成 residual + post_attention_layernorm + MLP + residual

也就是说：

- attention 是联合的
- MLP 是各自独立的

这正是“共享上下文、保留各自专长”的实现方式。

#### 一个很重要的隐藏前提

代码里：

```python
att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
```

这里直接把多头输出 reshape 为 `8 * head_dim`，隐含依赖当前实现的 attention head 数固定为 `8`。这与 `get_gemma_config()` 返回的两个 variant 一致，所以当前实现是自洽的，但它也意味着这段逻辑不是任意 Gemma 配置都能无缝复用。

### 8.9 `GemmaConfig`

这是一个非常轻量的配置容器，字段包括：

- `width`
- `depth`
- `mlp_dim`
- `num_heads`
- `num_kv_heads`
- `head_dim`

它不是 Hugging Face 的 config，而是给本地逻辑先做一层简化表达。

它的 `__init__()` 没有额外逻辑，就是把这几个参数逐字段保存下来，供 `get_gemma_config()` 和后续 HF config 映射使用。

### 8.10 `get_gemma_config(variant)`

根据字符串返回 `GemmaConfig`。

支持两个 variant：

#### `gemma_300m`

- `width = 1024`
- `depth = 18`
- `mlp_dim = 4096`
- `num_heads = 8`
- `num_kv_heads = 1`
- `head_dim = 256`

#### `gemma_2b`

- `width = 2048`
- `depth = 18`
- `mlp_dim = 16384`
- `num_heads = 8`
- `num_kv_heads = 1`
- `head_dim = 256`

一个很有意思的实现点是：两个 variant 的 `num_heads` 和 `head_dim` 一样，因此它们在联合 attention 时接口对齐，只是 hidden size 和 MLP 宽度不同。

---

## 9. 类：`PaliGemmaWithExpertModel`

这个类是“联合前缀分支与后缀 expert 分支”的桥梁。

### 9.1 类的职责

它并不是完整的 PI0 policy，而是一个更底层的“联合 Transformer 骨架”：

- 前缀分支：`PaliGemmaForConditionalGenerationWithPiGemma`
- 后缀分支：`PiGemmaForCausalLM`

### 9.2 `__init__(...)`

这个构造函数完成了五件大事。

#### 1. 根据本地 `GemmaConfig` 拼出 Hugging Face 配置对象

它会把：

- `vlm_config`
- `action_expert_config`

分别映射成：

- `CONFIG_MAPPING["paligemma"]()`
- `CONFIG_MAPPING["gemma"](...)`

也就是说，前面的轻量配置最终还是要落到 HF 的模型类上。

#### 2. 把 PaliGemma 与 expert 的 hidden size/head/层数对齐到本地设定

包括：

- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `head_dim`
- `num_hidden_layers`
- `num_key_value_heads`

#### 3. 构造真正的模型对象

```python
self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(...)
self.gemma_expert = PiGemmaForCausalLM(...)
```

这里用的不是原始 HF 模型，而是 `pi_gemma.py` 里的自定义版本。这样才能支持 gated residual 和 AdaRMS 接口。

#### 4. 关闭 expert 的 token embedding

```python
self.gemma_expert.model.embed_tokens = None
```

这很关键，说明 expert 不是靠 token id 驱动，而是完全依赖外部传入的 `inputs_embeds`，也就是状态/动作/时间融合后的 suffix embeddings。

#### 5. 处理精度与冻结策略

构造完成后会调用：

- `to_bfloat16_for_selected_params()`
- `_set_requires_grad()`

### 9.3 `to_bfloat16_for_selected_params(precision)`

作用是根据 `precision` 把参数转成对应 dtype，但会保留一部分模块为 `float32`。

若 `precision == "bfloat16"`：

- 全模型先转 `bfloat16`
- 再把以下路径转回 `float32`
  - `vision_tower`
  - `multi_modal_projector`
  - `input_layernorm`
  - `post_attention_layernorm`
  - `model.norm`

注释说明这么做是为了避免 vision 路径在 dtype 来回切换，引发优化器问题，并与 PI0.5 的做法保持一致。

### 9.4 `_set_requires_grad()`

负责冻结策略：

- 若 `freeze_vision_encoder=True`，冻结视觉塔参数，并把 vision tower 设为 eval
- 若 `train_expert_only=True`，直接冻结整个 `paligemma`

这两个开关的粒度不同：

- 前者只冻视觉编码器
- 后者把整个视觉语言模型都冻住

### 9.5 `train(mode=True)`

重写 `train()` 的目的是：即使外部把整个模型切回 train mode，也要保持被冻结的部分继续待在 eval mode。

这能避免：

- frozen 模块误进入 dropout/bn 的训练行为
- 与冻结意图不一致

### 9.6 `embed_image(image)`

图像编码流程：

1. 记录输入 dtype
2. 若不是 `float32`，先转 `float32`
3. 用 `self.paligemma.model.get_image_features(image)` 提取视觉特征
4. 取 `pooler_output`
5. 乘上 `sqrt(hidden_size)` 做尺度调整
6. 若需要，再转回原始 dtype

这里视觉路径之所以强制先转 `float32`，和前面“视觉塔保留 float32”是一致的。

### 9.7 `embed_language_tokens(tokens)`

直接调用：

```python
self.paligemma.model.language_model.embed_tokens(tokens)
```

这里只做 token embedding，本身不做 transformer 编码。

### 9.8 `forward(...)`

这是 `PaliGemmaWithExpertModel` 最重要的方法。它有三种运行模式。

它的统一返回格式是：

- `([prefix_output, suffix_output], prefix_past_key_values)`

也就是说：

- 第一个返回值永远是“前缀输出和后缀输出”的二元列表
- 第二个返回值只有在 prefix-only 且 `use_cache=True` 的情况下才真正携带可复用的 `past_key_values`

#### 模式 1：只有前缀，`inputs_embeds = [prefix, None]`

这发生在推理时的 prefix prefill 阶段。

行为是：

- 只跑 `paligemma.model.language_model.forward(...)`
- 返回前缀输出和 `past_key_values`

这一步的目标是缓存视觉语言前缀的 KV，后面每个 denoise step 不需要重复算一遍。

#### 模式 2：只有后缀，`inputs_embeds = [None, suffix]`

这发生在推理时的单步 denoise。

行为是：

- 只跑 `gemma_expert.model.forward(...)`
- 利用传入的 `past_key_values`
- 返回后缀输出

#### 模式 3：前缀和后缀同时存在，`inputs_embeds = [prefix, suffix]`

这发生在训练时。

行为最复杂：

1. 遍历每一层 `layer_idx`
2. 若开启 gradient checkpointing，则用 checkpoint 包裹
3. 每层调用 `compute_layer_complete(...)`
4. 所有层结束后，再对两个分支分别做 final norm

也就是说：

- 训练时不是直接调用 HF 模型的标准 forward
- 而是显式地逐层手搓“联合 attention + 分支 MLP”

这正是 `pi0` 与普通 PaliGemma/Gemma 拼接方案最不同的地方。

---

## 10. 类：`PI0Pytorch`

这是“算法本体”所在的类。它定义了训练损失、采样器、prefix/suffix embedding、RTC 接口。

### 10.1 `__init__(config, rtc_processor=None)`

这个构造函数做了下面几件事。

#### 1. 保存配置与 RTC 处理器

`self.config` 与 `self.rtc_processor` 都在这里绑定。

#### 2. 根据配置选定 Gemma variant

通过 `get_gemma_config()` 分别得到：

- `paligemma_config`
- `action_expert_config`

#### 3. 检查图像分辨率必须是正方形

若 `image_resolution[0] != image_resolution[1]`，直接报错。

这是因为当前 PaliGemma 路径假设输入是正方形图像。

#### 4. 构建联合前缀/后缀模型

```python
self.paligemma_with_expert = PaliGemmaWithExpertModel(...)
```

这里固定：

- `use_adarms=[False, False]`

说明 `pi0` 当前并不启用 AdaRMS；AdaRMS 主要是 `pi05` 的区别点。

#### 5. 构建状态/动作/时间投影层

- `action_in_proj`: `max_action_dim -> action_expert_width`
- `action_out_proj`: `action_expert_width -> max_action_dim`
- `state_proj`: `max_state_dim -> action_expert_width`
- `action_time_mlp_in`
- `action_time_mlp_out`

这里的设计含义很清楚：

- 状态和动作都会先投影到 action expert 的 hidden space
- 时间 embedding 也在同一维度空间里和动作融合

#### 6. 记录 gradient checkpointing 开关

`self.gradient_checkpointing_enabled = False`

#### 7. 可选 `torch.compile`

若 `config.compile_model=True`，会：

- `torch.set_float32_matmul_precision("high")`
- 编译 `sample_actions`
- 编译 `forward`

注意它不是编译整个 module，而是直接替换这两个 bound method。

### 10.2 `gradient_checkpointing_enable()`

作用是：

- 打开本类自己的 `gradient_checkpointing_enabled`
- 同时把三个子模块的 checkpointing 开关也打开
  - `paligemma.language_model`
  - `paligemma.vision_tower`
  - `gemma_expert.model`

这意味着 `pi0` 的 checkpointing 不是只在一层生效，而是同时覆盖：

- 嵌入/投影级的 `_apply_checkpoint`
- 联合 transformer 层级的 checkpoint
- 子模块内部的 checkpoint 逻辑

### 10.3 `gradient_checkpointing_disable()`

与上一个方法对称，负责关闭这些开关。

### 10.4 `_rtc_enabled()`

返回：

```python
config.rtc_config is not None and config.rtc_config.enabled
```

它只是在本类内部统一判断 RTC 是否启用。

### 10.5 `_apply_checkpoint(func, *args, **kwargs)`

这是一个包装器：

- 若开启 checkpointing 且当前是 training mode，就用 `torch.utils.checkpoint.checkpoint`
- 否则直接调用函数

这让 `embed_prefix()`、`embed_suffix()`、输出投影等小块逻辑也能吃到 checkpointing 的收益。

### 10.6 `_prepare_attention_masks_4d(att_2d_masks)`

把二维布尔 mask 转成 transformer attention 常用的四维 additive mask：

- `True -> 0.0`
- `False -> OPENPI_ATTENTION_MASK_VALUE`

这里的 `OPENPI_ATTENTION_MASK_VALUE` 是一个很大的负数，起到“几乎负无穷”的作用。

### 10.7 `sample_noise(shape, device)`

直接从标准高斯分布采样，dtype 固定为 `float32`。

PI0 的 denoise 是从这份噪声开始的。

### 10.8 `sample_time(bsize, device)`

作用是为一个 batch 采样时间 `t`。

流程：

1. 先用 `sample_beta()` 从 Beta 分布采样
2. 再做：

```text
time = beta_sample * scale + offset
```

默认参数下，`t` 会落在 `(0.001, 1.0)` 附近，而不会真的取到 0。

### 10.9 `embed_prefix(images, img_masks, lang_tokens, lang_masks)`

作用是生成前缀 token 序列及其 mask。

#### 图像部分

对每个 camera：

1. 调 `embed_image(img)`
2. 得到 `img_emb`，形状大致是 `(B, num_img_tokens, D)`
3. 用 `img_mask` 扩成 `(B, num_img_tokens)` 的 pad mask
4. attention block mask 全置 0

#### 语言部分

1. 调 `embed_language_tokens(lang_tokens)`
2. 再乘 `sqrt(lang_emb_dim)` 做尺度缩放
3. pad mask 使用 tokenizer 的 `attention_mask`
4. attention block mask 也全置 0

#### 最终输出

返回：

- `embs`
- `pad_masks`
- `att_masks`

其中 `att_masks` 全为 0，表示前缀内部处于同一个 attention block。

### 10.10 `embed_suffix(state, noisy_actions, timestep)`

作用是把状态、带噪动作、时间编码成后缀 token。

#### 状态 token

1. 若 `state_proj` 权重是 `float32`，先把 state 转成 `float32`
2. 过 `state_proj`
3. 加一个长度为 1 的 token 维
4. 为它创建：
   - `pad_mask = 1`
   - `att_mask = 1`

#### 时间 embedding

调用 `create_sinusoidal_pos_embedding(...)` 得到 `(B, D)` 的时间编码。

#### 动作 token

1. `noisy_actions` 过 `action_in_proj`
2. 把时间 embedding expand 到每个 action token
3. 与动作 embedding 在最后一维拼接
4. 过两层 MLP：
   - `action_time_mlp_in`
   - `SiLU`
   - `action_time_mlp_out`

#### `adarms_cond`

这里直接设成 `None`，说明 `pi0` 当前不使用 AdaRMS 条件输入。

#### suffix 的 block mask

这里最关键的代码是：

```python
att_masks += [1] + ([0] * (self.config.chunk_size - 1))
```

结合前面状态 token 的 `[1]`，最终形成：

- 状态 token：开一个新 block
- 第一个动作 token：再开一个新 block
- 其余动作 token：与第一个动作 token 同 block

这正对应前面说的块级注意力结构。

### 10.11 `forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None)`

这是训练前向。

按执行顺序理解：

#### 1. 若没有提供 `noise/time`，先采样

- `noise`: 高斯噪声
- `time`: Beta 时间

#### 2. 构造 flow matching 轨迹

```python
x_t = time * noise + (1 - time) * actions
u_t = noise - actions
```

#### 3. 生成前缀与后缀 embedding

- `embed_prefix(...)`
- `embed_suffix(...)`

#### 4. 若底层 attention 用的是 `bfloat16`，把 prefix/suffix embedding 也转到 `bfloat16`

这是为了 dtype 对齐。

#### 5. 构造联合 pad mask 和 block mask

然后通过：

- `make_att_2d_masks(...)`
- `torch.cumsum(pad_masks, dim=1) - 1`

生成真正给 transformer 用的：

- `attention_mask`
- `position_ids`

#### 6. 走联合模型前向，只取 suffix 输出

前缀部分只是提供条件，真正用于动作预测的是后缀最后 `chunk_size` 个 token。

#### 7. 输出投影

`suffix_out -> action_out_proj -> v_t`

输出维度回到 `max_action_dim`。

#### 8. 损失

返回：

```python
F.mse_loss(u_t, v_t, reduction="none")
```

这里故意不在底层做 reduce，是为了上层 policy 可以选择：

- 返回整体均值
- 返回逐样本 loss

### 10.12 `sample_actions(...)`

这是推理阶段的核心采样器。

#### 第一步：确定采样步数和初始噪声

若没传 `num_steps`，用配置里的 `num_inference_steps`。

若没传 `noise`，会采样形状为：

```text
(B, chunk_size, max_action_dim)
```

的高斯噪声。

#### 第二步：prefix 只跑一次并缓存

它会：

1. `embed_prefix(...)`
2. 构 prefix attention mask
3. 强制 `paligemma.language_model.config._attn_implementation = "eager"`
4. 调 `self.paligemma_with_expert.forward(inputs_embeds=[prefix_embs, None], use_cache=True)`

得到 `past_key_values`

这一步相当于 prefix prefill。

#### 第三步：初始化反积分

```python
dt = -1.0 / num_steps
x_t = noise
```

#### 第四步：循环 denoise

每一步：

1. 当前时间 `time = 1 + step * dt`
2. 构造 `time_tensor`
3. 包一层 `denoise_step_partial_call`
4. 若 RTC 开启，则把这个 denoise callable 交给 `rtc_processor.denoise_step(...)`
5. 否则直接跑普通 denoise
6. 用 `x_t = x_t + dt * v_t` 更新轨迹

#### 第五步：调试跟踪

若 RTC tracker 打开，会记录每一步的：

- `time`
- `x_t`
- `v_t`

#### 返回值

最终返回 `x_t`，也就是逼近 `x_0` 的动作 chunk。

### 10.13 `denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)`

这是推理时的单步 velocity 预测器。

它和训练 forward 的区别在于：prefix 不再重算，只利用缓存。

#### 1. 重新生成 suffix embedding

因为每一步 `x_t` 与 `timestep` 都变了，所以 suffix 必须重算。

#### 2. 构造 suffix query 对 prefix 的可见性

```python
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
```

这意味着：每个 suffix token 都能看所有有效 prefix token。

#### 3. suffix 内部仍使用块级 mask

通过 `make_att_2d_masks(suffix_pad_masks, suffix_att_masks)` 构造。

#### 4. 拼出完整 attention mask

把“suffix 看 prefix”与“suffix 看 suffix”两部分拼起来。

#### 5. 位置编码偏移

```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
```

它保证 suffix token 的 position id 接在 prefix 后面。

#### 6. 深拷贝 `past_key_values`

这是一个很细但很重要的实现点。因为 cache 对象可能在 forward 过程中被修改，所以这里每个 denoise step 都 `deepcopy` 一次 prefix cache，避免不同 step 互相污染。

#### 7. 只跑后缀分支并输出 velocity

最后仍然是：

```python
suffix_out -> action_out_proj
```

得到当前时间步的 `v_t`。

---

## 11. 类：`PI0Policy`

`PI0Policy` 是真正暴露给 LeRobot 训练/部署框架的 policy 类。它把 `PI0Pytorch` 包装成 `PreTrainedPolicy` 接口。

### 11.1 类属性

- `config_class = PI0Config`
- `name = "pi0"`

这让 LeRobot 的工厂系统知道：

- 这个 policy 用什么配置类
- 它在注册表中的名字是什么

### 11.2 `__init__(config, **kwargs)`

执行顺序如下：

1. 调父类构造函数
2. `config.validate_features()`
3. 保存 `config`
4. `init_rtc_processor()`
5. 构建 `self.model = PI0Pytorch(...)`
6. 若配置要求 gradient checkpointing，则开启
7. `self.model.to(config.device)`
8. `self.reset()`

这里有两个重要点：

#### 1. `validate_features()` 在模型构建前执行

这保证：

- 后面任何依赖 `input_features/output_features` 的逻辑
- 都能看到至少有 state/action 这些基本 feature

#### 2. action queue 在构造末尾清空

因此每个新 policy 实例默认从干净状态开始 rollout。

### 11.3 `from_pretrained(...)`

`PI0Policy` 自己重写了 `from_pretrained()`，没有直接复用基类的 safetensors 加载路径。它的逻辑可以概括成：

#### 1. 打印免责声明

说明这是 OpenPI 的直接移植实现。

#### 2. 若没传 `config`，先从预训练路径加载 config

#### 3. 先实例化一个“空权重” policy

```python
model = cls(config, **kwargs)
```

#### 4. 尝试用 `cached_file(..., "model.safetensors")` 读取权重

若失败，它会打印信息并直接返回当前 model，不再抛异常。

这意味着：如果权重没加载成功，调用方可能得到一个随机初始化模型。这一点在使用时必须心里有数。

#### 5. 修正 state dict key

通过 `_fix_pytorch_state_dict_keys(...)` 做兼容性处理。

#### 6. 补齐 `model.` 前缀

因为 `PI0Policy` 的核心子模块挂在 `self.model` 下面，而有些 checkpoint key 可能没这个前缀。

#### 7. `load_state_dict(...)`

根据 `strict` 参数加载，并打印：

- missing keys
- unexpected keys

#### 一个细节

这个重写版 `from_pretrained()` 最后没有像基类那样显式 `eval()`。不过 `select_action()` 和 `predict_action_chunk()` 内部都会先 `self.eval()`，所以常规推理不会受影响。

### 11.4 `_fix_pytorch_state_dict_keys(state_dict, model_config)`

这个函数的作用是兼容旧 checkpoint 或 OpenPI/LeRobot 结构差异。

它做了几类 key 修正：

#### 1. AdaRMS 相关 norm key 的跳过

如果 checkpoint 提供的是普通 norm 权重，但当前 expert 开启了 AdaRMS，就跳过这些 key，避免结构不匹配。

#### 2. `time_mlp_* -> action_time_mlp_*`

说明：

- 旧 checkpoint 可能沿用了别的命名
- 当前 `pi0` 使用的是 `action_time_mlp_in/out`

#### 3. 对 `patch_embedding` 给出警告

这里只是 warning，没有自动修复。

#### 4. 把 `lm_head.weight` 克隆到 `embed_tokens.weight`

这是一种与旧权重格式兼容的映射方式。

总之，这个函数不是“完整迁移器”，而是一个面向已知差异的 key 修补器。

### 11.5 `get_optim_params()`

直接返回 `self.parameters()`。

说明 `pi0` 没有再对参数组做更细粒度拆分，优化器外部若需要特殊参数组，需要另行处理。

### 11.6 `reset()`

重置内部状态：

- `_action_queue`
- `_queues[ACTION]`

其中真正被 `select_action()` 使用的是 `_action_queue`。

### 11.7 `init_rtc_processor()`

作用是根据 `config.rtc_config` 决定是否创建 `RTCProcessor`。

若创建成功，还会把它同步挂到 `self.model.rtc_processor` 上。

### 11.8 `_rtc_enabled()`

与 `PI0Pytorch._rtc_enabled()` 语义一致，只是 policy 层自己的便捷方法。

### 11.9 `_preprocess_images(batch)`

这是 `PI0Policy` 里非常关键的一个方法，因为 processor 层并没有真正把图像整理成 PaliGemma 想要的数值分布。

可以分步骤理解：

#### 1. 找出 present/missing image keys

基于 `self.config.image_features`：

- 在 batch 中存在的算 present
- 不存在的算 missing

若一个都没有，直接报错。

#### 2. 对每个 present image 做统一预处理

具体步骤：

1. 搬到和模型一致的 device
2. 保证 `float32`
3. 判断是 `[B, C, H, W]` 还是 `[B, H, W, C]`
4. 若是 CHW，先转成 HWC
5. 若分辨率不是 `config.image_resolution`，则 `resize_with_pad_torch(...)`
6. 从 `[0, 1]` 映射到 `[-1, 1]`
7. 若原来是 CHW，再转回 CHW
8. 记录 mask 为全 1

这里最核心的结论是：

> 图像真正的数值规范化不在 processor，而在这里完成。

#### 3. 对 missing image 构造空输入

对每个缺失相机：

- 构造一个全 `-1` 的图像
- 构造一个全 0 的 mask

为什么是 `-1`？

因为图像已经被映射到了 `[-1, 1]`，`-1` 对应“全黑/最小值”。

#### 4. 返回值

返回两个列表：

- `images`
- `img_masks`

这和 `embed_prefix()` 的入参直接对齐。

### 11.10 `prepare_state(batch)`

调用 `pad_vector(batch[OBS_STATE], self.config.max_state_dim)`。

作用是把状态向量 pad 到模型固定输入维度。

### 11.11 `prepare_action(batch)`

与 `prepare_state()` 对称，把动作 pad 到 `max_action_dim`。

### 11.12 `select_action(batch)`

作用是：给环境返回“当前该执行的一步动作”。

逻辑：

1. 断言 RTC 未启用
2. 切到 eval mode
3. 如果 `_action_queue` 为空，则先调用 `predict_action_chunk(batch)`
4. 只取前 `n_action_steps` 步，并压进队列
5. `popleft()` 返回当前要执行的单步动作

也就是说：

- 真正预测 chunk 的是 `predict_action_chunk()`
- `select_action()` 只是一个带缓存的“一步一步吐动作”的包装器

### 11.13 `predict_action_chunk(batch, **kwargs)`

作用是：给定当前观测，预测整段未来动作。

步骤如下：

1. 切到 eval mode
2. `_preprocess_images(batch)`
3. 从 batch 里取：
   - `observation.language.tokens`
   - `observation.language.attention_mask`
4. `prepare_state(batch)`
5. 调 `self.model.sample_actions(...)`
6. 把动作从 `max_action_dim` 截回真实 `output_features[ACTION].shape[0]`

这个“先 pad，后截回”的设计，使得模型内部维度可以固定为 32，而外部机器人只看到真实动作维度。

### 11.14 `forward(batch, reduction="mean")`

这是训练接口。

步骤：

1. 图像预处理
2. 取 language tokens/mask
3. `prepare_state`
4. `prepare_action`
5. 调 `self.model.forward(...)` 得到逐元素 loss
6. 截掉动作 pad 维
7. 构建 `loss_dict`

#### `loss_dict` 中有什么

- `loss_per_dim`: 在 batch 和时间维上平均后的每个动作维度损失
- `loss`: 总损失

#### `reduction="none"` 的意义

当 `reduction="none"` 时，返回的是每个样本的 loss，形状 `(B,)`。注释说明这是为 RA-BC weighting 之类的场景准备的。

### 11.15 `_get_default_peft_targets()`

返回一个 PEFT 默认目标模块配置，核心是一个正则：

- expert 里的 `self_attn.(q|v)_proj`
- 以及：
  - `state_proj`
  - `action_in_proj`
  - `action_out_proj`
  - `action_time_mlp_in`
  - `action_time_mlp_out`

这反映了作者对“默认最值得微调哪些层”的判断：

- 优先动 expert 的关键注意力投影
- 再动状态/动作的输入输出适配层

---

## 12. 你最该重点记住的几个实现细节

### 12.1 `pi0` 的 processor 和模型分工非常明确

- processor 负责：
  - task 末尾换行
  - tokenization
  - device 搬运
  - relative/absolute actions
  - state/action 归一化
- 模型负责：
  - 图像 resize/pad
  - 图像 `[0,1] -> [-1,1]`
  - state/action pad 到固定维度
  - 联合 Transformer
  - flow matching

### 12.2 `pi0` 内部固定维度是 32，不代表你的机器人就必须是 32 维

真实维度可以更小，代码会 pad。

但真实维度不能大于：

- `max_state_dim`
- `max_action_dim`

否则线性层维度会不匹配。

### 12.3 图像 feature 的归一化模式是 `IDENTITY`，这不是遗漏

这是有意设计。图像不是按数据集统计量做 normalize，而是按 PaliGemma/SigLIP 期望值域直接映射到 `[-1, 1]`。

### 12.4 相对动作不会改变模型结构

`use_relative_actions` 只改变 processor 链：

- preprocessor 里把 action 改成相对量
- postprocessor 里再恢复成绝对量

模型本体并不知道“自己在预测相对动作还是绝对动作”。

### 12.5 `pi0` 与 `pi05` 的一个关键区别，是这里完全没启用 AdaRMS

虽然底层 `pi_gemma.py` 支持 AdaRMS，但 `PI0Pytorch` 构造 `PaliGemmaWithExpertModel` 时固定传的是：

```python
use_adarms=[False, False]
```

所以当前 `pi0` 版本使用的是普通 gated residual + RMSNorm 路径。

### 12.6 推理速度的关键优化点在 prefix cache

每个 denoise step 都重新算 prefix 会很贵，所以当前实现先把视觉语言前缀做一次 `use_cache=True` 的 prefill，后面只重复计算 suffix。

这也是 `sample_actions()` 和 `denoise_step()` 被拆开的根本原因。

### 12.7 RTC 只适用于 `predict_action_chunk()`，不适用于 `select_action()`

代码里明确写了：

```python
assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"
```

原因很自然：RTC 的语义是“跨 chunk 融合”，而 `select_action()` 是单步队列消费接口。

### 12.8 有些 helper 函数明确标注为 OpenPI exact copy

例如：

- `create_sinusoidal_pos_embedding`
- `sample_beta`
- `make_att_2d_masks`
- `resize_with_pad_torch`

这说明当前实现的目标不是“重新发明一个 PI0”，而是尽量保留与 OpenPI 的数值与结构兼容性。

---

## 13. 一句话总结每个模块

为了最后再压缩一遍记忆负担，可以把四个模块记成下面这四句话：

- `__init__.py`: 把 `PI0Config`、`PI0Policy`、processor 工厂暴露给外部。
- `configuration_pi0.py`: 定义 `pi0` 的结构、维度、训练超参数、归一化约定与 feature 兜底规则。
- `processor_pi0.py`: 把 LeRobot batch 变成 `pi0` 真正可消费的输入，并把输出动作恢复到可执行语义。
- `modeling_pi0.py`: 用“PaliGemma 前缀 + Gemma expert 后缀 + flow matching + 反积分采样 + 可选 RTC”实现完整的 PI0 算法。

如果再进一步压缩成一句最核心的话，那么就是：

> LeRobot 里的 `pi0`，本质上是一个把视觉语言前缀缓存起来、再用联合 Transformer 对“状态 + 带噪动作 chunk”做 flow matching 去噪的 action chunk policy。
