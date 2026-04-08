#!/usr/bin/env python3

from __future__ import annotations

import copy
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from safetensors import safe_open
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

REQUIRED_POLICY_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)

PALIGEMMA_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
PALIGEMMA_TOKENIZER_DIRNAME = "paligemma_tokenizer"
PALIGEMMA_TOKENIZER_REQUIRED_CONFIG = "tokenizer_config.json"
PALIGEMMA_TOKENIZER_PRIMARY_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
)


def _is_valid_local_tokenizer_dir(tokenizer_dir: Path) -> bool:
    if not tokenizer_dir.is_dir():
        return False

    has_config = (tokenizer_dir / PALIGEMMA_TOKENIZER_REQUIRED_CONFIG).exists()
    has_primary_file = any((tokenizer_dir / name).exists() for name in PALIGEMMA_TOKENIZER_PRIMARY_FILES)
    return has_config and has_primary_file


def resolve_model_checkpoint(model_id_or_path: str, ckpt_dir: Path) -> Path:
    """
    Resolve a policy checkpoint to a local directory.

    - If `model_id_or_path` already exists on disk, use it directly.
    - Otherwise, download/sync the Hugging Face model repo into `ckpt_dir`
      and return that directory for local loading.
    """
    local_path = Path(model_id_or_path).expanduser()
    if local_path.exists():
        resolved = local_path.resolve()
        print(f"Using local checkpoint path: {resolved}")
        return resolved

    ckpt_dir = ckpt_dir.expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    missing_files = [name for name in REQUIRED_POLICY_FILES if not (ckpt_dir / name).exists()]
    if missing_files:
        print(f"Downloading {model_id_or_path} to {ckpt_dir}")
        snapshot_download(
            repo_id=model_id_or_path,
            repo_type="model",
            local_dir=ckpt_dir,
        )
    else:
        print(f"Using existing checkpoint files in {ckpt_dir}")

    return ckpt_dir


def resolve_policy_dtype(device: torch.device, requested_dtype: str | None) -> str:
    if requested_dtype is not None:
        return requested_dtype
    return "bfloat16" if device.type == "cuda" else "float32"


def resolve_paligemma_tokenizer_path(
    ckpt_dir: Path,
    tokenizer_path: str | Path | None = None,
    tokenizer_repo_id: str = PALIGEMMA_TOKENIZER_REPO,
) -> Path:
    if tokenizer_path is not None:
        explicit_path = Path(tokenizer_path).expanduser().resolve()
        if _is_valid_local_tokenizer_dir(explicit_path):
            print(f"Using explicit tokenizer path: {explicit_path}", flush=True)
            return explicit_path
        local_tokenizer_dir = explicit_path
        if explicit_path.exists():
            print(
                f"Tokenizer path exists but is incomplete, will try to download tokenizer files into: {local_tokenizer_dir}",
                flush=True,
            )
        else:
            print(
                f"Tokenizer path does not exist yet, will try to download tokenizer into: {local_tokenizer_dir}",
                flush=True,
            )
    else:
        local_tokenizer_dir = ckpt_dir.expanduser().resolve() / PALIGEMMA_TOKENIZER_DIRNAME

    if _is_valid_local_tokenizer_dir(local_tokenizer_dir):
        print(f"Using existing local tokenizer files in {local_tokenizer_dir}", flush=True)
        return local_tokenizer_dir

    local_tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer {tokenizer_repo_id} to {local_tokenizer_dir}", flush=True)
    try:
        snapshot_download(
            repo_id=tokenizer_repo_id,
            repo_type="model",
            local_dir=local_tokenizer_dir,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
                "*.model",
                "added_tokens.json",
            ],
        )
    except (GatedRepoError, HfHubHTTPError) as exc:
        raise RuntimeError(
            "Failed to download the gated PaliGemma tokenizer. "
            "Please request access to google/paligemma-3b-pt-224 and authenticate with "
            "`huggingface-cli login` or set `HF_TOKEN`, or pass `--tokenizer-path` to a local tokenizer directory."
        ) from exc

    if not _is_valid_local_tokenizer_dir(local_tokenizer_dir):
        raise RuntimeError(
            "Tokenizer download finished but the local tokenizer directory is still incomplete. "
            f"Expected '{PALIGEMMA_TOKENIZER_REQUIRED_CONFIG}' plus one of {PALIGEMMA_TOKENIZER_PRIMARY_FILES} "
            f"in {local_tokenizer_dir}."
        )

    return local_tokenizer_dir


def load_pi0_policy_config(model_path: Path, device: torch.device, dtype: str) -> PreTrainedConfig:
    print(f"Loading PI0 config from {model_path}", flush=True)
    return PreTrainedConfig.from_pretrained(
        model_path,
        local_files_only=True,
        cli_overrides=[f"--device={device}", f"--dtype={dtype}"],
    )


def _map_pi0_checkpoint_key(checkpoint_key: str) -> list[str]:
    mapped_key = checkpoint_key

    if checkpoint_key.startswith("time_mlp_in."):
        mapped_key = checkpoint_key.replace("time_mlp_in.", "action_time_mlp_in.")
    elif checkpoint_key.startswith("time_mlp_out."):
        mapped_key = checkpoint_key.replace("time_mlp_out.", "action_time_mlp_out.")

    target_keys = [mapped_key if mapped_key.startswith("model.") else f"model.{mapped_key}"]

    if checkpoint_key in (
        "model.paligemma_with_expert.paligemma.lm_head.weight",
        "paligemma_with_expert.paligemma.lm_head.weight",
    ):
        target_keys.append("model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight")

    return target_keys


def _materialize_nonpersistent_meta_buffers(policy: PI0Policy, device: torch.device) -> None:
    vision_embeddings = policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings
    if getattr(vision_embeddings.position_ids, "is_meta", False):
        position_ids = torch.arange(
            vision_embeddings.num_positions,
            device=device,
            dtype=vision_embeddings.position_ids.dtype,
        ).expand((1, -1))
        vision_embeddings.register_buffer("position_ids", position_ids, persistent=False)

    rotary_modules = (
        policy.model.paligemma_with_expert.paligemma.model.language_model.rotary_emb,
        policy.model.paligemma_with_expert.gemma_expert.model.rotary_emb,
    )
    for rotary_module in rotary_modules:
        if getattr(rotary_module.inv_freq, "is_meta", False):
            rope_init_fn = rotary_module.compute_default_rope_parameters
            if rotary_module.rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rotary_module.rope_type]
            inv_freq, attention_scaling = rope_init_fn(rotary_module.config, device=device)
            rotary_module.register_buffer(
                "inv_freq",
                inv_freq.to(device=device, dtype=rotary_module.inv_freq.dtype),
                persistent=False,
            )
            rotary_module.register_buffer(
                "original_inv_freq",
                inv_freq.clone().to(device=device, dtype=rotary_module.original_inv_freq.dtype),
                persistent=False,
            )
            rotary_module.attention_scaling = attention_scaling


def _ensure_no_meta_tensors(policy: PI0Policy) -> None:
    meta_params = [name for name, parameter in policy.named_parameters() if getattr(parameter, "is_meta", False)]
    meta_buffers = [name for name, buffer in policy.named_buffers() if getattr(buffer, "is_meta", False)]
    if meta_params or meta_buffers:
        raise RuntimeError(
            "Low-memory PI0 loader left meta tensors in the model. "
            f"Meta params: {meta_params[:10]}; meta buffers: {meta_buffers[:10]}"
        )


def load_pi0_policy_low_mem(
    model_path: Path,
    device: torch.device,
    dtype: str,
    progress_interval: int = 50,
) -> PI0Policy:
    policy_config = load_pi0_policy_config(model_path, device, dtype)
    meta_config = copy.deepcopy(policy_config)
    meta_config.device = "meta"

    print("Building PI0 model skeleton on meta device", flush=True)
    with init_empty_weights():
        policy = PI0Policy(meta_config)

    target_state = policy.state_dict()
    target_dtypes = {key: value.dtype for key, value in target_state.items()}
    expected_keys = set(target_dtypes)
    loaded_keys: set[str] = set()
    model_file = model_path / "model.safetensors"

    if not model_file.is_file():
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Streaming PI0 weights from {model_file} to {device} with dtype={dtype}", flush=True)
    with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = list(checkpoint.keys())

        for index, checkpoint_key in enumerate(checkpoint_keys, start=1):
            tensor = checkpoint.get_tensor(checkpoint_key)

            for target_key in _map_pi0_checkpoint_key(checkpoint_key):
                tensor_value = tensor
                target_dtype = target_dtypes[target_key]

                if tensor_value.dtype != target_dtype:
                    tensor_value = tensor_value.to(target_dtype)

                set_module_tensor_to_device(policy, target_key, device, value=tensor_value)
                loaded_keys.add(target_key)

            if index % progress_interval == 0 or index == len(checkpoint_keys):
                print(
                    f"Loaded {index}/{len(checkpoint_keys)} checkpoint tensors "
                    f"({len(loaded_keys)}/{len(expected_keys)} model entries)",
                    flush=True,
                )

    missing_keys = sorted(expected_keys - loaded_keys)
    if missing_keys:
        raise RuntimeError(
            "Low-memory PI0 loader did not populate all model parameters. "
            f"Missing keys: {missing_keys[:10]}"
        )

    _materialize_nonpersistent_meta_buffers(policy, device)
    _ensure_no_meta_tensors(policy)

    policy.config = policy_config
    policy.config.device = str(device)
    policy.eval()
    return policy
