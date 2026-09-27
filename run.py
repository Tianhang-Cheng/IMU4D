import os
import re

# Force GPU selection before any torch/CUDA import (overrides accelerate's device selection).
# Set TRAIN_GPU_ID=4 (or desired GPU id) in the shell or in the launch script.
if "TRAIN_GPU_ID" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TRAIN_GPU_ID"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import copy
import zlib
import json
import itertools
import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import tqdm
import random
import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from transformers import AutoConfig, AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, gather_object, set_seed
from safetensors.torch import load_file, save_file

from training.imu_dataset import ENV_EVAL_IMU_TRAJ, IMUDataset, set_cut_length
from training import lora as lora_utils
from dataset_process.wds_pipeline.wds_loader import (
    build_train_wds_loader,
    build_eval_wds_loader,
    SharedCut,
    set_cut_length_wds,
    list_wds_shards,
    load_rewrite_texts,
    num_wds_contiguous_eval_windows,
    num_wds_samples,
    num_wds_trainable_samples,
    restrict_eval_shards,
)
from models import get_mask_chedule, ShowoIMU

from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from utils.rotation2 import recover_absolute_rotation, convert_rotation
from models.time_series_quant import DiscreteQuantizer
from dataset_process.identity_catalog import (
    IdentityCatalog,
    canonical_category,
)
from dataset_process.object_taxonomy import is_static_ground, unique_instance_key
from dataset_process.ncsa import chair_objects as ncsa_chair_objects


IMU_SENSOR_NAMES = (
    "left_hip",
    "right_hip",
    "left_ear",
    "right_ear",
    "left_elbow",
    "right_elbow",
)
NUM_IMU_SENSORS = len(IMU_SENSOR_NAMES)


class HierarchicalIMULayoutSampler:
    """Sample common layouts frequently while exploring the remaining subsets."""

    def __init__(self, config, num_sensors: int = NUM_IMU_SENSORS):
        self.num_sensors = num_sensors
        self.full_ids = tuple(range(num_sensors))
        self.anchor_probability = float(config.anchor_probability)
        if not 0.0 <= self.anchor_probability <= 1.0:
            raise ValueError("anchor_probability must be between 0 and 1")

        self.anchors = []
        anchor_names = []
        for entry in config.anchors:
            active_ids = tuple(sorted(int(idx) for idx in entry.active_imu_id))
            self._validate_active_ids(active_ids, f"anchor {entry.name!s}")
            self.anchors.append(active_ids)
            anchor_names.append(str(entry.name))
        if not self.anchors:
            raise ValueError("train_imu_layout_sampling.anchors must not be empty")
        if len(set(self.anchors)) != len(self.anchors):
            raise ValueError("train_imu_layout_sampling contains duplicate anchors")

        anchor_set = set(self.anchors)
        self.exploration_layouts = {}
        self.exploration_counts = []
        self.exploration_weights = []
        for entry in config.exploration_point_count_weights:
            point_count = int(entry.point_count)
            weight = float(entry.weight)
            if not 1 <= point_count <= num_sensors:
                raise ValueError(f"Invalid exploration point count: {point_count}")
            if weight <= 0:
                raise ValueError("Exploration point-count weights must be positive")
            layouts = [
                layout
                for layout in itertools.combinations(self.full_ids, point_count)
                if layout not in anchor_set
            ]
            if not layouts:
                raise ValueError(
                    f"No non-anchor layouts remain for {point_count}-point exploration"
                )
            self.exploration_layouts[point_count] = layouts
            self.exploration_counts.append(point_count)
            self.exploration_weights.append(weight)

        if self.anchor_probability < 1.0 and not self.exploration_counts:
            raise ValueError("Exploration weights are required when anchor_probability < 1")
        self.anchor_names = tuple(anchor_names)

    def _validate_active_ids(self, active_ids, label):
        if not active_ids or len(set(active_ids)) != len(active_ids) or any(
            idx < 0 or idx >= self.num_sensors for idx in active_ids
        ):
            raise ValueError(f"Invalid active IMU ids for {label}: {list(active_ids)}")

    @staticmethod
    def _realisable(layouts, missing: set[int]):
        return [layout for layout in layouts if not (set(layout) & missing)] if missing else list(layouts)

    def sample_invalid_ids(self, missing_slots=()) -> list[int]:
        """Draw a layout the *sample* can actually realise.

        A real capture may not carry every model slot (``imu_missing_slots``:
        the meeting-room 2-point clips have no earbud).  Those slots are padded
        with zero acceleration and identity orientation and are masked anyway,
        so drawing a layout that names one silently turns the sample into a
        smaller layout -- half of the 2-point clips became *1-point* inputs,
        which pretraining never produces (``train_invalid_imu_id: "random"``
        only ever activates 3 or 5 sensors).  Restricting the draw to the
        layouts whose sensors the sample has keeps the requested point count.
        """
        missing = {int(slot) for slot in missing_slots}
        available = tuple(idx for idx in self.full_ids if idx not in missing)
        if not available:
            raise ValueError("Every IMU slot is missing from this sample")
        if random.random() < self.anchor_probability:
            anchors = self._realisable(self.anchors, missing)
            active_ids = random.choice(anchors) if anchors else available
        else:
            pools = [
                (count, weight, layouts)
                for count, weight in zip(self.exploration_counts, self.exploration_weights)
                for layouts in [self._realisable(self.exploration_layouts[count], missing)]
                if layouts
            ]
            if pools:
                _, _, layouts = random.choices(pools, weights=[pool[1] for pool in pools], k=1)[0]
                active_ids = random.choice(layouts)
            else:
                active_ids = available
        return sorted(set(self.full_ids) - set(active_ids))

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from utils.metrics import compute_mpjpe
from utils.pose_filter import lowpass_pose
from evaluation.footlock_traj import enabled_for as footlock_enabled_for, footlock_sample
from evaluation.object_list_override import override_categories as object_list_override_categories
from evaluation.object_list_override import verified_caption_categories
from dataset_process.asset_frames import canonical_extent_m
from dataset_process.custom_path import pretrained_showo_path, imu_data_path

SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, AverageMeter
from utils.training_utils import EMA

try:
    import apex
    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

def get_model_size_gb(model: torch.nn.Module) -> float:
    """Return total model size in gigabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_bytes = param_size + buffer_size
    return total_size_bytes / (1024 ** 3)  # convert bytes to GB

def extend_attn_mask(mask, M, dtype, invalid_positions):
    """
    Extend attention mask from [1, 1, L, L] to [1, 1, M, M].

    Notes:
        - If M < L, truncates the mask to [1, 1, M, M]
        - If M > L, extends with -inf for padding positions
        - Lower triangle (including diagonal) set to zero
        - Upper triangle set to -inf
        - If invalid_positions is provided (list of int), those positions are isolated:
          they only attend to themselves (row i and col i set to -inf except [i,i]=0).
    """
    L = mask.shape[-1]
    device = mask.device

    # Compact causal/padding mask. New autoregressive tokens are valid; masked
    # IMU positions are removed as keys. Query rows for those input positions
    # are never consumed by task heads during generation.
    if mask.dim() == 2:
        if M <= L:
            result = mask[:, :M].clone()
        else:
            result = torch.cat(
                [mask, torch.ones(mask.shape[0], M - L, dtype=mask.dtype, device=device)],
                dim=-1,
            )
        if invalid_positions:
            inv = torch.as_tensor(invalid_positions, device=device, dtype=torch.long)
            inv = inv[inv < M]
            if inv.numel() > 0:
                result[:, inv] = 0
        return result

    neg_inf = torch.finfo(dtype).min

    # Create a causal mask of shape [M, M]
    causal = torch.full((M, M), neg_inf, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)
    causal = causal + torch.tril(torch.zeros((M, M), dtype=dtype, device=device))

    # Add batch and head dimensions
    result = causal.unsqueeze(0).unsqueeze(0)

    # Copy original mask values where valid
    if M <= L:
        result = result[:, :, :M, :M]
        result[:, :, :, :] = torch.minimum(result, mask[:, :, :M, :M])
    else:
        result[:, :, :L, :L] = torch.minimum(
            result[:, :, :L, :L],
            mask
        )

    # Isolate invalid positions: only attend to self, others don't attend to them
    if invalid_positions is not None and len(invalid_positions) > 0:
        inv = torch.as_tensor(invalid_positions, device=device, dtype=torch.long)
        inv = inv[inv < M]
        if inv.numel() > 0:
            result[:, :, inv, :] = neg_inf
            result[:, :, :, inv] = neg_inf
            # Restore diagonal (i, i) = 0 (flat index i*M+i; fill_diagonal_ requires dim > 1)
            result[0, 0].flatten()[inv * (M + 1)] = 0.0

    # Check for inf or nan values
    if torch.isinf(result).any():
        raise ValueError("Output mask contains inf values")
    if torch.isnan(result).any():
        raise ValueError("Output mask contains nan values")
    return result


def truncate_attn_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    """Slice either a compact [B, L] mask or an additive [B, 1, L, L] mask."""
    if mask.dim() == 2:
        return mask[:, :length]
    if mask.dim() == 4:
        return mask[:, :, :length, :length]
    raise ValueError(f"Unsupported attention mask shape: {tuple(mask.shape)}")


def cached_decode_attention_mask(
    base_mask: torch.Tensor,
    total_length: int,
    invalid_positions,
) -> torch.Tensor:
    """Build only the mask needed by one cached autoregressive query token."""
    if base_mask.dim() == 2:
        result = torch.ones(
            base_mask.shape[0], total_length, dtype=base_mask.dtype, device=base_mask.device
        )
        if invalid_positions:
            inv = torch.as_tensor(invalid_positions, device=result.device, dtype=torch.long)
            inv = inv[inv < total_length]
            result[:, inv] = 0
        return result
    if base_mask.dim() == 4:
        result = torch.zeros(
            base_mask.shape[0], 1, 1, total_length,
            dtype=base_mask.dtype, device=base_mask.device,
        )
        if invalid_positions:
            inv = torch.as_tensor(invalid_positions, device=result.device, dtype=torch.long)
            inv = inv[inv < total_length]
            result[:, :, :, inv] = torch.finfo(base_mask.dtype).min
        return result
    raise ValueError(f"Unsupported attention mask shape: {tuple(base_mask.shape)}")


def get_invalid_imu_positions(input_imu_len: int, invalid_imu_id: list) -> list:
    """Return list of sequence indices that belong to invalid IMU blocks (for attention isolation)."""
    if not invalid_imu_id:
        return []
    block_size = (input_imu_len - 2) // 6
    positions = []
    for k in invalid_imu_id:
        start = 1 + k * block_size
        end = 1 + (k + 1) * block_size
        positions.extend(range(start, end))
    return positions

def build_attention_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """
    valid_mask: Bool tensor [n], True=真实token，False=pad
    Returns: float tensor [n, n]
             规则：仅当(1)因果下三角 & (2)两侧位置均为真实token 时为1；其余为0。
             即：pad 的整行与整列均为0（完全屏蔽）。
    """
    assert valid_mask.dtype == torch.bool and valid_mask.dim() == 1
    n = valid_mask.shape[0]
    device = valid_mask.device

    # 因果下三角
    causal = torch.tril(torch.ones((n, n), dtype=torch.float32, device=device))

    # 仅保留真实token行列（外积：行和列都必须为True才保留）
    vm = valid_mask.to(torch.float32)
    allow = causal * (vm[:, None] * vm[None, :])

    return allow  # 允许=1，不允许=0

# training.canonical_object_order (set once in main): list each sample's objects
# by category id, ground last, instead of the source order, so the object-id
# sequence is a deterministic function of the set (plan A's set head forces
# categories in this same order at inference).
CANONICAL_OBJECT_ORDER = False


def canonical_object_key(name, obj_name_to_id):
    category = canonical_category(name)
    return (is_static_ground(category), obj_name_to_id.get(category, 1 << 30), name)


def imu_to_input(model,
                time_series_quantizer,
                imu_batches,
                accelerator,
                uni_prompting,
                mask_dtype,
                obj_name_to_id,
                object_token_bias,
                identity_catalog,
                normalization_window_size:int,
                smooth_imu: bool,
                random_text: bool,
                invalid_imu_id: "random", # "random" or list[int]
                imu_layout_sampler: HierarchicalIMULayoutSampler | None = None,
                dynamic_object: bool=False,
                bidirectional_imu: bool=False,
                bidirectional_motion: bool=False,
                predict_objects: bool=True,
                fps=None,
                inference_mode: bool=False,
                ):
    """
    This function is used to convert IMU batch data into input format for the model
    Argument:
        invalid_imu_id: list[int], use the invalid imu ids to generate input embeddings
    """
    # assert invalid_imu_id == "random" or isinstance(invalid_imu_id, list), 'got invalid_imu_id: {invalid_imu_id}'
    if not (
        imu_layout_sampler is not None
        or invalid_imu_id == "random"
        or isinstance(invalid_imu_id, list)
    ):
        import pdb; pdb.set_trace()
    if isinstance(invalid_imu_id, list):
        assert all(isinstance(x, int) and 0 <= x <= 5 for x in invalid_imu_id)

    if CANONICAL_OBJECT_ORDER:
        for imu_batch in imu_batches:
            if imu_batch and imu_batch.get('objects'):
                imu_batch['objects'] = {
                    name: imu_batch['objects'][name]
                    for name in sorted(
                        imu_batch['objects'],
                        key=lambda n: canonical_object_key(n, obj_name_to_id),
                    )
                }

    imu_embeddings_batch = []
    query_embeddings_batch = []
    output_embeddings_batch = []
    all_embeddings_batch = []
    label_mean_batch = []
    label_std_batch = []
    label_static_batch = []
    label_text_batch = []
    label_text_token_count_batch = []
    label_object_batch = []
    input_imu_len_batch = []
    motion_len_batch = []

    label_mean_value_batch = []
    label_std_value_batch = []
    label_object_value_batch = []
    label_object_dynamic_value_batch = []
    label_object_dynamic_mask_batch = []
    label_object_anchor_mask_batch = []
    label_gt_motion_batch = []
    label_recon_weight_batch = []  # labels[15]: per-sample [n_var] weights for the recon loss
    label_object_identity_batch = []

    sample_idx_batch = []

    device = accelerator.device

    # weight type should be torch.float32
    assert time_series_quantizer.get_weight_type() == torch.float32
    _model = model.module if hasattr(model, 'module') else model
    get_tokens_embed = _model.embed_tokens

    get_token_embed_static = time_series_quantizer.static_embedder
    get_token_embed_mean = time_series_quantizer.mean_embedder
    get_token_embed_std = time_series_quantizer.std_embedder
    apply_time_embed = time_series_quantizer.time_embedder
    needs_status_embeddings = not bidirectional_motion and not inference_mode

    max_text_len = 0
    text_select_idxs = []
    text_tokens_batch = []
    for imu_batch in imu_batches:

        all_description = imu_batch['description']

        if len(imu_batch) == 0 or len(all_description) == 0:
            text_select_idxs.append(-1)
            text_tokens_batch.append([])
            continue

        if random_text:
            text_select_idx = random.randrange(len(all_description))
        else:
            text_select_idx = 0
        text_select_idxs.append(text_select_idx)
        description = all_description[text_select_idx]
        text_tokens = uni_prompting.text_tokenizer(description)['input_ids']

        # GPT-2-medium has a fixed 1024-entry learned position table.  The
        # current 480-frame recipe uses 120 motion tokens plus six 120-token
        # IMU streams, so an unusually long caption can otherwise overflow the
        # backbone even though ordinary captions fit.  Preserve every caption
        # token that fits the current sample and trim only the architecture-
        # impossible suffix.  Qwen/Phi have larger limits and are unchanged.
        backbone_config = _model.showo.config
        max_positions = getattr(
            backbone_config,
            'max_position_embeddings',
            getattr(backbone_config, 'n_positions', None),
        )
        if max_positions is not None:
            n_time_token = len(imu_batch['imu_data']) // normalization_window_size
            n_object = len(imu_batch.get('objects', {})) if predict_objects else 0
            # 6*n IMU tokens + n motion tokens, 19 fixed delimiters/task
            # tokens, and one ID plus one pose token per object.
            fixed_tokens = 7 * n_time_token + 19 + 2 * n_object
            available_text_tokens = max(int(max_positions) - fixed_tokens, 0)
            text_tokens = text_tokens[:available_text_tokens]
        text_tokens_batch.append(text_tokens)
        max_text_len = max(max_text_len, len(text_tokens))

    # One vocabulary lookup for all captions and object IDs. Separate lookups
    # per sample each allocate a dense vocabulary-sized gradient in backward.
    object_tokens_batch = []
    for sample in imu_batches:
        tokens = []
        if predict_objects:
            for name in sample.get('objects', {}):
                category = canonical_category(name)
                obj_id = obj_name_to_id.get(category)
                assert obj_id is not None, f"Object name {name} not found in obj_name_to_id mapping."
                tokens.append(obj_id + object_token_bias)
        object_tokens_batch.append(tokens)
    token_groups = text_tokens_batch + object_tokens_batch
    packed_ids = torch.tensor(
        [token for group in token_groups for token in group],
        dtype=torch.long, device=device,
    )
    token_embeddings = get_tokens_embed(packed_ids).split([len(group) for group in token_groups])
    text_embeddings_batch = token_embeddings[:len(imu_batches)]
    object_embeddings_batch = token_embeddings[len(imu_batches):]

    # import pdb; pdb.set_trace()

    max_obj_len = 0 # number of objects
    for imu_batch in imu_batches:
        if 'objects' in imu_batch:
            max_obj_len = max(max_obj_len, len(imu_batch['objects']))

    pad_token = uni_prompting.sptids_dict['<|pad|>']
    pad_embedding = get_tokens_embed(pad_token.to(device))  # [d_model]

    # Aggregate every sensor/window in one MLP call.  Samples may have different
    # temporal lengths, so concatenate the ragged [6 * n_window] blocks and split
    # the embeddings back afterwards.
    invalid_imu_id_lists = []
    imu_window_counts = []
    imu_windows_batch = []
    for imu_batch in imu_batches:
        if len(imu_batch) == 0:
            invalid_imu_id_lists.append([])
            imu_window_counts.append(0)
            continue

        sample_invalid_ids = invalid_imu_id
        if imu_layout_sampler is not None:
            sample_invalid_ids = imu_layout_sampler.sample_invalid_ids(
                imu_batch.get('imu_missing_slots', ())
            )
        elif invalid_imu_id == "random":
            valid_combinations = [
                [0, 1, 2, 4, 5],
                [0, 2, 4],
                [1, 2, 4],
                [0, 2, 5],
                [1, 2, 5],
            ]
            # Same rule as HierarchicalIMULayoutSampler: never draw a combination
            # that names a slot this sample does not carry, or the masking below
            # silently turns it into a smaller layout.
            missing = {int(slot) for slot in imu_batch.get('imu_missing_slots', [])}
            usable = [combo for combo in valid_combinations if not (set(combo) & missing)]
            active_ids = (
                random.choice(usable) if usable
                else sorted(set(range(NUM_IMU_SENSORS)) - missing)
            )
            sample_invalid_ids = sorted(set(range(NUM_IMU_SENSORS)) - set(active_ids))
        else:
            sample_invalid_ids = list(sample_invalid_ids)
        # A real-capture variant may have fewer physical devices than the
        # selected layout.  Its processed sample declares those padded model
        # slots; always mask them (e.g. the ear slots in NCSA 2-point clips).
        sample_invalid_ids = sorted(
            set(sample_invalid_ids)
            | {int(slot) for slot in imu_batch.get('imu_missing_slots', [])}
        )
        invalid_imu_id_lists.append(sample_invalid_ids)

        imu_data = imu_batch["imu_data"]
        n_token = len(imu_data) // normalization_window_size
        if n_token * normalization_window_size != len(imu_data):
            raise ValueError(
                f"IMU length {len(imu_data)} is not divisible by "
                f"normalization_window_size={normalization_window_size}"
            )
        if imu_data.shape[1:] != (NUM_IMU_SENSORS, 15):
            raise ValueError(
                f"Expected IMU shape [time, {NUM_IMU_SENSORS}, 15], got "
                f"{tuple(imu_data.shape)}"
            )
        # Reshape on the host; the whole batch is moved in ONE copy below. A
        # pageable host->device copy synchronises the stream, so per-sample
        # copies each stalled the CPU behind the previous micro-batch's backward.
        sensor_windows = (
            imu_data.permute(1, 0, 2)
            .contiguous()
            .reshape(NUM_IMU_SENSORS * n_token, normalization_window_size * 15)
        )
        imu_windows_batch.append(sensor_windows)
        imu_window_counts.append(n_token)

    if imu_windows_batch:
        all_imu_embeddings = _model.imu_aggregator(
            torch.cat(imu_windows_batch, dim=0).to(device=device, dtype=mask_dtype, non_blocking=True)
        )
        per_sample_sizes = [NUM_IMU_SENSORS * n for n in imu_window_counts if n > 0]
        aggregated_per_sample = iter(all_imu_embeddings.split(per_sample_sizes, dim=0))
    else:
        aggregated_per_sample = iter(())

    imu_sensor_embeddings_batch = []
    mask_token_id = torch.tensor(
        accelerator.unwrap_model(model).config.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    mask_embedding = get_tokens_embed(mask_token_id).to(mask_dtype)

    # Special-token ids and embeddings looked up once per call instead of
    # ~25 tiny `.to(device)` + embedding kernels per sample.
    _special_names = (
        '<|imu|>', '<|soimu_0|>', '<|eoimu_0|>', '<|soimu_1|>', '<|eoimu_1|>',
        '<|soimu_2|>', '<|eoimu_2|>', '<|soimu_3|>', '<|eoimu_3|>',
        '<|soimu_4|>', '<|eoimu_4|>', '<|soimu_5|>', '<|eoimu_5|>',
        '<|sostatus|>', '<|eostatus|>', '<|sot|>', '<|eot|>', '<|soobj|>', '<|eoobj|>',
    )
    sp_id = {name: uni_prompting.sptids_dict[name].to(device) for name in _special_names}
    # Keep the graph: these rows of embed_tokens are new (trainable) tokens.
    _sp_emb_all = get_tokens_embed(torch.cat([sp_id[name] for name in _special_names]))
    sp_emb = {
        name: _sp_emb_all[i:i + 1] for i, name in enumerate(_special_names)
    }
    ignore_token = torch.tensor([-100], dtype=torch.long, device=device)

    # Motion quantization for the whole batch in one call. Per-batch crop
    # buckets make every sample the same length, so the targets stack; the
    # per-sample fallback below covers ragged (eval / legacy) batches.
    _motion_lengths = {
        (len(b["transl"]), b["transl"].shape[-1], b["orient"].shape[-1], b["pose"].shape[-1])
        for b in imu_batches if len(b) > 0
    }
    batched_quant = None
    if len(_motion_lengths) == 1 and sum(len(b) > 0 for b in imu_batches) > 1:
        _quant_kwargs = {
            'normalization_window_size': normalization_window_size,
            'return_seperate_indices': True,
            'data_type': 'motion',
            'traj_nvars': next(iter(_motion_lengths))[1],
            'orient_nvars': next(iter(_motion_lengths))[2],
            'debug': False,
        }
        _motion_all = torch.stack([
            torch.cat([b["transl"], b["orient"], b["pose"]], dim=-1)
            for b in imu_batches if len(b) > 0
        ]).float().to(device)  # [n_valid, ntime, nvars], one host->device copy
        with torch.no_grad():
            (
                _, _q_static, _q_mean, _q_std, _q_mean_value, _q_std_value,
            ) = time_series_quantizer(_motion_all, **_quant_kwargs)
            if needs_status_embeddings:
                _q_static_emb = get_token_embed_static(_q_static).split(1)
                _q_dynamic_emb = torch.stack(
                    [get_token_embed_mean(_q_mean), get_token_embed_std(_q_std)], dim=-1
                ).split(1)
            else:
                # Quantized labels are still required; their teacher-forcing
                # embeddings are unused by query-based motion or inference.
                _q_static_emb = _q_dynamic_emb = [None] * _motion_all.shape[0]
        batched_quant = iter(zip(
            _q_static.split(1), _q_mean.split(1), _q_std.split(1),
            _q_mean_value.split(1), _q_std_value.split(1),
            _q_static_emb, _q_dynamic_emb,
            _motion_all.split(1),
        ))

    for n_token, sample_invalid_ids in zip(imu_window_counts, invalid_imu_id_lists):
        if n_token == 0:
            imu_sensor_embeddings_batch.append(None)
            continue
        sensor_embeddings = next(aggregated_per_sample).reshape(
            NUM_IMU_SENSORS, n_token, -1
        )
        imu_sensor_embeddings_batch.append(
            [
                mask_embedding.expand(n_token, -1)
                if sensor_id in sample_invalid_ids
                else sensor_embeddings[sensor_id]
                for sensor_id in range(NUM_IMU_SENSORS)
            ]
        )

    for batch_idx, imu_batch in enumerate(imu_batches):

        sample_idx = imu_batch['sample_idx']
        sample_idx_batch.append(sample_idx)

        if len(imu_batch) == 0:
            continue

        # cur_seq_len = len(imu_batch['imu_data'])
        # diff_seq_len = (max_seq_len - cur_seq_len) // normalization_window_size

        # seq_len_imu = imu_batch["transl"].shape[0]
        # print(f'seq_len_imu = {seq_len_imu}')
        # input: 'imu_airpod' (9), 'imu_iphone' (9), 'imu_watch' (9), 'imu_trans_airpod' (6), 'imu_trans_iphone' (6), 'imu_trans_watch' (6)
        # output: 'transl' (3), 'orient' (3)
        # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
        # Build formatted sequences for IMU
        # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
        """
        imu_static: torch.Tensor,  [bs, n_windows, nvars] = [1, len//4, nvars]
        imu_dynamic: torch.Tensor, [bs, n_windows, nvars*2] = [1, len//4, nvars*2]
        """
        imu_data = imu_batch["imu_data"]

        n_token = len(imu_data) // normalization_window_size

        invalid_imu_id_list = invalid_imu_id_lists[batch_idx]
        (
            imu_0_embeddings,
            imu_1_embeddings,
            imu_2_embeddings,
            imu_3_embeddings,
            imu_4_embeddings,
            imu_5_embeddings,
        ) = imu_sensor_embeddings_batch[batch_idx]

        if batched_quant is not None:
            (
                status_static, status_mean, status_std,
                status_mean_value, status_std_value,
                status_static_emb, status_dynamic_emb, _gt_motion_dev,
            ) = next(batched_quant)
        else:
            _gt_motion_dev = None
            motion_target = torch.cat(
                [imu_batch["transl"], imu_batch["orient"], imu_batch["pose"]], dim=-1
            )
            kwargs = {
                'normalization_window_size': normalization_window_size,
                'return_seperate_indices': True,
                'data_type': 'motion',
                'traj_nvars': imu_batch["transl"].shape[-1],
                'orient_nvars': imu_batch["orient"].shape[-1],
                'debug': False,
            }
            with torch.no_grad():
                (
                    _, status_static, status_mean, status_std,
                    status_mean_value, status_std_value,
                ) = time_series_quantizer(motion_target, **kwargs)
                if needs_status_embeddings:
                    status_static_emb = get_token_embed_static(status_static)
                    status_dynamic_mean_emb = get_token_embed_mean(status_mean)
                    status_dynamic_std_emb = get_token_embed_std(status_std)
                    status_dynamic_emb = torch.stack(
                        [status_dynamic_mean_emb, status_dynamic_std_emb], dim=-1
                    )

        # import pdb; pdb.set_trace()
        status_mean_value = status_mean_value[:, :, 0]
        status_std_value = status_std_value[:, :, 0]

        # Per-sample label reliability (sample key ``motion_supervise``): channel groups the
        # video pseudo-label is not trusted for get -100 token labels, so the CE ignores them
        # while the tokens are still fed as teacher-forced inputs. Variable order is
        # [traj | orient | pose] (see gt_motion_recon below); default = everything supervised.
        _sup = imu_batch.get("motion_supervise") or {}
        _traj_n = imu_batch["transl"].shape[-1]
        _orient_n = imu_batch["orient"].shape[-1]
        _ignore_cols = []
        if not _sup.get("traj", True):
            _ignore_cols += list(range(0, _traj_n))
        if not _sup.get("orient", True):
            _ignore_cols += list(range(_traj_n, _traj_n + _orient_n))
        if not _sup.get("pose", True):
            _ignore_cols += list(range(_traj_n + _orient_n, imu_batch["pose"].shape[-1] + _traj_n + _orient_n))
        # Same reliability applied to the continuous reconstruction loss: a [n_var] weight
        # (traj 3 | orient rot_dof | pose 21*rot_dof), 0 for untrusted groups.
        _n_var_recon = 3 + 22 * 6
        recon_channel_weight = torch.ones(_n_var_recon, device=device)
        if not _sup.get("traj", True):
            recon_channel_weight[0:3] = 0.0
        if not _sup.get("orient", True):
            recon_channel_weight[3:9] = 0.0
        if not _sup.get("pose", True):
            recon_channel_weight[9:] = 0.0
        label_status_mean, label_status_std, label_status_static = status_mean, status_std, status_static
        if _ignore_cols:
            if os.environ.get("IMU4D_DEBUG_SUPERVISE"):
                print(f"[motion_supervise] sample {batch_idx}: ignoring token columns {_ignore_cols[0]}..{_ignore_cols[-1]} "
                      f"({len(_ignore_cols)} of {status_mean.shape[-1]} vars), flags={_sup}", flush=True)
            label_status_mean = status_mean.clone(); label_status_mean[..., _ignore_cols] = -100
            label_status_std = status_std.clone(); label_status_std[..., _ignore_cols] = -100
            label_status_static = status_static.clone(); label_status_static[..., _ignore_cols] = -100

        # Build gt_motion_recon [n_time_token, chunk_size, 3+22*rot_dof] for L2 reconstruction loss (order: traj, orient, pose)
        compression_rate = time_series_quantizer.compression_rate
        if _gt_motion_dev is not None:
            gt_motion = _gt_motion_dev[0]  # already on device, same layout [ntime, 3+22*6]
        else:
            traj_t = imu_batch["transl"].float().to(device)
            orient_t = imu_batch["orient"].float().to(device)
            pose_t = imu_batch["pose"].float().to(device)
            gt_motion = torch.cat([traj_t, orient_t, pose_t], dim=1)  # [ntime, 3+22*6]
        ntime_trim = n_token * compression_rate
        gt_motion = gt_motion[:ntime_trim] # [n_time, 3+22*6]

        if status_mean_value.shape[1] != n_token:
            import pdb; pdb.set_trace()

        if bidirectional_motion:
            # This branch uses learnable queries, not teacher-forced status.
            # The status aggregator output would be discarded (no loss/grad).
            query_embeddings = _model.status_learnable_embeddings.repeat(n_token, 1)
        elif not inference_mode:
            status_embeddings = _model.status_aggregator(
                status_static_emb[0].to(mask_dtype),
                status_dynamic_emb[0].to(mask_dtype),
            )
        else:
            status_embeddings = torch.zeros(
                [n_token, _model.llm_hidden_size], dtype=mask_dtype, device=device
            )

        # Process text description
        if len(imu_batch['description']) == 0:
            description = None
            text_tokens = []
            text_embeddings = torch.zeros([0, get_tokens_embed.weight.shape[1]], dtype=mask_dtype).to(device)
        else:
            # randomly select one
            description = imu_batch['description'][text_select_idxs[batch_idx]]
            text_tokens = text_tokens_batch[batch_idx]
            text_embeddings = text_embeddings_batch[batch_idx]

        # Process object poses. Motion-text pretraining can explicitly disable
        # every object target while retaining both motion and caption losses.
        if not predict_objects or len(imu_batch['objects']) == 0:
            object_id_tokens = []
            # object_status_tokens = []
            object_mean_values = []
            object_dynamic_mean_values = []
            object_dynamic_mask = []
            object_id_embeddings = torch.zeros([0, get_tokens_embed.weight.shape[1]], dtype=mask_dtype).to(device)
            object_status_embeddings = torch.zeros([0, get_tokens_embed.weight.shape[1]], dtype=mask_dtype).to(device)
            cur_object_num = 0
            object_identity_indices = torch.empty(0, dtype=torch.long, device=device)
        else:
            object_id_tokens = []
            # object_status_tokens = []
            object_mean_values = []
            object_dynamic_mean_values = []
            object_dynamic_mask = []
            object_id_embeddings = []
            object_status_embeddings = []
            cur_object_num = 0
            object_identity_indices = []
            object_embedding_rows = object_embeddings_batch[batch_idx].split(1)
            for object_idx, obj in enumerate(imu_batch['objects'].keys()):
                category = canonical_category(obj)
                obj_id = obj_name_to_id.get(category, None)
                assert obj_id is not None, f"Object name {obj} not found in obj_name_to_id mapping."

                obj_id_token = obj_id + object_token_bias
                obj_id_embedding = object_embedding_rows[object_idx]

                if dynamic_object:
                    obj_rot = imu_batch['objects'][obj]['rot'] # [t, 6]
                    obj_transl = imu_batch['objects'][obj]['transl'] # [t,3]
                    obj_dynamic_status = torch.from_numpy(np.concatenate([obj_rot, obj_transl], axis=1)).float().to(device) # [t, 9]
                    obj_status = obj_dynamic_status[0:1] # first frame pose target, [1, 9]
                    neutral_scale = torch.ones_like(obj_status[:, -3:])
                    obj_status_for_embedding = torch.cat([obj_status, neutral_scale], dim=-1)
                    obj_status_embedding = _model.object_aggregator(
                        obj_status_for_embedding, allow_projection=True
                    )
                else:
                    obj_rot = imu_batch['objects'][obj]['rot'] # [6]
                    obj_transl = imu_batch['objects'][obj]['transl'] # [3]
                    try:
                        obj_status = torch.from_numpy(np.concatenate([obj_rot, obj_transl], axis=0)).float().to(device)[None] # [1, 9]
                        neutral_scale = torch.ones_like(obj_status[:, -3:])
                        obj_status_for_embedding = torch.cat([obj_status, neutral_scale], dim=-1)
                        # obj_mean_indices, obj_quantized_mean = time_series_quantizer.mean_quantizer.quantize(obj_status)  # [9], [9], numpy array
                        # obj_mean_indices_embed = get_token_embed_mean(obj_mean_indices)
                        obj_status_embedding = _model.object_aggregator(
                            obj_status_for_embedding, allow_projection=True
                        )
                    except:
                        import pdb; pdb.set_trace()

                object_id_embeddings.append(obj_id_embedding)
                object_status_embeddings.append(obj_status_embedding)
                object_id_tokens.append(obj_id_token)
                # object_status_tokens.append(obj_mean_indices.tolist())
                object_mean_values.append(obj_status)
                metadata = imu_batch.get('object_metadata', {}).get(obj, {})
                object_identity_indices.append(
                    identity_catalog.resolve_index(obj, metadata)
                )
                if dynamic_object:
                    # obj_dynamic_diff_status = obj_dynamic_status.clone()
                    # obj_dynamic_diff_status[1:] = obj_dynamic_diff_status[1:] - obj_dynamic_diff_status[:-1] # diff
                    object_dynamic_mean_values.append(obj_dynamic_status)
                    # The constant ground plane carries no per-frame track
                    # information; it is excluded from the track / motion-state
                    # losses so that mixing object-free sources (MotionMillion)
                    # into dynamic-object training adds no degenerate targets.
                    object_dynamic_mask.append(not is_static_ground(obj))

                cur_object_num += 1

            object_id_tokens = list(object_id_tokens)
            # object_status_tokens = torch.tensor(object_status_tokens)
            object_id_embeddings = torch.cat(object_id_embeddings, dim=0)  # [n_obj, d_model]
            object_status_embeddings = torch.cat(object_status_embeddings, dim=0)  # [n_obj, d_model]
            # import pdb; pdb.set_trace()
            object_mean_values = torch.cat(object_mean_values , dim=0)  # [n_obj, 9]
            object_identity_indices = torch.tensor(
                object_identity_indices, dtype=torch.long, device=device
            )
            if dynamic_object:
                object_dynamic_mean_values = torch.stack(object_dynamic_mean_values, dim=0)  # [n_obj, t, 9]
                object_dynamic_mask = torch.tensor(
                    object_dynamic_mask, dtype=torch.bool, device=device
                )  # [n_obj]

        n_time_step = len(imu_0_embeddings)

        _imu_embeddings = torch.cat([
            sp_emb['<|imu|>'], # 0

            sp_emb['<|soimu_0|>'], # 1
            apply_time_embed(imu_0_embeddings, fps=fps),
            sp_emb['<|eoimu_0|>'], # 2 + n

            sp_emb['<|soimu_1|>'], # 3 + n
            apply_time_embed(imu_1_embeddings, fps=fps),
            sp_emb['<|eoimu_1|>'],

            sp_emb['<|soimu_2|>'],
            apply_time_embed(imu_2_embeddings, fps=fps),
            sp_emb['<|eoimu_2|>'],

            sp_emb['<|soimu_3|>'],
            apply_time_embed(imu_3_embeddings, fps=fps),
            sp_emb['<|eoimu_3|>'],

            sp_emb['<|soimu_4|>'],
            apply_time_embed(imu_4_embeddings, fps=fps),
            sp_emb['<|eoimu_4|>'],

            sp_emb['<|soimu_5|>'],
            apply_time_embed(imu_5_embeddings, fps=fps),
            sp_emb['<|eoimu_5|>'],

            sp_emb['<|sostatus|>'],
        ], dim=0).to(mask_dtype)  # [length, d_model]

        imu_embeddings_batch.append(_imu_embeddings[None]) # [1, length, d_model]


        if bidirectional_motion:
            query_embeddings_with_time = apply_time_embed(query_embeddings, fps=fps) # [n_token, d_model]
            query_embeddings_batch.append(query_embeddings_with_time[None]) # [1, n_token, d_model]
        else:
            status_embeddings_with_time = apply_time_embed(status_embeddings, fps=fps) # [n_token, d_model]

        _output_embeddings = torch.cat([
            status_embeddings_with_time if not bidirectional_motion else query_embeddings_with_time, # not use this one if bidirectional_motion
            sp_emb['<|eostatus|>'],
            sp_emb['<|sot|>'],
            text_embeddings,
            sp_emb['<|eot|>'],
            sp_emb['<|soobj|>'],
            object_id_embeddings,
            sp_emb['<|eoobj|>'],
            object_status_embeddings,
        ], dim=0).to(mask_dtype) # [length, d_model]

        output_embeddings_batch.append(_output_embeddings[None]) # [1, length, d_model]

        _all_embeddings = torch.cat([_imu_embeddings[None], _output_embeddings[None]], dim=1) # [1, length, d_model]
        all_embeddings_batch.append(_all_embeddings) # [1, length, d_model]


        # import pdb; pdb.set_trace()

        _label_text_tokens = torch.cat([
            sp_id['<|eostatus|>'],
            sp_id['<|sot|>'],
            torch.tensor(text_tokens, dtype=torch.long, device=device),  # [length]
            (sp_id['<|eot|>'] if len(text_tokens) > 0 else ignore_token), # if no text, set to -100, don't predict <|eot|>
            (sp_id['<|soobj|>'] if predict_objects else ignore_token),
            torch.tensor(object_id_tokens, dtype=torch.long, device=device),
            (sp_id['<|eoobj|>'] if cur_object_num > 0 else ignore_token), # if no object, set to -100, don't predict <|eoobj|>
        ], dim=0)

        label_mean_batch.append(label_status_mean)
        label_std_batch.append(label_status_std)
        label_static_batch.append(label_status_static)
        label_text_batch.append(_label_text_tokens)
        label_text_token_count_batch.append(len(text_tokens))
        # label_object_batch.append(_label_object_tokens)

        label_mean_value_batch.append(status_mean_value)
        label_std_value_batch.append(status_std_value)
        label_object_value_batch.append(object_mean_values)
        label_gt_motion_batch.append(gt_motion)
        label_recon_weight_batch.append(recon_channel_weight)
        label_object_identity_batch.append(object_identity_indices)
        label_object_anchor_mask_batch.append(torch.tensor(
            [bool(imu_batch.get('object_anchor_valid', {}).get(name, True))
             for name in imu_batch['objects']] if predict_objects else [],
            dtype=torch.bool, device=device,
        ))
        if dynamic_object:
            label_object_dynamic_value_batch.append(object_dynamic_mean_values)
            label_object_dynamic_mask_batch.append(object_dynamic_mask)

        input_seq_len = 1 + (1 + n_time_step + 1) * 6 + 1 # input part of _imu_embeddings, until <|sostatus|> (included)
        n_status_token = n_token
        motion_len_batch.append(n_status_token)
        input_imu_len_batch.append(input_seq_len)
        # output_status_seq_len = status_dynamic.shape[1] * status_dynamic.shape[2] # status part of _label_tokens
        # output_text_seq_len = 1 + 1 + len(text_tokens) + 1 # text part of _label_tokens, include <|eostatus|>, and until <|eot|>
        # assert output_status_seq_len + output_text_seq_len == len(_label_tokens)
        # end batch construction

    # import pdb; pdb.set_trace()
    # concatenate all batches
    seq_all_embeddings = []
    seq_attention_mask = []
    use_compact_causal_mask = not bidirectional_imu and not bidirectional_motion

    max_length = max(x.shape[1] for x in all_embeddings_batch)

    # import pdb; pdb.set_trace()

    for cur_embedding in all_embeddings_batch:
        # cur_embedding: [1, length, d_model]
        cur_len = cur_embedding.shape[1]

        # Handle the case where cur_len might exceed max_length (defensive programming)
        # pad at the end
        if cur_len < max_length:
            # Padding needed
            pad_length = max_length - cur_len
            cur_embedding = torch.cat([cur_embedding, pad_embedding.expand(pad_length, -1)[None]], dim=1)

            valid_position = torch.ones(cur_len, dtype=torch.bool)
            invalid_position = torch.zeros(pad_length, dtype=torch.bool)
            valid_mask = torch.cat([valid_position, invalid_position])
        elif cur_len > max_length:
            raise ValueError
        else:
            valid_mask = torch.ones(cur_len, dtype=torch.bool)

        seq_all_embeddings.append(cur_embedding)
        if use_compact_causal_mask:
            # Hugging Face backbones combine this 2D padding mask with their
            # native causal mask, preserving fused SDPA/FlashAttention paths.
            seq_attention_mask.append(valid_mask[None])
        else:
            cur_mask = build_attention_mask(valid_mask)
            if cur_len < max_length:
                for k in range(cur_len, max_length):
                    cur_mask[k, k] = 1 # avoid all-masked padding query rows
            inverted_mask = 1.0 - cur_mask.type(cur_embedding.dtype)
            inverted_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(mask_dtype).min
            )
            seq_attention_mask.append(inverted_mask[None, None].to(mask_dtype))

    seq_all_embeddings = torch.cat(seq_all_embeddings, dim=0).to(device, non_blocking=True) # [batch_size, length, d_model]
    seq_attention_mask = torch.cat(seq_attention_mask, dim=0).to(device, non_blocking=True)

    imu_embeddings_batch = torch.cat(imu_embeddings_batch, dim=0).to(device, non_blocking=True) # [batch_size, length, d_model]

    if bidirectional_motion:
        query_embeddings_batch = torch.cat(query_embeddings_batch, dim=0).to(device, non_blocking=True) # [batch_size, length, d_model]


    # make the attention mask of IMU parts to be bidirectional
    if bidirectional_imu:
        # Modify attention mask to allow bidirectional attention within IMU parts
        # seq_attention_mask shape: [batch_size, 1, length, length]
        # input_imu_len_batch: list of IMU input lengths for each sample
        batch_size = seq_attention_mask.shape[0]
        for batch_idx in range(batch_size):
            input_imu_len = input_imu_len_batch[batch_idx]

            # Set the IMU part [0:input_imu_len, 0:input_imu_len] to allow bidirectional attention
            # Since the mask is inverted (0 = allow, -inf = block), we set it to 0 for all positions
            # import pdb; pdb.set_trace()
            seq_attention_mask[batch_idx, 0, :input_imu_len, :input_imu_len] = 0.0
            # Note: The causal mask for positions beyond input_imu_len remains unchanged

    if bidirectional_motion:
        # Modify attention mask to allow bidirectional attention within motion (status) parts only.
        # Text and object parts remain causal. Mask convention: 0 = allow, -inf = block.
        # seq_attention_mask shape: [batch_size, 1, length, length]
        batch_size = seq_attention_mask.shape[0]
        seq_len = seq_attention_mask.shape[2]
        for batch_idx in range(batch_size):
            input_imu_len = input_imu_len_batch[batch_idx]
            n_motion = motion_len_batch[batch_idx]
            motion_start = input_imu_len
            motion_end = min(motion_start + n_motion, seq_len)
            if motion_end > motion_start:
                seq_attention_mask[batch_idx, 0, motion_start:motion_end, motion_start:motion_end] = 0.0

    # Isolate invalid IMU positions: they only attend to themselves, others don't attend to them
    # if invalid_imu_id_list:
    #     neg_inf = torch.finfo(seq_attention_mask.dtype).min
    #     batch_size = seq_attention_mask.shape[0]
    #     seq_len = seq_attention_mask.shape[2]
    #     for batch_idx in range(batch_size):
    #         input_imu_len = input_imu_len_batch[batch_idx]
    #         inv = torch.tensor(
    #             get_invalid_imu_positions(input_imu_len, invalid_imu_id_list),
    #             device=seq_attention_mask.device,
    #             dtype=torch.long,
    #         )
    #         inv = inv[inv < seq_len]
    #         if inv.numel() > 0:
    #             seq_attention_mask[batch_idx, 0, inv, :] = neg_inf
    #             seq_attention_mask[batch_idx, 0, :, inv] = neg_inf
    #             # Diagonal (i,i) = 0; fill_diagonal_ requires dim > 1, so use flat index
    #             seq_attention_mask[batch_idx, 0].flatten()[inv * (seq_len + 1)] = 0.0

    labels = [label_mean_batch, label_std_batch, label_static_batch, label_text_batch, None,
              label_mean_value_batch, label_std_value_batch, label_object_value_batch,
              label_object_dynamic_value_batch, None, label_gt_motion_batch,
              label_object_identity_batch, label_text_token_count_batch,
              label_object_dynamic_mask_batch, label_object_anchor_mask_batch,
              label_recon_weight_batch]

    input_dict = {
        'seq_all_embeddings': seq_all_embeddings,
        'imu_embeddings_batch': imu_embeddings_batch,
        'query_embeddings_batch': query_embeddings_batch,
        'labels': labels,
        'seq_attention_mask': seq_attention_mask,
        'input_imu_len_batch': input_imu_len_batch,
        'invalid_imu_id_list': invalid_imu_id_list,
        'sample_idx_batch': sample_idx_batch,
    }

    return input_dict

def _modality_grad_log(model):
    """Render the last micro-batch's per-modality backbone gradient norms.

    Empty unless model.modality_grad_stats or model.modality_grad_clip is set,
    so the log line is unchanged for every existing profile.
    """
    inner = getattr(model, "module", model)
    norms = getattr(inner, "modality_grad_norm", None)
    if not norms:
        return ""
    import torch as _torch

    parts = []
    for name in ("motion", "text", "scene"):
        values = norms.get(name)
        if not values:
            continue
        stacked = _torch.stack([v.float() for v in values])
        ordered = stacked.sort().values.tolist()
        p50 = ordered[len(ordered) // 2]
        p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        parts.append(
            f"Gnorm_{name}: {p50:0.3f}/{p90:0.3f}/{ordered[-1]:0.3f} "
            f"(n={len(ordered)})"
        )
    norms.clear()
    return " ".join(parts) + " " if parts else ""


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    global CANONICAL_OBJECT_ORDER
    CANONICAL_OBJECT_ORDER = bool(config.training.get('canonical_object_order', False))

    # Get job_id from config (can be set via CLI as job_id=0)
    job_id = config.get('job_id', 0)
    total_jobs = config.get('total_jobs', 1)
    assert total_jobs > 0, "total_jobs must be greater than 0"
    assert job_id >= 0 and job_id < total_jobs, "job_id must be between 0 and total_jobs - 1"

    mode = config.experiment.mode # train or test
    train_selected_dataset = config.experiment.train_selected_dataset # HUMOTO, LINGO, ParaHome, humanml, None
    eval_selected_dataset = config.experiment.eval_selected_dataset # HUMOTO, LINGO, ParaHome, humanml, None
    eval_selected_imu_seq = config.experiment.get("eval_selected_imu_seq", None)
    if eval_selected_imu_seq in (None, "", "None"):
        eval_selected_imu_seq = None

    # import pdb; pdb.set_trace()

    if train_selected_dataset == 'None':
        train_selected_dataset = None
    if train_selected_dataset is not None:
        assert train_selected_dataset in ["HUMOTO", "HiPHI", "OMOMO", "LINGO", "ParaHome", 'humanml', 'imuposer', 'dipimu', 'ncsa']
    if eval_selected_dataset == 'None':
        eval_selected_dataset = None
    if eval_selected_dataset is not None:
        assert eval_selected_dataset in ["HUMOTO", "HiPHI", "OMOMO", "LINGO", "ParaHome", 'humanml', 'imuposer', 'dipimu', 'ncsa']
    assert mode in ["train", "test"], "Mode must be train or test"
    # experiment.full_eval_only=True loads the latest checkpoint (and its EMA
    # shadow) exactly like a training resume, runs one full evaluation into
    # evaluation/full/step-N/, and exits without training.
    full_eval_only = bool(config.experiment.get("full_eval_only", False))
    # experiment.full_eval_rerun_only=True additionally restricts the run to the
    # configured Rerun samples (their datasets, the prediction layout only) and
    # stops each pass once the sample is exported. Metrics are not written.
    full_eval_rerun_only = bool(config.experiment.get("full_eval_rerun_only", False))
    if full_eval_rerun_only:
        full_eval_only = True
    if full_eval_only:
        assert mode == "train", "experiment.full_eval_only requires experiment.mode=train"
        assert config.experiment.resume_from_checkpoint, (
            "experiment.full_eval_only requires experiment.resume_from_checkpoint=True"
        )
        assert int(config.experiment.get("full_eval_every", 0)) > 0, (
            "experiment.full_eval_only requires experiment.full_eval_every > 0"
        )
    if mode == "test":
        assert eval_selected_dataset is not None or eval_selected_imu_seq is not None, (
            "test mode requires experiment.eval_selected_dataset and/or experiment.eval_selected_imu_seq"
        )
    assert eval_selected_imu_seq is None or mode == "test", "eval_selected_imu_seq is only valid for mode=test"

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    # Independent of TF32: variable-length training defaults to no cuDNN
    # algorithm search. Tokenizer initialization must preserve this choice.
    torch.backends.cudnn.benchmark = bool(config.training.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    config.experiment.viz_dir = str(Path(config.experiment.output_dir) / "viz_result_val")
    config.experiment.viz_test_dir = str(Path(config.experiment.output_dir) / "viz_test")
    config.experiment.viz_train_dir = str(Path(config.experiment.output_dir) / "viz_result_train")

    dynamic_object = config.model.dynamic_object
    # Which heads carry a loss. `model.supervise` is the general form; the three
    # legacy *_only booleans remain valid as its single-element cases.
    _configured_supervise = config.model.get('supervise', None)
    if _configured_supervise is not None:
        assert not (config.model.text_only or config.model.motion_only
                    or config.model.get('scene_only', False)), (
            "model.supervise replaces text_only / motion_only / scene_only"
        )
        supervise = tuple(str(name) for name in _configured_supervise)
    elif config.model.text_only:
        supervise = ('text',)
    elif config.model.motion_only:
        supervise = ('motion',)
    elif config.model.get('scene_only', False):
        supervise = ('scene',)
    else:
        supervise = ('motion', 'text', 'scene')
    keep_motion, keep_text, keep_scene = (
        'motion' in supervise, 'text' in supervise, 'scene' in supervise
    )
    assert keep_motion or keep_text or keep_scene, "model.supervise is empty"
    # Motion left unsupervised means the motion positions are teacher-forced GT
    # (M2T / M2S / M2TS), which only exists in the causal layout.
    # Unconditional generation: the motion span is neither supervised nor
    # teacher-forced, it stays the learned queries, so there is nothing anywhere
    # in the sequence to condition on (IMU masked, caption dropped). Without
    # this flag `motion not supervised` alone forces the ground-truth motion
    # back in, which is the opposite of unconditional.
    uncond_motion_input = bool(config.model.get('uncond_motion_input', False))
    if uncond_motion_input:
        assert not keep_motion, (
            "model.uncond_motion_input means the motion span is left empty; "
            "remove 'motion' from model.supervise"
        )
        assert config.model.showo.get('bidirectional_motion', False), (
            "model.uncond_motion_input needs showo.bidirectional_motion=True so "
            "the motion positions hold learned queries instead of ground truth"
        )
    gt_motion_input = not keep_motion and not uncond_motion_input
    if gt_motion_input:
        assert not config.model.showo.get('bidirectional_motion', False), (
            "leaving 'motion' out of model.supervise needs "
            "showo.bidirectional_motion=False so the motion positions hold the "
            "teacher-forced ground-truth motion tokens"
        )
    # A profile that does not train the text head drops the caption from the
    # sequence entirely (training/imu_dataset.py).
    drop_caption = not keep_text
    # The single-head cases, derived from `supervise` rather than read from the
    # legacy booleans: a profile written with `model.supervise` leaves those
    # False, and the loader / eval paths still branch on them (eval_model's
    # text_only branch in particular, which a supervise:[text] profile must take
    # -- it has no motion_query_hidden_states for the teacher-forced branch to
    # consume).
    supervise_motion_only = supervise == ('motion',)
    supervise_text_only = supervise == ('text',)
    if dynamic_object:
        dynamic_dataset = (
            train_selected_dataset if mode == "train" else eval_selected_dataset
        )
        if dynamic_dataset is None and mode == "train":
            # Mixed training: sources without object trajectories (MotionMillion)
            # may be trained alongside the HOI sources. They only carry the
            # loader's synthetic ground plane, which is a per-sample constant and
            # is excluded from the object-track / motion-state supervision
            # (dataset_process/object_taxonomy.py is_static_ground), so they add
            # no degenerate dynamic targets. Report the mix for the record.
            _dynamic_sources = {"hiphi", "omomo", "humoto"}
            _dynamic_roots = [str(config.dataset.params.imu_path_or_url)]
            _dynamic_roots += [
                str(r) for r in config.dataset.params.get("train_imu_paths_or_urls", []) or []
            ]
            _dynamic_roots += [
                str(e.get("root"))
                for e in (config.experiment.get("full_eval_datasets", None) or [])
                if e.get("root") is not None
            ]
            _object_roots, _static_roots = [], []
            for _root in dict.fromkeys(_dynamic_roots):
                with open(os.path.join(_root, "wds", "manifest.json"), "r") as _f:
                    _source = json.load(_f).get("source")
                (_object_roots if _source in _dynamic_sources else _static_roots).append(
                    (_root, _source)
                )
            assert _object_roots, (
                "Dynamic object training without train_selected_dataset requires at "
                f"least one HOI source {sorted(_dynamic_sources)}; got "
                f"{[src for _, src in _static_roots]}"
            )
            if _static_roots:
                # Runs before Accelerator() exists, so accelerate's logger is
                # not usable here yet.
                print(
                    f"Dynamic-object training over {len(_object_roots)} object "
                    f"source(s) {[src for _, src in _object_roots]} and "
                    f"{len(_static_roots)} ground-plane-only source(s) "
                    f"{[src for _, src in _static_roots]} (excluded from the "
                    "track / motion-state losses).",
                    flush=True,
                )
        elif mode == "train":
            assert dynamic_dataset in {'HUMOTO', 'HiPHI', 'OMOMO'}, (
                "Dynamic object training/evaluation requires HUMOTO, HiPHI, or OMOMO"
            )

    import wandb
    if mode == "test" or full_eval_only: # disable wandb for evaluation-only runs
        os.environ["WANDB_MODE"] = "disabled"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=False, # the effective batch size is batch_size_imu * #GPU * #accumulation
        # Opt-in for multi-GPU fine-tunes whose frozen/unused heads get no gradient
        # (e.g. real-IMU LoRA runs): DDP otherwise aborts on the second step.
        kwargs_handlers=(
            [DistributedDataParallelKwargs(find_unused_parameters=True)]
            if os.environ.get("DDP_FIND_UNUSED", "0") == "1" else None
        ),
    )

    """
    split_batches=True
    Accelerator tries to split each batch across GPUs.
    Your batch has size 1 → cannot split across 4 GPUs.
    In this case, Accelerate will replicate the batch on all GPUs, because splitting 1 sample across 4 GPUs isn’t possible.

    split_batches=False
    Accelerator does not split the batch at all.
    Each GPU receives the entire batch, which is size 1.
    So again: each GPU sees the same sample.

    each GPU gets roughly N/4 samples per epoch
    """

    total_batch_size_per_gpu = config.training.batch_size_imu
    total_batch_size = total_batch_size_per_gpu * accelerator.num_processes * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    log_datefmt = "%m/%d/%Y %H:%M:%S"
    logging.basicConfig(
        format=log_format,
        datefmt=log_datefmt,
        level=logging.INFO,
    )
    # Save log file to experiment output directory (main process only to avoid conflicts)
    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        if full_eval_only:
            log_file = Path(config.experiment.output_dir) / "evaluation" / "full" / "full_eval_only.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_file = Path(config.experiment.output_dir) / "train.log"
        print(f"Saving log to {log_file}")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))
        logging.getLogger().addHandler(file_handler)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    # An evaluation-only run must not overwrite the training run's config.yaml.
    if accelerator.is_main_process and not full_eval_only:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    base_model = config.model.showo.get('base_model', 'showo')
    supported_base_models = {'showo', 'gpt2-medium', 'qwen3'}
    if base_model not in supported_base_models:
        raise ValueError(
            f"Unsupported model.showo.base_model={base_model!r}; "
            f"expected one of {sorted(supported_base_models)}"
        )
    tokenizer_path = config.model.showo.llm_model_path
    if tokenizer_path is None:
        raise ValueError("config.model.showo.llm_model_path must be set")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    backbone_config = AutoConfig.from_pretrained(tokenizer_path)
    backbone_hidden_size = backbone_config.hidden_size

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
                                       ),
                                       ignore_id=-100)

    print(len(tokenizer))  # e.g., 50305
    print('special tokens : \n', uni_prompting.sptids_dict)

    # add imu special tokens to sptids_dict
    # here 1 is the masking token
    # token_bias == vocab_size
    # token_bias = len(uni_prompting.text_tokenizer) + vq_model.quantize.codebook_size + 1
    token_bias = len(uni_prompting.text_tokenizer) + 8192 + 1 # add 1 for the masking token
    uni_prompting.sptids_dict['<|soimu_0|>'] = torch.tensor([token_bias + 0])
    uni_prompting.sptids_dict['<|eoimu_0|>'] = torch.tensor([token_bias + 1])
    uni_prompting.sptids_dict['<|soimu_1|>'] = torch.tensor([token_bias + 2])
    uni_prompting.sptids_dict['<|eoimu_1|>'] = torch.tensor([token_bias + 3])
    uni_prompting.sptids_dict['<|soimu_2|>'] = torch.tensor([token_bias + 4])
    uni_prompting.sptids_dict['<|eoimu_2|>'] = torch.tensor([token_bias + 5])
    uni_prompting.sptids_dict['<|soimu_3|>'] = torch.tensor([token_bias + 6])
    uni_prompting.sptids_dict['<|eoimu_3|>'] = torch.tensor([token_bias + 7])
    uni_prompting.sptids_dict['<|soimu_4|>'] = torch.tensor([token_bias + 8])
    uni_prompting.sptids_dict['<|eoimu_4|>'] = torch.tensor([token_bias + 9])
    uni_prompting.sptids_dict['<|soimu_5|>'] = torch.tensor([token_bias + 10])
    uni_prompting.sptids_dict['<|eoimu_5|>'] = torch.tensor([token_bias + 11])
    uni_prompting.sptids_dict['<|sostatus|>'] = torch.tensor([token_bias + 12])
    uni_prompting.sptids_dict['<|eostatus|>'] = torch.tensor([token_bias + 13])
    uni_prompting.sptids_dict['<|imu|>'] = torch.tensor([token_bias + 14])  # imu task token
    uni_prompting.sptids_dict['<|soobj|>'] = torch.tensor([token_bias + 15]) # object task start token
    uni_prompting.sptids_dict['<|eoobj|>'] = torch.tensor([token_bias + 16]) # object task end token

    n_imu_special_token = (uni_prompting.sptids_dict['<|eoobj|>'] - uni_prompting.sptids_dict['<|soimu_0|>']).item() + 1

    identity_catalog_path = Path(
        config.model.get(
            'object_identity_catalog_path',
            'dataset_process/object_identity_catalog.json',
        )
    )
    identity_catalog = IdentityCatalog.load(identity_catalog_path)
    obj_name_list = list(identity_catalog.categories)
    obj_name_to_id = {name: idx for idx, name in enumerate(obj_name_list)}
    n_object_class_token = len(obj_name_list)
    object_token_bias = token_bias + n_imu_special_token
    object_token_id_to_name = {v + object_token_bias: k for k, v in obj_name_to_id.items()}
    identity_asset_category_ids = [
        obj_name_to_id[str(asset['category'])]
        for asset in identity_catalog.assets
    ]

    # add imu special tokens and times series tokens
    config.model.showo.vocab_size = token_bias + n_imu_special_token + n_object_class_token

    bidirectional_imu = config.model.showo.bidirectional_imu
    bidirectional_motion = config.model.showo.get('bidirectional_motion', False)
    print(f"Bidirectional IMU: {bidirectional_imu}")
    print(f"Bidirectional motion: {bidirectional_motion}")
    print(f"Supervised heads: {list(supervise)} "
          f"(gt_motion_input={gt_motion_input}, drop_caption={drop_caption})")


    # load time series tokenizer
    use_rope_embed = config.model.get('use_rope_embed', False)
    time_series_quantizer = DiscreteQuantizer(
            compression_rate=config.model.normalization_window_size,
            d_model=backbone_hidden_size,
            totem_folder=config.model.totem_folder,
            large_model=config.model.large_totem_model,
            n_bins=1024,
            accumulate=config.model.accumulate,
            accumulate_orient=config.model.accumulate_orient,
            accumulate_pose=config.model.accumulate_pose,
            local_coordinate=config.model.local_coordinate,
            use_6d=True,
            use_rope_embed=use_rope_embed).to(accelerator.device)
    print(f"Loaded time series quantizer.")
    time_series_quantizer.eval()
    time_series_quantizer.requires_grad_(False)

    static_vocab_size = time_series_quantizer.get_static_vocab_size()
    dynamic_vocab_size = time_series_quantizer.get_dynamic_vocab_size()

    # Initialize Show-o model
    start_time = time.time()

    with accelerator.main_process_first():
        # define model
        from transformers import PretrainedConfig
        if base_model in {'gpt2-medium', 'qwen3'}:
            llm_path = config.model.showo.llm_model_path
            showo_cfg = OmegaConf.to_container(config.model.showo, resolve=True)
            exclude_keys = {'bidirectional_motion', 'add_gate', 'all_continuous_recon', 'partial_continuous_recon'}
            showo_cfg_filtered = {k: v for k, v in showo_cfg.items() if k not in exclude_keys}
            model_config = {
                **backbone_config.to_dict(),
                **showo_cfg_filtered,
                'base_model': base_model,
                'llm_model_path': llm_path,
            }
        else:
            model_config = PretrainedConfig.from_pretrained(
                os.path.join(pretrained_showo_path, 'config.json')
            ).to_dict()
        random_init_backbone = bool(config.model.showo.get('random_init_backbone', False))
        model_config['random_init_backbone'] = random_init_backbone
        assert not (random_init_backbone and config.model.showo.load_from_showo), (
            "random_init_backbone and load_from_showo are mutually exclusive"
        )

        object_geometry_features = config.model.get('object_geometry_features', None)
        if object_geometry_features:
            # The features are indexed by the identity head's asset index, so a
            # catalog rebuilt after the features would silently shift them.
            geometry_asset_ids = np.load(object_geometry_features)['asset_ids'].tolist()
            catalog_asset_ids = [asset['asset_id'] for asset in identity_catalog.assets]
            if geometry_asset_ids != catalog_asset_ids:
                raise ValueError(
                    f"{object_geometry_features} does not match {identity_catalog_path}; "
                    "rerun dataset_process/asset_geometry_features.py"
                )
            print(f"Object geometry conditioning: {object_geometry_features}")

        model = ShowoIMU(
                totem_folder=config.model.totem_folder,
                accumulate=config.model.accumulate,
                accumulate_orient=config.model.accumulate_orient,
                accumulate_pose=config.model.accumulate_pose,
                low_cpu_mem_usage=True,
                compression_rate=config.model.normalization_window_size,
                time_series_static_vocab_size=static_vocab_size,
                time_series_dynamic_vocab_size=dynamic_vocab_size,
                dynamic_object=dynamic_object,
                device_map=None,
                supervise=supervise,
                bidirectional_motion=config.model.showo.get('bidirectional_motion', False),
                add_gate=config.model.showo.get('add_gate', False),
                all_continuous_recon=config.model.showo.get('all_continuous_recon', False),
                partial_continuous_recon=config.model.showo.get('partial_continuous_recon', False),
                identity_asset_category_ids=identity_asset_category_ids,
                identity_embedding_dim=config.model.get('identity_embedding_dim', 256),
                identity_loss_weight=config.model.get('identity_loss_weight', 1.0),
                object_pose_loss_weight=config.model.get('object_pose_loss_weight', 1.0),
                text_loss_weight=config.model.get('text_loss_weight', 1.0),
                object_track_head=config.model.get('object_track_head', 'mlp'),
                object_track_hidden=config.model.get('object_track_hidden', 256),
                object_track_velocity_weight=config.model.get('object_track_velocity_weight', 0.0),
                object_track_acceleration_weight=config.model.get('object_track_acceleration_weight', 0.0),
                object_track_loss_scale=config.model.get('object_track_loss_scale', 200.0),
                object_track_rotation_residual=config.model.get('object_track_rotation_residual', 'add'),
                object_anchor_head=config.model.get('object_anchor_head', 'regression'),
                object_anchor_flow_hidden=config.model.get('object_anchor_flow_hidden', 512),
                object_anchor_flow_steps=config.model.get('object_anchor_flow_steps', 20),
                object_anchor_flow_weight=config.model.get('object_anchor_flow_weight', 1.0),
                object_detach_latent=config.model.get('object_detach_latent', False),
                object_track_root_feature=config.model.get('object_track_root_feature', False),
                object_track_frame=config.model.get('object_track_frame', 'world'),
                object_track_state_head=config.model.get('object_track_state_head', False),
                object_track_state_weight=config.model.get('object_track_state_weight', 5.0),
                object_track_state_speed_threshold=config.model.get('object_track_state_speed_threshold', 0.05),
                object_track_state_angular_threshold=config.model.get('object_track_state_angular_threshold', 5.0),
                object_track_state_filter=config.model.get('object_track_state_filter', 5),
                object_track_state_min_run=config.model.get('object_track_state_min_run', 6),
                object_track_fps=config.model.get('object_track_fps', 30.0),
                object_geometry_features=object_geometry_features,
                object_geometry_hidden=config.model.get('object_geometry_hidden', 512),
                modality_grad_clip=config.model.get('modality_grad_clip', None),
                modality_grad_scale=config.model.get('modality_grad_scale', None),
                modality_grad_stats=config.model.get('modality_grad_stats', False),
                object_set_head=bool(config.model.get('object_set_head', False)),
                object_set_num_categories=max(obj_name_to_id.values()) + 1,
                object_set_token_bias=object_token_bias,
                object_set_ignore_ids=[
                    v for k, v in obj_name_to_id.items() if is_static_ground(k)
                ],
                object_set_weight=config.model.get('object_set_weight', 1.0),
                **model_config,
            )
        print(f"Show-o model size: {get_model_size_gb(model):.2f} GB")
        print("Initialized Show-o model.")

        if base_model == 'showo' and not config.experiment.resume_from_checkpoint:
            assert config.model.showo.load_from_showo or random_init_backbone, (
                "load_from_showo must be True when resume_from_checkpoint is False "
                "(unless model.showo.random_init_backbone is set)"
            )

        if base_model == 'showo' and config.model.showo.load_from_showo:
            # Load from pretrained Show-o model
            logger.info(f"Loading pretrained Show-o model from {pretrained_showo_path}")
            checkpoint = load_file(os.path.join(pretrained_showo_path, 'pytorch_model.safetensors'))
            model_state_dict = model.state_dict()
            loaded_keys = 0
            skipped_keys = []
            with torch.no_grad():
                for k, v in tqdm.tqdm(checkpoint.items(), desc="Loading checkpoint", disable=not accelerator.is_main_process):
                    if k in model_state_dict and v.shape == model_state_dict[k].shape:
                        model_state_dict[k].copy_(v)
                        loaded_keys += 1
                    else:
                        skipped_keys.append(k)
            logger.info(f"Loaded keys: {loaded_keys}, Skipped keys: {len(skipped_keys)}")
            logger.info("Loaded pretrained weights into Show-o model.")
        elif base_model == 'showo' and random_init_backbone:
            logger.info(
                "Show-o backbone built from the phi-1.5 architecture with random "
                "initialization; no pretrained weights were loaded."
            )
        elif base_model in {'gpt2-medium', 'qwen3'}:
            logger.info(
                "Using pretrained %s backbone from %s (loaded in model init).",
                base_model,
                config.model.showo.llm_model_path,
            )

        old_embeddings_len = len(model.embed_tokens.weight.data) # 58498
        if base_model == 'showo':
            assert old_embeddings_len == 58498, f"Old embeddings length {old_embeddings_len} does not match the expected value 58498"
        elif base_model in {'gpt2-medium', 'qwen3'}:
            print(f"Old embeddings length {old_embeddings_len}")

        if config.model.showo.vocab_size != model.vocab_size:
            print(f"Resizing model token embeddings from {model.vocab_size} to {config.model.showo.vocab_size}")
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size

    print(f"Done loading the showo model in {time.time() - start_time:.2f} seconds.")

    mask_id = model.config.mask_token_id
    # assert mask_id == model.vocab_size - 1, f"Mask token id {mask_id} should be vocab_size - 1 ({model.vocab_size - 1})"
    # import pdb; pdb.set_trace()

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    overfit = config.model.overfit
    # We move random contiguous cutting from the dataset to the batch
    # level (imu_to_input) so that all samples in a batch share the
    # same cut_length. Keep this flag to control batch-level cutting.
    train_random_cut = config.experiment.train_random_cut # for training data augmentation
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_opt = ["time_series_quantizer"]
    keep_trainable = []
    if config.training.get('finetune_on_specific_dataset', False):
        no_opt.append("imu_aggregator")
        no_opt.append("status_aggregator")
        no_opt.append("object_aggregator")
        no_opt.append("object_mean_head")
        no_opt.append("pose_refine")
        no_opt.append("pose_head")
        no_opt.append("motion_head")
        no_opt.append("motion_refine")
        no_opt.append("mean_head")
        no_opt.append("std_head")
        no_opt.append("value_head")
        no_opt.append("lm_head")
        no_opt.append("status_learnable_embeddings")
        no_opt.append("embed_tokens")
        # Modules to keep trainable despite the finetune freeze, e.g. the object
        # heads when the real-world set carries object labels (NCSA chairs).
        # Matched on the parameter name below, so it also overrides a shorter
        # no_opt entry that is a substring ("mean_head" in "object_mean_head").
        keep_trainable = list(config.training.get('finetune_trainable_modules', None) or [])
        unknown = sorted(set(keep_trainable) - set(no_opt))
        if unknown:
            raise ValueError(f"training.finetune_trainable_modules names modules the finetune freeze does not cover: {unknown}")
        no_opt = [n for n in no_opt if n not in keep_trainable]
        logger.info(f"Kept trainable despite finetune mode: {keep_trainable}")

        logger.info(f"Set finetune mode, excluding parameters from optimization:")
        for n in no_opt:
            logger.info(f"- {n}")
        logger.info(f"###########################")

    if base_model == 'showo':
        # no_decay = ["bias", "layer_norm.weight","embeddings.weight"]
        no_decay = ["bias", "layernorm.weight","embed_tokens.weight", "asset_embeddings.weight", "status_learnable_embeddings"]
    elif base_model == 'gpt2-medium':
        no_decay = ["bias", "wte", "wpe", "ln_1", "ln_2", "ln_f", "embed_tokens.weight", "asset_embeddings.weight"]
    elif base_model == 'qwen3':
        no_decay = ["bias", "norm.weight", "embed_tokens.weight", "asset_embeddings.weight", "status_learnable_embeddings"]

    # ------------------------------------------------------------------
    # LoRA: freeze the backbone and train low-rank adapters instead.  The
    # adapters are folded back into the base weights when a checkpoint is
    # written (see save_checkpoint), so downstream evaluation / fine-tuning
    # reads an ordinary full checkpoint.
    lora_config = config.training.get('lora', None)
    lora_enabled = bool(lora_config is not None and lora_config.get('enabled', False))
    if lora_enabled:
        lora_rank = int(lora_config.get('r', 32))
        # A null alpha keeps the conventional alpha / r = 2 scaling, so the rank
        # can be changed on its own without also changing the update magnitude.
        lora_alpha = lora_config.get('alpha', None)
        lora_alpha = 2.0 * lora_rank if lora_alpha is None else float(lora_alpha)
        lora_info = lora_utils.apply_lora(
            model,
            scope=str(lora_config.get('scope', 'showo')),
            target_modules=list(
                lora_config.get('target_modules', lora_utils.DEFAULT_TARGET_MODULES)
            ),
            r=lora_rank,
            alpha=lora_alpha,
            dropout=float(lora_config.get('dropout', 0.0)),
            train_norms=bool(lora_config.get('train_norms', True)),
            train_biases=bool(lora_config.get('train_biases', False)),
        )
        logger.info(
            "LoRA enabled: rank=%d alpha=%s dropout=%s scope=%s targets=%s; "
            "%d modules adapted, %.2fM adapter parameters, %d frozen tensors "
            "kept trainable.",
            lora_info["rank"],
            lora_info["alpha"],
            lora_info["dropout"],
            lora_info["scope"],
            ",".join(lora_info["target_modules"]),
            len(lora_info["replaced"]),
            lora_info["lora_parameters"] / 1e6,
            len(lora_info["unfrozen"]),
        )

    train_only = config.training.get('train_only', [])
    if isinstance(train_only, str):
        train_only = [train_only]
    if len(train_only) > 0:
        # First, freeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Then, unfreeze only the parameters in train_only
        for name, param in model.named_parameters():
            if any(train_substring in name for train_substring in train_only):
                param.requires_grad = True
                logger.info(f"Including {name} for optimization")

    # Additionally exclude parameters in no_opt
    for name, param in model.named_parameters():
        if any(no in name for no in no_opt) and not any(keep in name for keep in keep_trainable):
            param.requires_grad = False
            # logger.info(f"Excluding {name} from optimization")

    # print the name of the parameters that will be optimized
    print("########################### Excluding parameters with weight decay ###########################")
    for n, p in model.named_parameters():
        if p.requires_grad and not any(nd in n for nd in no_decay):
            logger.info(f"Optimizing {n} with weight decay")
    print("########################### Optimizing parameters without weight decay ###########################")
    for n, p in model.named_parameters():
        if p.requires_grad and any(nd in n for nd in no_decay):
            logger.info(f"Optimizing {n} without weight decay")

    # import pdb; pdb.set_trace()

    # print the name of all parameters
    for n, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f"Parameter: {n}, shape: {p.shape}")

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %.2fM / %.2fM (%.3f%%)",
        trainable_parameters / 1e6,
        total_parameters / 1e6,
        100.0 * trainable_parameters / max(total_parameters, 1),
    )

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
            # One fused kernel per dtype/device instead of hundreds of small
            # per-tensor launches over the 1.45 B parameters. Mathematically
            # identical; training.fused_optimizer=false restores the old path.
            fused=bool(config.training.get("fused_optimizer", True)) and torch.cuda.is_available(),
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")
    dataset_config = config.dataset.params
    # Prefer the run configuration so evaluation can point at a small local shard
    # subset without editing dataset_process/custom_path.py.  Older configs omit
    # this field and keep the historical global path.
    motion_data_root = dataset_config.get("imu_path_or_url", imu_data_path)
    train_data_roots = dataset_config.get("train_imu_paths_or_urls", [])
    if isinstance(train_data_roots, str):
        train_data_roots = [train_data_roots]
    else:
        train_data_roots = list(train_data_roots)
    if not train_data_roots:
        train_data_roots = [motion_data_root]
    # Rewritten captions are enabled by default.  Omit train_rewrite_roots to
    # discover the conventional sibling qwen3_0.6B_rewrite_v1 directory for
    # every WDS root; use an empty item to disable one particular source.
    use_rewrite_texts = dataset_config.get("use_rewrite_texts", True)
    train_rewrite_roots = dataset_config.get("train_rewrite_roots", None)
    if train_rewrite_roots is not None:
        train_rewrite_roots = (
            [train_rewrite_roots]
            if isinstance(train_rewrite_roots, str)
            else list(train_rewrite_roots)
        )
    train_source_weights = dataset_config.get("train_imu_source_weights", [])
    if isinstance(train_source_weights, (int, float)):
        train_source_weights = [float(train_source_weights)]
    else:
        train_source_weights = list(train_source_weights)
    if train_source_weights:
        assert len(train_source_weights) == len(train_data_roots), (
            "dataset.params.train_imu_source_weights must have one entry per "
            "train_imu_paths_or_urls root."
        )
        assert all(float(weight) > 0 for weight in train_source_weights), (
            "dataset.params.train_imu_source_weights must be positive."
        )

    assert config.training.batch_size_imu > 0, "Batch size must be greater than 0"


    def_dataset = partial(IMUDataset,
        root=motion_data_root,
        overfit=overfit,
        motion_only=supervise_motion_only,
        text_only=supervise_text_only,
        scene_only=drop_caption,
        fps=config.dataset.params.fps,
        dynamic_object=dynamic_object,
        acc_scale=config.dataset.params.get('acc_scale', 1.0),
        gyro_scale=config.dataset.params.get('gyro_scale', 1.0),
    )

    # ---- Training data: stream from WebDataset shards (dataset_process/wds_pipeline). ----
    # DDP-safe: shards split per node/worker + resampled + fixed steps/epoch. The per-epoch
    # random cut length is shared with the workers via shared_cut_imu (set_cut_length_wds).
    # experiment.train_cut_per_batch=True instead draws one length per batch in
    # [min_train_imu_len, max_train_imu_len] (multiples of train_cut_step) inside the
    # workers (long-context stage).
    shared_cut_imu = SharedCut(
        config.experiment.max_train_imu_len,
        min_length=config.experiment.min_train_imu_len,
        max_length=config.experiment.max_train_imu_len,
        per_batch=bool(config.experiment.get("train_cut_per_batch", False)),
        step=int(config.experiment.get("train_cut_step", 1)),
        batch_overhead_frames=int(config.experiment.get("train_cut_batch_overhead_frames", 0)),
    )
    if shared_cut_imu.per_batch:
        logger.info(
            "Per-batch crop buckets (frames -> batch size): %s",
            {L: shared_cut_imu.batch_size_for(L, config.training.batch_size_imu) for L in shared_cut_imu.buckets()},
        )
    # A single external sample must run without downloaded training shards.
    if mode == "train":
        train_sources = []
        num_train_samples = 0
        for train_data_root in train_data_roots:
            source_shards = list_wds_shards(train_data_root, "train")
            assert len(source_shards) > 0, (
                f"No training shards under {train_data_root}/wds/train. Run "
                "the dataset-specific pack_wds.py script or download the dataset first."
            )
            source_samples = num_wds_samples(train_data_root, "train")
            # What the loader's dataset filters actually keep: the packed split also
            # holds upstreams outside the allow-list (MotionGV / Mirror_MotionGV are
            # 63% of the MotionMillion train split), which never reach a batch and
            # must not count towards an epoch.  Falls back to the packed total when
            # the source has no wds/prefix_counts.json index.
            source_trainable, trainable_is_exact = num_wds_trainable_samples(
                train_data_root,
                "train",
                target_datasets=([train_selected_dataset] if train_selected_dataset is not None else None),
            )
            num_train_samples += source_samples
            train_sources.append((train_data_root, source_shards, source_samples, source_trainable))
            logger.info(
                "Added mixed training source %s: %d shards, %d samples, %d trainable%s.",
                train_data_root,
                len(source_shards),
                source_samples,
                source_trainable,
                "" if trainable_is_exact else " (approximate: no prefix index, run "
                "dataset_process/wds_pipeline/index_prefix_counts.py)",
            )
        if train_source_weights:
            # Resampled WebDataset selects from the shard list.  Repeating a
            # source's shard URLs inversely to its sample count makes source-level
            # selection follow the requested weights (approximately when shards
            # are similarly sized), rather than letting the largest dataset win.
            source_rates = [
                float(weight) / source_samples
                for weight, (_, _, source_samples, _) in zip(train_source_weights, train_sources)
            ]
            base_rate = min(source_rates)
            source_repeats = [max(1, int(round(rate / base_rate))) for rate in source_rates]
        else:
            source_repeats = [1] * len(train_sources)

        train_rewritten_texts = load_rewrite_texts(
            train_data_roots,
            "train",
            rewrite_roots=train_rewrite_roots,
            enabled=use_rewrite_texts,
        )
        if use_rewrite_texts:
            logger.info(
                "Loaded %d rewritten training captions from available sidecars.",
                len(train_rewritten_texts),
            )

        train_shards = []
        effective_num_train_samples = 0
        effective_num_trainable_samples = 0
        for (train_data_root, source_shards, source_samples, source_trainable), repeat in zip(
            train_sources, source_repeats
        ):
            train_shards.extend(source_shards * repeat)
            effective_num_train_samples += source_samples * repeat
            effective_num_trainable_samples += source_trainable * repeat
            if repeat > 1:
                logger.info("Repeating mixed source %s %dx for weighted sampling.", train_data_root, repeat)
        # One epoch = one pass' worth of sample exposures over the data the loader
        # keeps.  Two corrections over the packed-count / batch_size_imu estimate
        # this replaces, which understated the real number of passes ~3x on the
        # pretraining mix:
        #   * dropped upstreams (see num_wds_trainable_samples above) are out of the
        #     denominator;
        #   * a micro-batch holds mean_batch_size samples, not batch_size_imu --
        #     that is the size at max_train_imu_len only, and the shorter buckets of
        #     experiment.train_cut_per_batch take proportionally more samples
        #     (8 at 480 frames but 36 at 60, averaging 16.4).
        # The stream is resampled with replacement, so an epoch is still a nominal
        # length, not a guarantee that every clip was visited exactly once, and only
        # the dataset filter is modelled here: clips that a bucket's length floor,
        # filter_short_text or require_text rejects still consume stream, not epoch.
        mean_batch_size = shared_cut_imu.mean_batch_size(config.training.batch_size_imu)
        batches_per_epoch = max(
            math.ceil(effective_num_trainable_samples / (mean_batch_size * accelerator.num_processes)), 1
        )
        # global_step counts optimizer steps, so the epoch length used for step
        # bookkeeping has to be in the same unit as global_step, not in micro-batches.
        num_update_steps_per_epoch = max(
            math.ceil(batches_per_epoch / int(config.training.gradient_accumulation_steps)), 1
        )
        logger.info(
            "Epoch length: %d trainable samples of %d packed / %.1f samples per micro-batch "
            "/ %d process(es) = %d micro-batches = %d optimizer steps.",
            effective_num_trainable_samples,
            effective_num_train_samples,
            mean_batch_size,
            accelerator.num_processes,
            batches_per_epoch,
            num_update_steps_per_epoch,
        )
    else:
        train_shards = []
        train_rewritten_texts = {}
        num_train_samples = 0
        effective_num_train_samples = 0
        batches_per_epoch = 1
        num_update_steps_per_epoch = 1

    # Device-realism augmentation of the virtual IMUs (imu_synthesis/imu_noise.py)
    # and the real-sensor smoothing switch, shared by the train and eval loaders.
    _imu_noise_raw = config.training.get("imu_noise", None)
    imu_noise_cfg = (
        OmegaConf.to_container(_imu_noise_raw, resolve=True)
        if _imu_noise_raw is not None and not isinstance(_imu_noise_raw, dict)
        else _imu_noise_raw
    )
    smooth_real_imu_acc = bool(config.training.get("smooth_real_imu_acc", False))
    _real_heading_aug = config.training.get("real_imu_heading_aug", None)
    real_heading_aug = None if _real_heading_aug is None else {
        str(_k): _v for _k, _v in dict(_real_heading_aug).items()
    }
    if real_heading_aug and bool(real_heading_aug.get("enabled", False)):
        logger.info("Real-IMU per-device heading augmentation: %s", real_heading_aug)
    _short_window = config.training.get("short_window_global_supervision", None)
    short_window_global_supervision = None if _short_window is None else {
        str(_k): _v for _k, _v in dict(_short_window).items()
    }
    if short_window_global_supervision and bool(short_window_global_supervision.get("enabled", False)):
        logger.info(
            "Short-window global supervision (restores traj/orient on masked labels): %s",
            short_window_global_supervision,
        )
    if imu_noise_cfg is not None and bool(imu_noise_cfg.get("enabled", False)):
        logger.info("Synthetic IMU device-realism augmentation enabled: %s", imu_noise_cfg)
    logger.info("Real-IMU 3-tap acceleration smoothing: %s", smooth_real_imu_acc)
    # NCSA meeting-room chairs (dataset_process/ncsa/chair_objects.py). Passed by
    # environment so the forked loader workers of every train / eval loader see it.
    ncsa_object_root = str(config.dataset.params.get("ncsa_object_root", None) or "")
    if ncsa_object_root and not os.path.isfile(os.path.join(ncsa_object_root, "sample_index.json")):
        raise FileNotFoundError(f"dataset.params.ncsa_object_root has no sample_index.json: {ncsa_object_root}")
    os.environ[ncsa_chair_objects.ENV_ROOT] = ncsa_object_root
    logger.info("NCSA chair objects: %s", ncsa_object_root or "off")
    # Body-shape re-simulated virtual IMUs for evaluation (evaluation/imu_body_shape.py).
    eval_imu_traj_source = str(config.experiment.get("eval_imu_traj_source", None) or "")
    if eval_imu_traj_source and not os.path.isfile(eval_imu_traj_source):
        raise FileNotFoundError(f"experiment.eval_imu_traj_source does not exist: {eval_imu_traj_source}")
    os.environ[ENV_EVAL_IMU_TRAJ] = eval_imu_traj_source
    logger.info("Eval virtual-IMU override: %s", eval_imu_traj_source or "off")
    if mode == "train":
        train_dataloader_imu = build_train_wds_loader(
            train_shards,
            batch_size=config.training.batch_size_imu,
            steps_per_epoch=batches_per_epoch,
            shared_cut=shared_cut_imu,
            num_workers=dataset_config.num_workers,
            random_cut=train_random_cut,
            # Rewritten captions are already validated. A short sentence can still
            # be complete supervision (for example, "A person remains stationary.").
            filter_short_text=dataset_config.get("filter_short_text", False),
            # Captioning-only profiles skip clips without a caption instead of
            # spending a step on a batch that carries no text supervision.
            require_text=dataset_config.get("require_text", False),
            target_datasets=([train_selected_dataset] if train_selected_dataset is not None else None),
            motion_only=supervise_motion_only,
            scene_only=drop_caption,
            dynamic_object=dynamic_object,
            fps=config.dataset.params.fps,
            imu_seq_max_len=config.experiment.max_train_imu_len,
            acc_scale=config.dataset.params.get('acc_scale', 1.0),
            gyro_scale=config.dataset.params.get('gyro_scale', 1.0),
            add_imu_noise=bool(config.training.get("add_imu_noise", False)),
            rewritten_texts=train_rewritten_texts,
            imu_noise_cfg=imu_noise_cfg,
            smooth_real_imu_acc=smooth_real_imu_acc,
            real_heading_aug=real_heading_aug,
            short_window_global_supervision=short_window_global_supervision,
        )
        # ---- Eval loaders: deterministic single-pass streams; ranks partition by index. ----
        eval_target = [eval_selected_dataset] if eval_selected_dataset not in (None, "None") else None
        _eval_kwargs = dict(
            num_workers=dataset_config.num_workers,
            fps=config.dataset.params.fps,
            imu_seq_max_len=config.experiment.max_eval_imu_len,
            motion_only=supervise_motion_only,
            scene_only=drop_caption,
            dynamic_object=dynamic_object,
            # Non-train splits only apply the fixed eval low-pass of training.imu_noise.
            imu_noise_cfg=imu_noise_cfg,
            smooth_real_imu_acc=smooth_real_imu_acc,
            real_heading_aug=real_heading_aug,
            short_window_global_supervision=short_window_global_supervision,
        )
        # Train-sample visualization: a few sequences from the train shards (split="test"
        # so the train-only too-short filter stays off).
        train_seq_dataloader_imu = build_eval_wds_loader(
            train_shards,
            cut_length=config.experiment.max_eval_imu_len,
            split="test",
            rewritten_texts=train_rewritten_texts,
            **_eval_kwargs,
        )
        if overfit:
            overfit_num_samples = int(config.training.get("overfit_num_samples", 0))
            if overfit_num_samples > 0:
                from training.fixed_samples import FixedSampleBatches
                import pickle

                fixed_samples = []
                for batch in train_seq_dataloader_imu:
                    fixed_samples.extend(batch)
                    if len(fixed_samples) >= overfit_num_samples:
                        break
                fixed_samples = fixed_samples[:overfit_num_samples]
                if len(fixed_samples) != overfit_num_samples:
                    raise ValueError("Not enough samples for the fixed overfit diagnostic")
                train_dataloader_imu = FixedSampleBatches(
                    fixed_samples, min(config.training.batch_size_imu, overfit_num_samples),
                    batches_per_epoch,
                )
                train_seq_dataloader_imu = FixedSampleBatches(fixed_samples, 1)
                if accelerator.is_main_process:
                    with open(Path(config.experiment.output_dir) / "overfit_samples.pkl", "wb") as handle:
                        pickle.dump(fixed_samples, handle)
                logger.info("Fixed overfit: %d samples, IDs=%s", len(fixed_samples),
                            [sample.get("sample_idx", "unknown") for sample in fixed_samples])
            val_seq_dataloader_imu = train_seq_dataloader_imu
        else:
            val_shards = list_wds_shards(motion_data_root, "val")
            assert len(val_shards) > 0, "No validation shards found."
            # Same single-shard hazard the full-eval loader guards against below:
            # webdataset's check_empty aborts the run the instant a worker is handed
            # no shard, and dataset.params.num_workers is one number for every
            # source.  The real-IMU meeting-room releases pack their whole val/test
            # split into ONE shard, so a worker count sized for a multi-shard source
            # kills the first periodic eval.
            _val_kwargs = dict(_eval_kwargs)
            if len(val_shards) < _val_kwargs["num_workers"]:
                logger.info(
                    "Validation split has %d shard(s); clamping num_workers from %d "
                    "to avoid an empty-shard worker.",
                    len(val_shards), _val_kwargs["num_workers"],
                )
                _val_kwargs["num_workers"] = len(val_shards)
            val_rewritten_texts = load_rewrite_texts(
                [motion_data_root], "val", enabled=use_rewrite_texts
            )
            if use_rewrite_texts:
                logger.info(
                    "Loaded %d rewritten validation captions from available sidecars.",
                    len(val_rewritten_texts),
                )
            val_seq_dataloader_imu = build_eval_wds_loader(
                val_shards, cut_length=config.experiment.max_eval_imu_len,
                target_datasets=eval_target, split="val",
                rewritten_texts=val_rewritten_texts,
                require_rewritten_text=bool(use_rewrite_texts),
                **_val_kwargs
            )

    else:
        train_dataloader_imu = None
        train_seq_dataloader_imu = None
        val_seq_dataloader_imu = None
        eval_target = [eval_selected_dataset] if eval_selected_dataset not in (None, "None") else None
        _eval_kwargs = dict(
            num_workers=dataset_config.num_workers,
            fps=config.dataset.params.fps,
            imu_seq_max_len=config.experiment.max_eval_imu_len,
            motion_only=supervise_motion_only,
            scene_only=drop_caption,
            dynamic_object=dynamic_object,
            imu_noise_cfg=imu_noise_cfg,
            smooth_real_imu_acc=smooth_real_imu_acc,
            real_heading_aug=real_heading_aug,
            short_window_global_supervision=short_window_global_supervision,
        )

    full_eval_every = int(config.experiment.get("full_eval_every", 0))
    # Cascade evaluation: read an upstream run's predicted motion instead of the
    # ground truth a motion-input profile (m2t / m2s / m2ts) is normally fed.
    # The value is a cache built by evaluation/cascade_motion.py from that run's
    # per_sequence dump. Only the motion changes -- sample set, window lengths
    # and caption references stay exactly what this profile would use otherwise,
    # so the result is comparable row-for-row with the ground-truth-motion pass.
    _configured_cascade_motion = config.experiment.get("full_eval_motion_source", None)
    full_eval_cascade_motion = None
    if _configured_cascade_motion:
        if not gt_motion_input:
            raise ValueError(
                "experiment.full_eval_motion_source only applies to a motion-input "
                "profile (motion left out of model.supervise); this one predicts motion"
            )
        from evaluation.cascade_motion import CascadeMotionSource

        full_eval_cascade_motion = CascadeMotionSource(str(_configured_cascade_motion))
        logger.info(
            f"Cascade motion input: {len(full_eval_cascade_motion)} predicted samples "
            f"from {full_eval_cascade_motion.source} "
            f"(cache {full_eval_cascade_motion.path})"
        )
    # Evaluation window lengths (frames) for the full benchmark. Every
    # dataset/layout pass runs once per length and lands under its own
    # evaluation/full/step-N/frames-NNN/ subtree. Clips shorter than
    # full_eval_min_clip_frames (default: the shortest length) are excluded
    # from every pass, exactly as the historical single-length protocol
    # excluded clips shorter than its window; longer clips are windowed to the
    # first min(clip_len, frames) frames, so all passes score the same clip set
    # and a long window evaluates shorter clips whole instead of dropping them.
    # Default: the historical single max_eval_imu_len pass.
    configured_full_eval_frames = config.experiment.get("full_eval_frames", None)
    if configured_full_eval_frames is None:
        full_eval_frames = [int(config.experiment.max_eval_imu_len)]
    else:
        full_eval_frames = [int(value) for value in configured_full_eval_frames]
    if not full_eval_frames or any(value <= 0 for value in full_eval_frames):
        raise ValueError(
            "experiment.full_eval_frames must list positive frame counts, got "
            f"{full_eval_frames}"
        )
    if len(set(full_eval_frames)) != len(full_eval_frames):
        raise ValueError(
            f"experiment.full_eval_frames contains duplicates: {full_eval_frames}"
        )
    configured_min_clip = config.experiment.get("full_eval_min_clip_frames", None)
    full_eval_min_clip_frames = (
        min(full_eval_frames) if configured_min_clip is None else int(configured_min_clip)
    )
    if not 0 < full_eval_min_clip_frames <= min(full_eval_frames):
        raise ValueError(
            "experiment.full_eval_min_clip_frames must be in "
            f"[1, min(full_eval_frames)={min(full_eval_frames)}], got "
            f"{full_eval_min_clip_frames}"
        )
    # Frame offset of the full-eval window inside each clip. The historical
    # protocol scores the first `frames` frames after annotation.action_start
    # (offset 0). A non-zero offset moves the window later in the same clip,
    # which separates "the error grows with the position inside the window"
    # (chunked / re-anchored inference would fix it) from "the motion later in
    # the clip is simply harder" (it would not). Clips shorter than
    # offset + frames yield a shorter window.
    full_eval_start_offset = int(config.experiment.get("eval_start_offset", 0) or 0)
    if full_eval_start_offset < 0:
        raise ValueError(
            f"experiment.eval_start_offset must be >= 0, got {full_eval_start_offset}"
        )
    full_eval_jobs = []
    if mode == "train" and full_eval_every > 0:
        configured_full_eval_datasets = config.experiment.get(
            "full_eval_datasets", None
        )
        if configured_full_eval_datasets is None:
            configured_full_eval_datasets = [
                {
                    "name": str(
                        config.experiment.get("full_eval_dataset", "LINGO")
                    ),
                    "root": motion_data_root,
                    "sample_num": int(
                        config.experiment.get("full_eval_sample_num", 1945)
                    ),
                }
            ]
        if not configured_full_eval_datasets:
            raise ValueError(
                "experiment.full_eval_datasets must contain at least one entry"
            )

        seen_full_eval_datasets = set()
        for dataset_config_entry in configured_full_eval_datasets:
            dataset_name = str(dataset_config_entry.get("name"))
            if not dataset_name or dataset_name == "None":
                raise ValueError("Each full-eval dataset entry must define name")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", dataset_name):
                raise ValueError(
                    f"Invalid full-eval dataset name {dataset_name!r}; use "
                    "letters, digits, '_' or '-'."
                )
            if dataset_name in seen_full_eval_datasets:
                raise ValueError(
                    f"Duplicate full-eval dataset name: {dataset_name}"
                )
            seen_full_eval_datasets.add(dataset_name)

            dataset_root = str(dataset_config_entry.get("root", motion_data_root))
            # The display name may distinguish multiple roots with the same
            # packed sample source (for example NCSA's 2pt and 3pt meeting-room
            # test sets). Filtering must continue to use the sample source.
            dataset_sources_raw = dataset_config_entry.get(
                "sources", dataset_config_entry.get("source", dataset_name)
            )
            if dataset_sources_raw is None:
                dataset_sources = []
            elif isinstance(dataset_sources_raw, str):
                dataset_sources = [str(dataset_sources_raw)]
            else:
                dataset_sources = [str(source) for source in dataset_sources_raw]
            if not dataset_sources or any(
                not source or source == "None" for source in dataset_sources
            ):
                raise ValueError(
                    f"Full-eval dataset {dataset_name} has invalid sources "
                    f"{dataset_sources!r}"
                )
            sample_id_regex = dataset_config_entry.get("sample_id_regex", None)
            sample_id_regex = (
                None if sample_id_regex is None else str(sample_id_regex)
            )
            if sample_id_regex is not None:
                try:
                    re.compile(sample_id_regex)
                except re.error as error:
                    raise ValueError(
                        f"Invalid sample_id_regex for {dataset_name}: "
                        f"{sample_id_regex!r}"
                    ) from error
            sample_ids_file = dataset_config_entry.get("sample_ids_file", None)
            sample_ids = None
            if sample_ids_file is not None:
                sample_ids_path = Path(str(sample_ids_file)).expanduser()
                if not sample_ids_path.is_absolute():
                    sample_ids_path = Path.cwd() / sample_ids_path
                if not sample_ids_path.is_file():
                    raise FileNotFoundError(
                        f"Full-eval sample_ids_file for {dataset_name} does not exist: "
                        f"{sample_ids_path}"
                    )
                sample_ids = [
                    line.strip()
                    for line in sample_ids_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
                if not sample_ids:
                    raise ValueError(
                        f"Full-eval sample_ids_file for {dataset_name} is empty: "
                        f"{sample_ids_path}"
                    )
                if len(set(sample_ids)) != len(sample_ids):
                    raise ValueError(
                        f"Full-eval sample_ids_file for {dataset_name} contains "
                        f"duplicate ids: {sample_ids_path}"
                    )
                bad_prefixes = [
                    sample_id for sample_id in sample_ids
                    if sample_id.split("/", 1)[0] not in dataset_sources
                ]
                if bad_prefixes:
                    raise ValueError(
                        f"Full-eval sample_ids_file for {dataset_name} contains ids "
                        f"outside sources {dataset_sources}: {bad_prefixes[:3]}"
                    )
                if sample_id_regex is not None:
                    sample_id_pattern = re.compile(sample_id_regex)
                    bad_regex_ids = [
                        sample_id for sample_id in sample_ids
                        if sample_id_pattern.search(sample_id) is None
                    ]
                    if bad_regex_ids:
                        raise ValueError(
                            f"Full-eval sample_ids_file for {dataset_name} contains ids "
                            f"outside sample_id_regex {sample_id_regex!r}: "
                            f"{bad_regex_ids[:3]}"
                        )
            dataset_layouts_raw = dataset_config_entry.get("layouts", None)
            dataset_layouts = (
                None if dataset_layouts_raw is None
                else {str(layout) for layout in dataset_layouts_raw}
            )
            contiguous_windows = bool(
                dataset_config_entry.get("contiguous_windows", False)
            )
            contiguous_window_min_frames = int(
                dataset_config_entry.get("contiguous_window_min_frames", 24)
            )
            if contiguous_window_min_frames <= 0:
                raise ValueError(
                    f"Full-eval contiguous_window_min_frames for {dataset_name} "
                    "must be positive"
                )
            available_samples = num_wds_samples(dataset_root, "test")
            configured_sample_num = dataset_config_entry.get("sample_num", None)
            if contiguous_windows:
                # The 60-frame stream has the most windows, so it is a safe
                # upper bound for every per-frame loader.  Shorter streams end
                # naturally at StopIteration; their summaries record their
                # exact evaluated count.
                eval_sample_num = num_wds_contiguous_eval_windows(
                    dataset_root,
                    "test",
                    min(full_eval_frames),
                    target_datasets=dataset_sources,
                    min_clip_length=full_eval_min_clip_frames,
                    min_window_length=contiguous_window_min_frames,
                    sample_id_regex=sample_id_regex,
                    sample_ids=sample_ids,
                )
                if configured_sample_num is not None:
                    raise ValueError(
                        f"Full-eval dataset {dataset_name} cannot combine sample_num "
                        "with contiguous_windows; evaluate the complete selected stream."
                    )
            else:
                eval_sample_num = (
                    (len(sample_ids) if sample_ids is not None else available_samples)
                    if configured_sample_num is None
                    else int(configured_sample_num)
                )
            if eval_sample_num <= 0:
                raise ValueError(
                    f"Full-eval sample_num for {dataset_name} must be positive"
                )
            if not contiguous_windows:
                eval_sample_num = min(eval_sample_num, available_samples)
            if (
                not contiguous_windows
                and sample_ids is not None
                and eval_sample_num != len(sample_ids)
            ):
                raise ValueError(
                    f"Full-eval sample_num for {dataset_name} ({eval_sample_num}) must "
                    f"match the {len(sample_ids)} ids in {sample_ids_path}"
                )

            dataset_shards = list_wds_shards(dataset_root, "test")
            # With wds/sample_index.json (index_prefix_counts.py) the stream
            # skips shards without a selected sample and never unpickles the
            # members the loader's filters would drop -- LINGO is 1.1% of the
            # MotionMillion test split, so this removes ~99% of the decoding.
            # Without the index both stay as they are and the loader filters
            # after decoding, as before.
            dataset_shards, dataset_allowed_keys = restrict_eval_shards(
                dataset_root, "test", dataset_shards,
                target_datasets=dataset_sources,
                sample_id_regex=sample_id_regex,
                sample_ids=sample_ids,
            )
            if dataset_allowed_keys is not None:
                logger.info(
                    "Full-eval dataset %s: sample index keeps %d of %d test shards "
                    "and %d samples before decoding.",
                    dataset_name, len(dataset_shards),
                    len(list_wds_shards(dataset_root, "test")), len(dataset_allowed_keys),
                )
            if not dataset_shards:
                raise FileNotFoundError(
                    f"No test shards found for full-eval dataset {dataset_name} "
                    f"under {dataset_root}"
                )
            # webdataset's check_empty aborts the whole run the instant a worker
            # is handed no shard. dataset.params.num_workers is one number for
            # every source, but HiPHI / OMOMO / HUMOTO ship a SINGLE test shard
            # each while MotionMillion ships 23 -- so a worker count sized for
            # MotionMillion kills any full eval that reaches a single-shard
            # source. Clamp per dataset to what it actually has.
            _eval_kwargs_for_dataset = dict(_eval_kwargs)
            if len(dataset_shards) < _eval_kwargs_for_dataset["num_workers"]:
                logger.info(
                    "Full-eval dataset %s has %d shard(s); clamping num_workers "
                    "from %d to avoid an empty-shard worker.",
                    dataset_name, len(dataset_shards),
                    _eval_kwargs_for_dataset["num_workers"],
                )
                _eval_kwargs_for_dataset["num_workers"] = len(dataset_shards)
            full_eval_rewritten_texts = load_rewrite_texts(
                [dataset_root], "test", enabled=use_rewrite_texts
            )
            require_full_eval_rewrite = bool(use_rewrite_texts) and not (
                supervise_motion_only or drop_caption
            )
            if require_full_eval_rewrite and not full_eval_rewritten_texts:
                raise FileNotFoundError(
                    f"Full evaluation for {dataset_name} requires rewritten text, "
                    f"but no valid test/shared-split rewrite sidecars were found "
                    f"for {dataset_root}"
                )
            if use_rewrite_texts:
                logger.info(
                    "Loaded %d rewritten full-eval captions for %s; samples "
                    "without a rewrite are excluded from this multimodal eval.",
                    len(full_eval_rewritten_texts),
                    dataset_name,
                )
            # cut_length and imu_seq_max_len are baked into the loader, so build
            # one deterministic stream per evaluation length.
            dataset_loaders = {
                frames: build_eval_wds_loader(
                    dataset_shards,
                    cut_length=frames,
                    target_datasets=dataset_sources,
                    split="test",
                    min_clip_length=full_eval_min_clip_frames,
                    sample_id_regex=sample_id_regex,
                    sample_ids=sample_ids,
                    rewritten_texts=full_eval_rewritten_texts,
                    require_rewritten_text=require_full_eval_rewrite,
                    shift=full_eval_start_offset,
                    contiguous_windows=contiguous_windows,
                    contiguous_window_min_frames=contiguous_window_min_frames,
                    allowed_keys=dataset_allowed_keys,
                    **{**_eval_kwargs_for_dataset, "imu_seq_max_len": frames},
                )
                for frames in full_eval_frames
            }
            full_eval_jobs.append(
                (dataset_name, dataset_loaders, eval_sample_num, dataset_layouts)
            )
            logger.info(
                "Configured full evaluation: dataset=%s, sources=%s, sample_id_regex=%s, "
                "sample_ids_file=%s, "
                "root=%s, samples=%d, layouts=%s, frames=%s, min_clip_frames=%d, "
                "start_offset=%d, contiguous_windows=%s, contiguous_tail_min_frames=%d, "
                "every=%d steps.",
                dataset_name,
                dataset_sources,
                sample_id_regex,
                str(sample_ids_path) if sample_ids is not None else None,
                dataset_root,
                eval_sample_num,
                sorted(dataset_layouts) if dataset_layouts is not None else "all",
                full_eval_frames,
                full_eval_min_clip_frames,
                full_eval_start_offset,
                contiguous_windows,
                contiguous_window_min_frames,
                full_eval_every,
            )

    # num_update_steps_per_epoch is computed above (optimizer steps per pass over
    # the trainable samples).  The epoch loop is only a driver -- training stops
    # on max_train_steps -- so the cap is raised whenever the configured number
    # of epochs would end the run early.  Without this, a small fine-tuning set
    # (15 NCSA sessions is one micro-batch per epoch) exits after a few dozen
    # optimizer steps; configs/showo_finetune_ncsa.yaml had to set
    # num_train_epochs: 4000 by hand for exactly this reason.
    num_train_epochs = config.training.get('num_train_epochs', 100)
    required_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch) + 1
    if required_epochs > num_train_epochs:
        logger.info(
            "Raising num_train_epochs %d -> %d so that max_train_steps (%d) is reachable "
            "at %d optimizer steps per epoch.",
            num_train_epochs, required_epochs, config.training.max_train_steps,
            num_update_steps_per_epoch,
        )
        num_train_epochs = required_epochs

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    successful_resume = False
    strict_resume = config.experiment.get("strict_resume", True)

    load_without_optimizer = config.experiment.load_without_optimizer


    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.ckpt_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        assert path is not None, "No checkpoint found"

        # load from single GPU checkpoint
        if mode == "test" or load_without_optimizer:
            checkpoint_path = os.path.join(config.experiment.ckpt_dir, path)
            global_step = int(os.path.basename(checkpoint_path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {checkpoint_path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(os.path.join(checkpoint_path, "unwrapped_model", "pytorch_model.bin"), map_location=accelerator.device)

            model_state_dict = accelerator.unwrap_model(model).state_dict()

            checkpoint_state_dict = state_dict
            # Non-strict loading would silently drop a trained geometry head
            # when the config does not enable it (e.g. a benchmark built from
            # the yaml without OBJ_GEOM_FEATURES), evaluating a different model.
            if (
                any(key.startswith("object_geometry_embed.") for key in checkpoint_state_dict)
                and "object_geometry_embed.0.weight" not in model_state_dict
            ):
                raise ValueError(
                    f"{checkpoint_path} was trained with object geometry conditioning; set "
                    "model.object_geometry_features (OBJ_GEOM_FEATURES=dataset_process/asset_geometry_bps.npz)"
                )
            compatible_state_dict = {}
            missing_keys = []
            unexpected_keys = []
            shape_mismatches = []

            if not strict_resume:

                reset_object_modules = bool(
                    load_without_optimizer
                    and config.experiment.get(
                        "reset_object_modules_on_weight_load", False
                    )
                )
                reinitialized_object_keys = []
                reinitialized_object_vocab_keys = []

                for key in model_state_dict.keys():
                    if key in checkpoint_state_dict:
                        model_param = model_state_dict[key]
                        ckpt_param = checkpoint_state_dict[key].to(model_param.device)

                        # A taxonomy change invalidates every learned object
                        # component, even when its tensor shape still matches.
                        # Preserve backbone/motion/text/IMU parameters only;
                        # object heads/aggregators and identity retrieval start
                        # from the new model initialization.
                        if reset_object_modules and key.startswith(
                            ("object_", "identity_head.")
                        ):
                            reinitialized_object_keys.append(key)
                            continue
                        if reset_object_modules and (
                            key.endswith("embed_tokens.weight")
                            or key.endswith("lm_head.weight")
                            or key.endswith("lm_head.bias")
                        ):
                            new_param = model_param.clone()
                            prefix_rows = min(
                                int(object_token_bias),
                                ckpt_param.shape[0],
                                model_param.shape[0],
                            )
                            new_param[:prefix_rows] = ckpt_param[:prefix_rows]
                            compatible_state_dict[key] = new_param
                            reinitialized_object_vocab_keys.append(key)
                            continue

                        if ckpt_param.shape == model_param.shape:
                            compatible_state_dict[key] = ckpt_param
                        else:
                            # Handle shape mismatch
                            shape_mismatches.append(
                                f"{key}: {ckpt_param.shape} -> {model_param.shape}"
                            )

                            # Copy overlapping region
                            new_param = model_param.clone()
                            slices = tuple(
                                slice(0, min(ckpt_param.shape[i], model_param.shape[i]))
                                for i in range(min(ckpt_param.dim(), model_param.dim()))
                            )
                            new_param[slices] = ckpt_param[slices]
                            compatible_state_dict[key] = new_param

                # Load compatible weights into unwrapped model
                missing_keys, unexpected_keys = model.load_state_dict(
                    compatible_state_dict, strict=False
                )

                logger.info(f"Loaded {len(compatible_state_dict)} compatible weights into unwrapped model.")
                logger.info(f"Missing keys (new modules): {len(missing_keys)}")
                logger.info(f"Unexpected keys (removed): {len(unexpected_keys)}")
                checkpoint_only_keys = sorted(
                    set(checkpoint_state_dict) - set(model_state_dict)
                )
                if checkpoint_only_keys:
                    logger.info(
                        "Dropped %d checkpoint tensors absent from the model: %s",
                        len(checkpoint_only_keys),
                        ", ".join(checkpoint_only_keys),
                    )
                if missing_keys:
                    logger.info("Missing keys (new modules): %s", ", ".join(missing_keys))
                logger.info(f"Shape mismatches handled: {len(shape_mismatches)}")
                if reset_object_modules:
                    logger.info(
                        "Reinitialized %d object-module tensors: %s",
                        len(reinitialized_object_keys),
                        ", ".join(reinitialized_object_keys),
                    )
                    logger.info(
                        "Reinitialized semantic object rows in %d vocabulary tensors: %s",
                        len(reinitialized_object_vocab_keys),
                        ", ".join(reinitialized_object_vocab_keys),
                    )
                logger.info(f"Loaded single GPU checkpoint from {checkpoint_path}.")

            else:
                model.load_state_dict(checkpoint_state_dict, strict=True)
                logger.info(f"Strictly loaded checkpoint from {checkpoint_path}.")

            logger.info("Preparing model, optimizer and lr_scheduler")
            # wds train loader is a streaming IterableDataset and is intentionally NOT
            # prepared: accelerate cannot restore its state, so dataloader-state resume is dropped.
            model, optimizer, lr_scheduler = accelerator.prepare(
                model, optimizer, lr_scheduler,
            )
            successful_resume = True

            # A weights-only load is also used to initialize a new training
            # stage.  Keep its optimizer and scheduler fresh in that case,
            # rather than treating the source checkpoint's step as a resume.
            if config.experiment.get("reset_step_on_weight_load", False):
                logger.info("Resetting global step and epoch after weights-only initialization.")
                global_step = 0
                first_epoch = 0
            elif global_step > 0:
                # Continuing the source checkpoint's step numbering.  The
                # scheduler was rebuilt with this load and sits at step 0 while
                # training resumes at global_step, so advance it to match: it is
                # stepped once per optimizer step and counts independently of
                # global_step, and leaving it behind would spend the remaining
                # (max_train_steps - global_step) steps on the front of the
                # curve and stop before it ever decays to its floor.
                logger.info(
                    "Advancing the rebuilt lr_scheduler to step %d to match the resumed step.",
                    global_step,
                )
                for _ in range(global_step):
                    lr_scheduler.step()
                logger.info(
                    "Resumed learning rate: %.3e", lr_scheduler.get_last_lr()[0]
                )

        # load from distributed checkpoint
        elif mode == "train" and not load_without_optimizer:
            checkpoint_path = os.path.join(config.experiment.ckpt_dir, path)

            logger.info("Preparing model, optimizer and lr_scheduler")
            # wds train loader is NOT prepared (streaming); dataloader-state resume dropped.
            model, optimizer, lr_scheduler = accelerator.prepare(
                model, optimizer, lr_scheduler,
            )

            global_step = int(os.path.basename(checkpoint_path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            accelerator.print("Loaded complete checkpoint")

            metadata_path = Path(checkpoint_path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                accelerator.print(f"Loaded checkpoint metadata: step {metadata.get('global_step')}, epoch {metadata.get('epoch')}")
                first_epoch = metadata.get('epoch')

            accelerator.print(f"Resumed from step {global_step}, epoch {first_epoch}")
            successful_resume = True

    if not successful_resume and config.experiment.resume_from_checkpoint and not overfit:
        raise ValueError("Failed to resume from checkpoint but resume_from_checkpoint is set to True.")

    # no checkpoint found, train from scratch
    if not successful_resume:
        # wds train loader is NOT prepared (streaming IterableDataset).
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler,
        )
        logger.info("No checkpoint found, training from scratch.")

    if hasattr(model, 'module'):
        mask_dtype = model.module.embed_tokens.weight.dtype
    else:
        mask_dtype = model.embed_tokens.weight.dtype


    train_invalid_imu_id = config.experiment.train_invalid_imu_id
    if OmegaConf.is_list(train_invalid_imu_id):
        train_invalid_imu_id = list(train_invalid_imu_id)
    eval_invalid_imu_id = config.experiment.eval_invalid_imu_id
    if OmegaConf.is_list(eval_invalid_imu_id):
        eval_invalid_imu_id = list(eval_invalid_imu_id)

    train_imu_layout_sampler = None
    layout_sampling_config = config.experiment.get("train_imu_layout_sampling", None)
    if (
        mode == "train"
        and layout_sampling_config is not None
        and bool(layout_sampling_config.get("enabled", False))
    ):
        train_imu_layout_sampler = HierarchicalIMULayoutSampler(layout_sampling_config)
        logger.info(
            "Enabled hierarchical IMU layout sampling: anchor_probability=%.3f, "
            "anchors=%s, exploration_point_counts=%s.",
            train_imu_layout_sampler.anchor_probability,
            train_imu_layout_sampler.anchor_names,
            train_imu_layout_sampler.exploration_counts,
        )

    configured_eval_imu_configs = config.experiment.get("eval_imu_configs", None)
    if configured_eval_imu_configs is None:
        # Keep existing configs/scripts backward compatible.
        eval_imu_configs = [(None, eval_invalid_imu_id)]
    else:
        eval_imu_configs = []
        for imu_config in configured_eval_imu_configs:
            name = str(imu_config.name)
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(
                    f"Invalid eval IMU config name {name!r}; use letters, digits, '_' or '-'."
                )
            has_active_ids = "active_imu_id" in imu_config
            has_invalid_ids = "invalid_imu_id" in imu_config
            if has_active_ids == has_invalid_ids:
                raise ValueError(
                    f"Evaluation config {name!r} must define exactly one of "
                    "active_imu_id or invalid_imu_id."
                )
            if has_active_ids:
                active_ids = [int(idx) for idx in imu_config.active_imu_id]
                invalid_ids = sorted(set(range(NUM_IMU_SENSORS)) - set(active_ids))
                ids_to_validate = active_ids
                id_kind = "active"
            else:
                invalid_ids = [int(idx) for idx in imu_config.invalid_imu_id]
                ids_to_validate = invalid_ids
                id_kind = "invalid"
            if len(set(ids_to_validate)) != len(ids_to_validate) or any(
                idx < 0 or idx >= NUM_IMU_SENSORS for idx in ids_to_validate
            ):
                raise ValueError(
                    f"Invalid {id_kind} IMU ids for evaluation config {name!r}: "
                    f"{ids_to_validate}"
                )
            point_count = NUM_IMU_SENSORS - len(invalid_ids)
            point_prefix = re.match(r"^(\d+)pt(?:_|$)", name)
            if point_prefix and int(point_prefix.group(1)) != point_count:
                raise ValueError(
                    f"Evaluation config {name!r} names {point_prefix.group(1)} points "
                    f"but activates {point_count}."
                )
            eval_imu_configs.append((name, invalid_ids))
        if not eval_imu_configs:
            raise ValueError("experiment.eval_imu_configs must contain at least one entry")

    full_eval_rerun_config = config.experiment.get("full_eval_rerun", None)
    if full_eval_jobs and full_eval_rerun_config is not None and bool(
        full_eval_rerun_config.get("enabled", False)
    ):
        rerun_layout = str(full_eval_rerun_config.prediction_layout)
        available_layouts = {name or "default" for name, _ in eval_imu_configs}
        if rerun_layout not in available_layouts:
            raise ValueError(
                f"full_eval_rerun.prediction_layout={rerun_layout!r} is not "
                f"one of {sorted(available_layouts)}"
            )
        configured_dataset_names = {name for name, _, _, _ in full_eval_jobs}
        # Each dataset maps to one sample id or a list of sample ids; every
        # id gets its own rerun/<motion-id>/ export directory.
        rerun_samples = {}
        for name, sample_ids in full_eval_rerun_config.samples.items():
            if sample_ids is None:
                # Profiles disable an inherited source by setting it to null.
                continue
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            sample_ids = [str(sample_id) for sample_id in sample_ids]
            if not sample_ids or len(set(sample_ids)) != len(sample_ids):
                raise ValueError(
                    f"full_eval_rerun.samples[{name!r}] must be a non-empty list of "
                    f"distinct sample ids, got {sample_ids}"
                )
            rerun_samples[str(name)] = sample_ids
        unknown_datasets = set(rerun_samples) - configured_dataset_names
        if unknown_datasets:
            raise ValueError(
                "full_eval_rerun.samples contains unknown datasets: "
                f"{sorted(unknown_datasets)}"
            )
        if not rerun_samples:
            raise ValueError("full_eval_rerun.samples must name at least one full-eval dataset")
        for dataset_name, sample_ids in rerun_samples.items():
            for sample_id in sample_ids:
                if not sample_id.startswith(f"{dataset_name}/"):
                    raise ValueError(
                        f"Rerun sample {sample_id!r} must start with "
                        f"{dataset_name!r} followed by '/'"
                    )
        required_asset_roots = {"omomo", "hiphi", "humoto"}
        configured_asset_roots = set(full_eval_rerun_config.asset_roots.keys())
        if configured_asset_roots != required_asset_roots:
            raise ValueError(
                "full_eval_rerun.asset_roots must contain exactly "
                f"{sorted(required_asset_roots)}"
            )
    else:
        full_eval_rerun_config = None
        rerun_samples = {}

    eval_model_func = partial(eval_model,
        time_series_quantizer=time_series_quantizer,
        uni_prompting=uni_prompting,
        object_token_id_to_name=object_token_id_to_name,
        obj_name_to_id=obj_name_to_id,
        object_token_bias=object_token_bias,
        identity_catalog=identity_catalog,
        accelerator=accelerator,
        config=config,
        dynamic_object=dynamic_object,
        bidirectional_imu=bidirectional_imu,
        bidirectional_motion=bidirectional_motion,
        fps=config.dataset.params.fps,
        motion_only=supervise_motion_only,
        text_only=supervise_text_only,
        gt_motion_input=gt_motion_input,
        uncond_motion_input=uncond_motion_input,
        decode_objects=keep_scene,
        # Sample a caption only from a text head that carries a loss; the
        # auto default (null) follows model.supervise, an explicit value wins.
        decode_text=(
            keep_text
            if config.experiment.get("eval_decode_text", None) is None
            else bool(config.experiment.eval_decode_text)
        ),
        generate_number=True,
        max_new_text_tokens=config.experiment.max_new_text_tokens,
        max_object_id_tokens=config.experiment.max_object_id_tokens,
        min_object_id_tokens=config.experiment.min_object_id_tokens,
        object_list_override=config.experiment.get("eval_object_list_override", None),
        teacher_force_text=bool(config.experiment.get("eval_teacher_force_text", False)),
        caption_verify_threshold=float(config.experiment.get("eval_caption_verify_threshold", 0.05)),
        feed_soobj=bool(config.experiment.get("eval_feed_soobj", True)),
        text_diverse=load_text_diverse(config),
        scene_sample=load_scene_sample(config),
        forced_decode=load_forced_decode(
            config.experiment.get("eval_forced_decode_file", None),
            config.experiment.get("eval_forced_decode_variant", None),
        ),
        object_multilabel_threshold=config.experiment.get("eval_object_multilabel_threshold", 0.1),
    )

    def teacher_forced_val(step: int, max_batches: int = 32) -> dict:
        """Teacher-forced validation losses, directly comparable to the step_* ones.

        The periodic eval only reports generation metrics, and its text CE is
        computed on the model's own sampled prefix -- exposure-biased and, until
        the decoder went greedy, not even reproducible. That leaves nothing to
        early-stop on. This runs the SAME forward as training (ground-truth text
        in the sequence, one pass, no rollout) over a fixed slice of the val
        stream, so val_loss_text can be read against step_loss_text.

        Determinism matters more than coverage here: a fixed layout
        (train_invalid_imu_id, no layout sampler) and random_text=False, so a
        change between two evals is the model moving, not the sampling.
        """
        was_training = model.training
        model.eval()
        totals, n = {}, 0
        with torch.no_grad():
            for batch_idx, imu_batches in enumerate(val_seq_dataloader_imu):
                if batch_idx >= max_batches:
                    break
                if not imu_batches:
                    continue
                vin = imu_to_input(
                    model,
                    time_series_quantizer,
                    imu_batches,
                    accelerator,
                    uni_prompting,
                    mask_dtype,
                    obj_name_to_id,
                    object_token_bias,
                    identity_catalog,
                    normalization_window_size=config.model.normalization_window_size,
                    smooth_imu=config.model.smooth_imu,
                    invalid_imu_id=train_invalid_imu_id,
                    imu_layout_sampler=None,
                    random_text=False,
                    dynamic_object=dynamic_object,
                    bidirectional_imu=bidirectional_imu,
                    bidirectional_motion=bidirectional_motion,
                    predict_objects=config.model.get('predict_objects', True),
                    fps=config.dataset.params.fps,
                )
                vlosses = model(
                    input_embeddings=vin['seq_all_embeddings'],
                    query_embeddings=vin['query_embeddings_batch'],
                    attention_mask=vin['seq_attention_mask'],
                    labels=vin['labels'],
                    batch_size_imu=1,
                    input_imu_len=vin['input_imu_len_batch'],
                )
                for k in ("total_loss", "loss_text", "loss_mean", "loss_std",
                          "loss_static", "text_accuracy"):
                    v = vlosses.get(k, None)
                    if v is not None:
                        totals[k] = totals.get(k, 0.0) + float(v.detach().float().reshape(()))
                n += 1
        if was_training:
            model.train()
        if n == 0:
            return {}
        return {f"val_{k}": v / n for k, v in totals.items()}

    def run_full_eval(unwrapped_model, step: int) -> None:
        if not full_eval_jobs:
            return
        step_output_root = os.path.join(
            config.experiment.output_dir,
            "evaluation",
            "full",
            f"step-{step:06d}",
        )
        if accelerator.is_main_process:
            os.makedirs(step_output_root, exist_ok=True)
            manifest = {
                "schema_version": 2,
                "evaluation": "full",
                "experiment": str(config.experiment.name),
                "step": int(step),
                "datasets": [name for name, _, _, _ in full_eval_jobs],
                "imu_layouts": [name or "default" for name, _ in eval_imu_configs],
                # Evaluation window lengths; every length owns a frames-NNN subtree.
                "frames": [int(frames) for frames in full_eval_frames],
                "min_clip_frames": int(full_eval_min_clip_frames),
                # Frame offset of the window inside each clip (0 = historical).
                "start_offset": int(full_eval_start_offset),
                "layout": (
                    "frames-<NNN>/datasets/<source>/"
                    "{metrics/<imu_layout>,rerun/<motion-id>}"
                ),
                "rerun": None,
            }
            if full_eval_rerun_config is not None:
                manifest["rerun"] = {
                    "prediction_layout": str(full_eval_rerun_config.prediction_layout),
                    "samples": {
                        name: list(sample_ids)
                        for name, sample_ids in rerun_samples.items()
                    },
                    "files_per_sample": [
                        "gt.rrd", "pred.rrd", "compare.rrd", "metadata.json"
                    ],
                }
            manifest_path = Path(step_output_root) / "manifest.json"
            manifest_tmp = manifest_path.with_suffix(".json.tmp")
            manifest_tmp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(manifest_tmp, manifest_path)
        if full_eval_rerun_only and full_eval_rerun_config is None:
            raise ValueError(
                "experiment.full_eval_rerun_only=True requires an enabled "
                "experiment.full_eval_rerun configuration"
            )
        rerun_only_scratch = os.path.join(step_output_root, ".rerun_only_scratch")
        for frames in full_eval_frames:
            frames_dir = f"frames-{frames:03d}"
            frames_output_root = os.path.join(step_output_root, frames_dir)
            for dataset_name, dataset_loaders, eval_sample_num, dataset_layouts in full_eval_jobs:
                dataset_output_root = os.path.join(
                    frames_output_root, "datasets", dataset_name.lower()
                )
                if full_eval_rerun_only and dataset_name not in full_eval_rerun_config.samples:
                    continue
                for imu_config_name, invalid_ids in eval_imu_configs:
                    layout_name = imu_config_name or "default"
                    if dataset_layouts is not None and layout_name not in dataset_layouts:
                        continue
                    if full_eval_rerun_only and layout_name != str(
                        full_eval_rerun_config.prediction_layout
                    ):
                        continue
                    rerun_export_job = None
                    if (
                        full_eval_rerun_config is not None
                        and layout_name == str(full_eval_rerun_config.prediction_layout)
                        and dataset_name in full_eval_rerun_config.samples
                    ):
                        target_sample_ids = list(rerun_samples[dataset_name])
                        output_dirs = {}
                        for target_sample_id in target_sample_ids:
                            motion_id = target_sample_id.rsplit("/", 1)[-1]
                            safe_motion_id = re.sub(r"[^A-Za-z0-9._-]", "_", motion_id)
                            output_dirs[target_sample_id] = os.path.join(
                                dataset_output_root, "rerun", safe_motion_id
                            )
                        rerun_export_job = {
                            "target_sample_ids": target_sample_ids,
                            "dataset": dataset_name,
                            "layout": layout_name,
                            "eval_frames": int(frames),
                            "output_dirs": output_dirs,
                            "smplx_model": str(full_eval_rerun_config.smplx_model),
                            "asset_roots": {
                                str(name): str(path)
                                for name, path in full_eval_rerun_config.asset_roots.items()
                            },
                            "record_fps": float(full_eval_rerun_config.record_fps),
                            "max_seconds": float(full_eval_rerun_config.max_seconds),
                            "fail_on_error": bool(full_eval_rerun_config.fail_on_error),
                            "stop_after_export": full_eval_rerun_only,
                        }
                    metrics_output_root = os.path.join(
                        dataset_output_root, "metrics", layout_name
                    )
                    if full_eval_rerun_only:
                        # Partial passes must never overwrite the complete summaries.
                        metrics_output_root = os.path.join(
                            rerun_only_scratch, frames_dir, dataset_name.lower(), layout_name
                        )
                    eval_model_func(
                        model=unwrapped_model,
                        dataloader=dataset_loaders[frames],
                        global_step=step,
                        split="test",
                        eval_num=eval_sample_num,
                        # Full evaluation normally keeps only compact summaries.
                        # Opt in to per-sample motion exports when downstream
                        # PA-MPJPE/MPJRE/MPJVE/MTE scoring needs decoded arrays.
                        save_sample=bool(
                            config.experiment.get("full_eval_save_samples", False)
                        ),
                        invalid_imu_id=invalid_ids,
                        postfix=f"{dataset_name}_{layout_name}_f{frames:03d}",
                        eval_output_root=metrics_output_root,
                        structured_output=True,
                        rerun_export_job=rerun_export_job,
                        eval_frames=int(frames),
                        cascade_motion_source=full_eval_cascade_motion,
                    )
        if full_eval_rerun_only and accelerator.is_main_process:
            shutil.rmtree(rerun_only_scratch, ignore_errors=True)
    ##################################
    #         Evaluation Only        #
    ##################################
    # After this block, we will exit the script
    if mode == "test":

        # delete optimizer and lr_scheduler
        optimizer = None
        lr_scheduler = None
        torch.cuda.empty_cache()

        shift_values = [0] if eval_selected_imu_seq is not None else [0, 2]

        start_time = time.time()
        for shift_value in shift_values:

            if eval_selected_imu_seq is not None:
                # Single external .pkl (README single-sequence inference): map-style, no shards.
                test_dataset_imu = def_dataset(
                    split="test",
                    selected_dataset=eval_selected_dataset,
                    selected_imu_seq=eval_selected_imu_seq,
                    shift=shift_value,
                    shuffle_list=False,
                    random_cut=False,
                    random_mask_text=False,
                    add_imu_noise=False,
                    IMUSEQMAXLEN=config.experiment.max_eval_imu_len,
                )
                test_seq_dataloader_imu = DataLoader(
                    test_dataset_imu,
                    batch_size=1,
                    sampler=SequentialSampler(test_dataset_imu),
                    collate_fn=test_dataset_imu.collate_fn,
                    shuffle=False,
                    num_workers=dataset_config.num_workers,
                )
                test_eval_num = len(test_seq_dataloader_imu)
                print(
                    f"Evaluating single IMU sequence {eval_selected_imu_seq} "
                    f"({test_eval_num} sample(s)), shift={shift_value} frames"
                )
            else:
                # Full dataset from the wds test shards (single-pass, deterministic).
                test_shards = list_wds_shards(motion_data_root, "test")
                assert len(test_shards) > 0, "No test shards found."
                test_rewritten_texts = load_rewrite_texts(
                    [motion_data_root], "test", enabled=use_rewrite_texts
                )
                test_seq_dataloader_imu = build_eval_wds_loader(
                    test_shards,
                    cut_length=config.experiment.max_eval_imu_len,
                    shift=shift_value,
                    target_datasets=eval_target,
                    split="test",
                    rewritten_texts=test_rewritten_texts,
                    require_rewritten_text=bool(use_rewrite_texts),
                    **_eval_kwargs,
                )
                # Upper bound; the stream ends early via dataset filtering / StopIteration.
                test_eval_num = min(
                    num_wds_samples(motion_data_root, "test"),
                    int(config.experiment.max_eval_sample_num),
                )
                print(
                    f"Evaluating on {eval_selected_dataset} test shards "
                    f"(<= {test_eval_num} samples), shifted {shift_value} frames"
                )

            for imu_config_name, invalid_ids in eval_imu_configs:
                postfix_parts = [imu_config_name, f"shifted_{shift_value}"]
                eval_model_func(
                    model=accelerator.unwrap_model(model),
                    eval_num=test_eval_num,
                    dataloader=test_seq_dataloader_imu,
                    global_step=0,
                    split="test",
                    postfix="_".join(part for part in postfix_parts if part),
                    job_id=job_id,
                    total_jobs=total_jobs,
                    invalid_imu_id=invalid_ids,
                    save_sample=config.experiment.save_test_sample,
                )
        accelerator.wait_for_everyone()
        accelerator.print(f"Evaluation done in {time.time() - start_time} seconds.")
        # copy the loaded checkpoint to the same dir as saved samples (output_dir)
        if config.experiment.save_ckpt_when_eval and accelerator.is_main_process:
            ckpt_dirs = os.listdir(config.experiment.ckpt_dir)
            ckpt_dirs = [d for d in ckpt_dirs if d.startswith("checkpoint")]
            ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
            if ckpt_dirs:
                ckpt_name = ckpt_dirs[-1]
                src_ckpt = os.path.join(config.experiment.ckpt_dir, ckpt_name)
                result_dir = Path(config.experiment.output_dir)
                # remove all existing checkpoints in output folder
                for d in os.listdir(result_dir):
                    if d.startswith("checkpoint"):
                        old_ckpt = result_dir / d
                        if old_ckpt.is_dir():
                            shutil.rmtree(old_ckpt)
                            accelerator.print(f"Removed existing checkpoint {old_ckpt}")
                dest_ckpt = result_dir / ckpt_name
                shutil.copytree(src_ckpt, dest_ckpt)
                accelerator.print(f"Copied checkpoint from {src_ckpt} to {dest_ckpt}")
        exit(0)

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Number of unique samples in train set = {num_train_samples}")
    if effective_num_train_samples != num_train_samples:
        logger.info(f"  Effective weighted samples per epoch = {effective_num_train_samples}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    step_samples = 0
    samples_last_step = config.training.gradient_accumulation_steps * total_batch_size_per_gpu
    last_cut_len = 0
    end = time.time()

    is_start_eval = config.experiment.is_start_eval
    is_start_sample = config.experiment.is_start_sample
    periodic_eval_every = int(config.experiment.eval_every)
    periodic_sample_every = int(config.experiment.sample_every)
    if periodic_eval_every < 0 or periodic_sample_every < 0:
        raise ValueError("experiment.eval_every and sample_every must be >= 0")

    print(f'total epoch = {num_train_epochs}')

    ##################################
    #          EMA (train only)      #
    ##################################
    # The EMA copy tracks only the *trainable* weights (frozen modules such as
    # the time-series quantizer are shared). It is advanced once per optimizer
    # step and is used for evals / sampling / the final released weights; the
    # live weights keep training and resume always continues from live weights.
    use_ema = mode == "train" and bool(config.experiment.get("use_ema", True))
    ema_decay = float(config.experiment.get("ema_decay", 0.999))
    ema = None
    ema_params = None

    @contextmanager
    def _ema_weights_for_eval(active=True):
        # Swap the live weights for the smoothed EMA copy while evaluating, then
        # always restore the live weights (training continuity). No-op when EMA
        # is disabled, in test mode, or when this step runs no evaluation (so we
        # never pay the full-model copy on ordinary training steps).
        if ema is None or not active:
            yield
            return
        ema.store(ema_params)
        ema.copy_to(ema_params)
        try:
            yield
        finally:
            ema.restore(ema_params)

    if use_ema:
        unwrapped_model_for_ema = accelerator.unwrap_model(model)
        ema_params = [p for p in unwrapped_model_for_ema.parameters() if p.requires_grad]
        if not ema_params:
            raise ValueError("use_ema=True but no trainable parameters found.")
        # The shadow lives on the accelerator. A CPU shadow (used for Show-o
        # until 2026-09-14 to save ~5.7 GB) cost ~6 s per optimizer step: the
        # per-tensor GPU->CPU copy plus the CPU lerp over 1.45 B fp32 values
        # dominated the 5-9 s step time. On-device with a fused foreach lerp
        # the same update takes ~8 ms. training.ema_device="cpu" restores the
        # old behaviour for a memory-starved run.
        ema_device = config.training.get("ema_device", None)
        ema = EMA(ema_params, decay=ema_decay, device=ema_device)
        # A full-state resume reloads the EMA shadow saved with the checkpoint.
        # Pre-EMA checkpoints have no ema_model.pt, so start the shadow from the
        # (already loaded) weights instead of the model initialization.
        if successful_resume and config.experiment.resume_from_checkpoint and not load_without_optimizer:
            resume_ckpt_dirs = sorted(
                (d for d in os.listdir(config.experiment.ckpt_dir) if d.startswith("checkpoint")),
                key=lambda x: int(x.split("-")[1]),
            )
            if resume_ckpt_dirs:
                ema_file = Path(config.experiment.ckpt_dir) / resume_ckpt_dirs[-1] / "ema_model.pt"
                if ema_file.is_file():
                    ema.load_state_dict(torch.load(ema_file, map_location="cpu"))
                    # Only an on-accelerator shadow needs moving back after the
                    # CPU-side load; showo keeps it off the GPU.
                    if ema_device is None:
                        ema.to(accelerator.device)
                    accelerator.print(f"Loaded EMA shadow from {ema_file}")
                else:
                    accelerator.print(
                        "No EMA state in checkpoint (pre-EMA run); initializing EMA from loaded weights."
                    )
        accelerator.print(
            f"EMA enabled: decay={ema_decay}, tracking {len(ema_params)} trainable "
            f"parameters on {ema_device or accelerator.device}."
        )

    export_ema_checkpoint = config.experiment.get('export_ema_checkpoint', None)
    if export_ema_checkpoint:
        if not successful_resume:
            raise ValueError("experiment.export_ema_checkpoint requires a resumed checkpoint")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with _ema_weights_for_eval():
                unwrapped_model = accelerator.unwrap_model(model)
                if lora_utils.has_lora(unwrapped_model):
                    state_dict = lora_utils.merged_state_dict(unwrapped_model)
                else:
                    state_dict = unwrapped_model.state_dict()
                out_path = Path(export_ema_checkpoint)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state_dict, out_path)
            accelerator.print(
                f"Exported EMA-merged checkpoint (step={global_step}, ema={ema is not None}) "
                f"to {export_ema_checkpoint}"
            )
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    if full_eval_only:
        if not full_eval_jobs:
            raise ValueError(
                "experiment.full_eval_only=True but no full-eval datasets are configured"
            )
        if not successful_resume:
            raise ValueError("experiment.full_eval_only=True requires a resumed checkpoint")
        logger.info(
            "Full evaluation only: step=%d, ema=%s, output=%s",
            global_step,
            ema is not None,
            os.path.join(config.experiment.output_dir, "evaluation", "full", f"step-{global_step:06d}"),
        )
        accelerator.wait_for_everyone()
        with _ema_weights_for_eval():
            model.eval()
            run_full_eval(accelerator.unwrap_model(model), global_step)
        accelerator.wait_for_everyone()
        accelerator.print(f"Full evaluation finished for step {global_step}.")
        accelerator.end_training()
        return

    if overfit and int(config.training.get("overfit_num_samples", 0)) > 0:
        model.eval()
        with _ema_weights_for_eval():
            for imu_config_name, invalid_ids in eval_imu_configs:
                eval_model_func(
                    model=accelerator.unwrap_model(model),
                    dataloader=val_seq_dataloader_imu,
                    global_step=global_step, split="val",
                    eval_num=config.training.overfit_num_samples,
                    save_sample=False, invalid_imu_id=invalid_ids,
                    postfix=imu_config_name or "",
                    eval_output_root=os.path.join(config.experiment.output_dir,
                        "evaluation", "periodic", f"step-{global_step:06d}"),
                )

    for epoch in range(first_epoch, first_epoch + num_train_epochs): #FIXME: set a more reasonable way to set the number of epochs

        # Publish one crop length before this epoch's WebDataset iterator starts.
        # Updating it per step races with worker prefetch and can make samples
        # within one batch use different temporal lengths. (No-op range refresh
        # when experiment.train_cut_per_batch is enabled.)
        set_cut_length_wds(
            shared_cut_imu,
            config.experiment.min_train_imu_len,
            config.experiment.max_train_imu_len,
        )
        model.train()
        # wds resamples shards every epoch; no DistributedSampler.set_epoch needed.
        # iterate over train_dataloader_imu
        # Time the wait for each batch so data_time reports the loader stall
        # (the CPU no longer syncs on the GPU every micro-batch, so a long
        # wait here means the workers are the bottleneck). Summed over the
        # micro-batches of one optimizer step, like batch_time.
        _train_iter = iter(train_dataloader_imu)
        _step_data_time = 0.0
        imu_batch_idx = -1
        while True:
            _fetch_start = time.time()
            try:
                imu_batches = next(_train_iter)
            except StopIteration:
                break
            _step_data_time += time.time() - _fetch_start
            imu_batch_idx += 1

            # Optional profiler window over micro-batches [a, b):
            # IMU4D_PROFILE=a:b writes a Chrome trace + a key_averages table to
            # experiment.output_dir/profile/. Diagnostic only; off by default.
            _prof_spec = os.environ.get("IMU4D_PROFILE", "")
            if _prof_spec and accelerator.is_main_process:
                _prof_a, _prof_b = (int(x) for x in _prof_spec.split(":"))
                if imu_batch_idx == _prof_a:
                    _profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    )
                    _profiler.__enter__()
                    _prof_wall_start = time.time()
                elif imu_batch_idx == _prof_b:
                    torch.cuda.synchronize()
                    _prof_wall = time.time() - _prof_wall_start
                    _profiler.__exit__(None, None, None)
                    _prof_dir = os.path.join(config.experiment.output_dir, "profile")
                    os.makedirs(_prof_dir, exist_ok=True)
                    _profiler.export_chrome_trace(os.path.join(_prof_dir, f"trace_{_prof_a}_{_prof_b}.json"))
                    with open(os.path.join(_prof_dir, f"key_averages_{_prof_a}_{_prof_b}.txt"), "w") as _f:
                        _f.write(f"wall seconds over {_prof_b - _prof_a} micro-batches: {_prof_wall:.3f}\n\n")
                        _f.write(_profiler.key_averages().table(sort_by="cuda_time_total", row_limit=60))
                        _f.write("\n\n")
                        _f.write(_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=60))
                    logger.info("Profiler window written to %s", _prof_dir)
                    if os.environ.get("IMU4D_PROFILE_ONLY", "0") == "1":
                        accelerator.end_training()
                        return
            # Per-batch crop buckets make the batch size vary; track actual samples
            # for throughput logging.
            step_samples += len(imu_batches)
            last_cut_len = len(imu_batches[0]["imu_data"]) if len(imu_batches) and len(imu_batches[0]) else 0

            with accelerator.accumulate(model):
                input_dict = \
                    imu_to_input(
                        model,
                        time_series_quantizer,
                        imu_batches,
                        accelerator,
                        uni_prompting,
                        mask_dtype,
                        obj_name_to_id,
                        object_token_bias,
                        identity_catalog,
                        normalization_window_size=config.model.normalization_window_size,
                        smooth_imu=config.model.smooth_imu,
                        invalid_imu_id=train_invalid_imu_id,
                        imu_layout_sampler=train_imu_layout_sampler,
                        random_text=not overfit,
                        dynamic_object=dynamic_object,
                        bidirectional_imu=bidirectional_imu,
                        bidirectional_motion=bidirectional_motion,
                        predict_objects=config.model.get('predict_objects', True),
                        fps=config.dataset.params.fps,
                    )
                all_embeddings = input_dict['seq_all_embeddings']
                input_embeddings = input_dict['imu_embeddings_batch']
                query_embeddings = input_dict['query_embeddings_batch']
                labels = input_dict['labels']
                attention_mask = input_dict['seq_attention_mask']
                input_imu_len = input_dict['input_imu_len_batch']

                loss_dict = model(
                    # casual
                    input_embeddings=all_embeddings, # [batch_size, length, d_model]
                    # bidirectional motion
                    # imu_embeddings=input_embeddings, # [batch_size, length, d_model]
                    query_embeddings=query_embeddings, # [batch_size, length, d_model]
                    # shared
                    attention_mask=attention_mask, # [batch_size, 1, length, length]
                    labels=labels, # [batch_size, length]
                    batch_size_imu=1,
                    input_imu_len=input_imu_len,  # [batch_size]
                )

                total_loss = loss_dict["total_loss"]

                # Gather the losses across processes for logging, but only on
                # the micro-batch that completes a logged optimizer step. Doing
                # it on every micro-batch cost ~20 gather + .item() host syncs
                # (plus a print) per forward, which kept the CPU from running
                # ahead into the next batch's Python-heavy input preparation.
                # One stacked tensor -> one gather -> one device-to-host copy.
                _log_this_step = accelerator.sync_gradients and (
                    (global_step + 1) % config.experiment.log_every == 0
                )
                if _log_this_step:
                    _metric_names = [
                        "total_loss", "loss_mean", "loss_std", "loss_static", "loss_recon",
                        "loss_text", "loss_object_token", "loss_object_pose",
                        "loss_obj_dynamic_pose", "loss_object_anchor_flow",
                        "loss_object_track_state", "obj_track_state_accuracy",
                        "loss_object_identity", "mean_accuracy", "std_accuracy",
                        "static_accuracy", "obj_id_accuracy", "object_identity_accuracy",
                        "text_accuracy",
                    ]
                    _stacked = torch.stack([
                        loss_dict[name].detach().float().reshape(()) for name in _metric_names
                    ])  # [n_metrics]
                    _gathered = accelerator.gather(_stacked[None]).mean(dim=0).cpu()
                    if getattr(accelerator.unwrap_model(model), 'object_set_classifier', None) is not None:
                        logger.info(
                            f"step {global_step + 1}: loss_object_set "
                            f"{loss_dict['loss_object_set'].item():.4f} object_set_f1 "
                            f"{loss_dict['object_set_f1'].item():.3f} (local rank)"
                        )
                    (
                        total_loss_global, loss_mean_global, loss_std_global,
                        loss_static_global, loss_recon_global, loss_text_global,
                        loss_object_token_global, loss_object_pose_global,
                        loss_obj_dynamic_pose_global, loss_object_anchor_flow_global,
                        loss_object_track_state_global, obj_track_state_accuracy_global,
                        loss_object_identity_global, mean_accuracy_global,
                        std_accuracy_global, static_accuracy_global,
                        obj_id_accuracy_global, object_identity_accuracy_global,
                        text_accuracy_global,
                    ) = _gathered.unbind(0)
                    # obj_pose_accuracy / obj_dynamic_pose_accuracy are not gathered (historical).
                    obj_pose_accuracy_global = torch.tensor(0.0)
                    obj_dynamic_pose_accuracy_global = torch.tensor(0.0)

                if _log_this_step and accelerator.is_main_process:
                    print('epoch = {}, step = {}, total_loss = {:.2e}, loss_mean = {:.2e}, loss_std = {:.2e}, loss_static_status = {:.2e}, loss_recon = {:.2e}, loss_text = {:.2e}, loss_object_token = {:.2e}, loss_object_pose = {:.2e}, loss_obj_dynamic_pose = {:.2e}, loss_anchor_flow = {:.2e}, mean_acc = {:.4f}, std_acc = {:.4f}, static_acc = {:.4f}, text_acc = {:.2f}, obj_id_acc = {:.4f}, obj_pose_acc = {:.4f}, obj_dynamic_pose_acc = {:.4f}, len = {}'.format(
                        epoch,
                        global_step,
                        total_loss_global.item(),
                        loss_mean_global.item(),
                        loss_std_global.item(),
                        loss_static_global.item(),
                        loss_recon_global.item(),
                        loss_text_global.item(),
                        loss_object_token_global.item(),
                        loss_object_pose_global.item(),
                        loss_obj_dynamic_pose_global.item(),
                        loss_object_anchor_flow_global.item(),
                        mean_accuracy_global.item(),
                        std_accuracy_global.item(),
                        static_accuracy_global.item(),
                        text_accuracy_global.item(),
                        obj_id_accuracy_global.item(),
                        obj_pose_accuracy_global.item(),
                        obj_dynamic_pose_accuracy_global.item(),
                        loss_dict["length"]
                    ))

                accelerator.backward(total_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                # set grad to zero
                embedding_weight_grad = accelerator.unwrap_model(model).embed_tokens.weight.grad
                if embedding_weight_grad is not None:
                    embedding_weight_grad[:old_embeddings_len].zero_() # Set gradients for the original embeddings to zero

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                # if (
                #         accelerator.sync_gradients
                #         and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                #         and accelerator.is_main_process
                # ):
                #     log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                data_time_m.update(_step_data_time)
                _step_data_time = 0.0
                end = time.time()
                samples_last_step, step_samples = step_samples, 0

                # Advance the EMA copy after this optimizer step, before any
                # eval/sample runs on the smoothed weights.
                if ema is not None:
                    ema.step(ema_params)

                _run_sample_now = periodic_sample_every > 0 and (
                    global_step == 0 or (global_step + 1) % periodic_sample_every == 0
                ) or is_start_sample
                _run_eval_now = periodic_eval_every > 0 and (
                    global_step == 0 or (global_step + 1) % periodic_eval_every == 0
                ) or is_start_eval
                if os.environ.get("IMU4D_PROFILE_ONLY", "0") == "1" and _prof_spec:
                    _run_sample_now = _run_eval_now = False
                with _ema_weights_for_eval(active=_run_sample_now or _run_eval_now):
                    if _run_sample_now:
                        accelerator.wait_for_everyone()
                        is_start_sample = False
                        model.eval()
                        if not overfit:
                            for imu_config_name, invalid_ids in eval_imu_configs:
                                eval_model_func(
                                    model=accelerator.unwrap_model(model),
                                    dataloader=val_seq_dataloader_imu,
                                    global_step=global_step + 1,
                                    split="val",
                                    eval_num=3,
                                    save_sample=True,
                                    invalid_imu_id=invalid_ids,
                                    generate_number=False,
                                    postfix=imu_config_name or "",
                                    eval_output_root=os.path.join(
                                        config.experiment.output_dir,
                                        "evaluation",
                                        "periodic",
                                        f"step-{global_step + 1:06d}",
                                    ),
                                )
                        eval_model_func(
                            model=accelerator.unwrap_model(model),
                            dataloader=train_seq_dataloader_imu,
                            global_step=global_step + 1,
                            split="train",
                            eval_num=3,
                            save_sample=True,
                            invalid_imu_id=train_invalid_imu_id,
                            generate_number=False,
                        )
                        # (wds train loader manages its own cut via shared_cut_imu; the
                        # map-style enable_random_cut re-toggle is no longer needed.)
                        model.train()
                        accelerator.wait_for_everyone()

                    # generate numbers for train and val set
                    if _run_eval_now:
                        accelerator.wait_for_everyone()
                        is_start_eval = False
                        model.eval()
                        for imu_config_name, invalid_ids in eval_imu_configs:
                            eval_model_func(
                                model=accelerator.unwrap_model(model),
                                dataloader=val_seq_dataloader_imu,
                                global_step=global_step + 1,
                                split="val",
                                eval_num=config.experiment.max_eval_sample_num,
                                save_sample=False,
                                invalid_imu_id=invalid_ids,
                                postfix=imu_config_name or "",
                                eval_output_root=os.path.join(
                                    config.experiment.output_dir,
                                    "evaluation",
                                    "periodic",
                                    f"step-{global_step + 1:06d}",
                                ),
                            )
                        # Teacher-forced validation loss: the only signal here
                        # that is comparable to the training losses, hence the
                        # one to early-stop on.
                        _tf_val = teacher_forced_val(global_step + 1)
                        if _tf_val and accelerator.is_main_process:
                            logger.info(
                                "teacher-forced val @ step %d: %s",
                                global_step + 1,
                                ", ".join(f"{k}={v:.4f}" for k, v in sorted(_tf_val.items())),
                            )
                            accelerator.log(_tf_val, step=global_step + 1)
                        model.train()
                        accelerator.wait_for_everyone()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = samples_last_step / batch_time_m.val
                    logs = {
                        "epoch": epoch,
                        "step_total_loss": total_loss_global.item(),
                        "step_loss_mean": loss_mean_global.item(),
                        "step_loss_std": loss_std_global.item(),
                        "step_loss_static": loss_static_global.item(),
                        "step_loss_recon": loss_recon_global.item(),
                        "step_loss_text": loss_text_global.item(),
                        "step_loss_object_token": loss_object_token_global.item(),
                        "step_loss_object_pose": loss_object_pose_global.item(),
                        "step_loss_obj_dynamic_pose": loss_obj_dynamic_pose_global.item(),
                        "step_loss_object_anchor_flow": loss_object_anchor_flow_global.item(),
                        "step_loss_object_track_state": loss_object_track_state_global.item(),
                        "step_obj_track_state_accuracy": obj_track_state_accuracy_global.item(),
                        "step_loss_object_identity": loss_object_identity_global.item(),
                        "step_object_identity_accuracy": object_identity_accuracy_global.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "train/samples_per_step": samples_last_step,
                        "train/cut_len": last_cut_len,
                        "train/batch_size": len(imu_batches),
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {total_loss_global.item():0.4f} "
                        f"Loss_mean: {loss_mean_global.item():0.4f} "
                        f"Loss_std: {loss_std_global.item():0.4f} "
                        f"Loss_static: {loss_static_global.item():0.4f} "
                        f"Loss_text: {loss_text_global.item():0.4f} "
                        f"Loss_object_token: {loss_object_token_global.item():0.4f} "
                        f"Loss_object_pose: {loss_object_pose_global.item():0.4f} "
                        f"Loss_object_identity: {loss_object_identity_global.item():0.4f} "
                        f"Loss_obj_dynamic_pose: {loss_obj_dynamic_pose_global.item():0.4f} "
                        f"Loss_anchor_flow: {loss_object_anchor_flow_global.item():0.4f} "
                        f"Loss_track_state: {loss_object_track_state_global.item():0.4f} "
                        f"State_acc: {obj_track_state_accuracy_global.item():0.4f} "
                        f"Identity_acc: {object_identity_accuracy_global.item():0.4f} "
                        f"{_modality_grad_log(model)}"
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    # save_checkpoint(model, config, accelerator, global_step + 1)
                    save_checkpoint(model, optimizer, lr_scheduler, config, accelerator, global_step + 1, epoch=epoch, ema=ema)
                    print(f"Saving checkpoint at step {global_step + 1}")
                    accelerator.wait_for_everyone()

                # Run the long benchmark after checkpointing, so an evaluation
                # failure never loses the corresponding trained state.
                if full_eval_every > 0 and (global_step + 1) % full_eval_every == 0:
                    accelerator.wait_for_everyone()
                    with _ema_weights_for_eval():
                        model.eval()
                        run_full_eval(accelerator.unwrap_model(model), global_step + 1)
                        model.train()
                    accelerator.wait_for_everyone()

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
        # The inner break only ends the epoch; stop the epoch loop as well so a
        # large num_train_epochs cannot run past max_train_steps.
        if global_step >= config.training.max_train_steps:
            break

    accelerator.wait_for_everyone()

    # Ensure the final checkpoint has a complete LINGO evaluation. Avoid doing
    # it twice when max_train_steps is itself a full-eval boundary.
    if full_eval_every > 0 and global_step % full_eval_every != 0:
        with _ema_weights_for_eval():
            model.eval()
            run_full_eval(accelerator.unwrap_model(model), global_step)
        accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, optimizer, lr_scheduler, config, accelerator, global_step, epoch=epoch, ema=ema)

    # Save the final trained checkpoint. When EMA is enabled, release the
    # smoothed weights (the live weights already went into the checkpoints).
    if accelerator.is_main_process:
        if ema is not None:
            ema.copy_to(ema_params)
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()

def last_dim_logit_to_index(logits, temperature=1.0, top_k:int=None):
    """
    logits: arbitrary shape with last dim = vocab_size, e.g. [bs, len, vocab_size] or [bs, len, n, vocab_size]
    Returns: same shape as input except the last dim (vocab_size), i.e. sampled token indices.

    top_k is a sampling constraint — before we sample from the probability distribution, we keep only the k highest-probability tokens in each row of the logits and mask all the others with -inf.
    That means:
    If top_k = 1, this becomes greedy sampling (always take the max).
    If top_k = vocab_size or None, this is just normal sampling over the whole vocabulary.
    If top_k = 5, only the top 5 logits for each position are kept; the rest are excluded from sampling
    """
    orig_shape = logits.shape[:-1]  # all dims except vocab_size
    logits = logits.reshape(-1, logits.size(-1))  # [N, vocab_size]
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits[logits < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    idx = torch.multinomial(probs, num_samples=1)
    idx = idx.reshape(orig_shape)  # restore to input shape (without last dim)
    return idx

def topk_accuracy(logits, targets, k:int=1):
    """
    logits: [bs, len_pred, vocab_size]
    targets: [bs, len_gt]
    k: top-k accuracy to compute
    Returns: scalar accuracy
    """
    bs, len_pred, _ = logits.shape
    _, len_gt = targets.shape

    min_len = min(len_pred, len_gt)

    # Cut to min length
    logits_cut = logits[:, :min_len]
    targets_cut = targets[:, :min_len]

    # Get top-k predicted indices along vocab dimension
    topk_indices = torch.topk(logits_cut, k, dim=-1).indices  # [bs, min_len, k]

    # Check if target is in top-k predictions
    correct_cut = (topk_indices == targets_cut.unsqueeze(-1)).any(dim=-1)  # [bs, min_len]
    valid_cut = targets_cut != -100  # ignored labels (masked channel groups) do not count

    if len_pred < len_gt:
        # Pad with False (incorrect) for missing predictions
        pad_len = len_gt - len_pred
        pad_false = torch.zeros(bs, pad_len, dtype=torch.bool, device=logits.device)
        correct = torch.cat([correct_cut, pad_false], dim=1)
        valid = torch.cat([valid_cut, torch.ones(bs, pad_len, dtype=torch.bool, device=logits.device)], dim=1)
    else:
        correct = correct_cut  # Already cut to target length
        valid = valid_cut

    # Compute mean accuracy over non-ignored targets
    if valid.any():
        return correct[valid].float().mean()
    return torch.zeros((), device=logits.device)

def compute_perplexity(logits, indices):
    """
    Compute perplexity for a sequence of predictions.

    Args:
        logits: List of logit tensors, each of shape [n_var, n_bins]
        indices: List of index tensors, each of shape [n_var]

    Returns:
        perplexity: Scalar tensor
    """
    total_log_prob = 0.0
    total_tokens = 0

    for logit, idx in zip(logits, indices):
        # Compute log probabilities
        log_probs = F.log_softmax(logit, dim=-1)
        # Get log prob of selected indices
        selected_log_probs = log_probs.gather(1, idx.unsqueeze(1)).squeeze(1)
        total_log_prob += selected_log_probs.sum()
        total_tokens += idx.numel()

    # Perplexity = exp(-average log probability)
    avg_log_prob = total_log_prob / total_tokens
    perplexity = torch.exp(-avg_log_prob)
    return perplexity



def _drift_world_yaw(t: torch.Tensor, sigma_end_deg: float, rng: np.random.Generator) -> torch.Tensor:
    """Per-sensor world-yaw random walk of the attitude estimate (evaluation probe).

    Mirrors ``imu_synthesis.imu_noise``'s ``yaw_drift``: an independent random
    walk ``phi_s(t)`` per sensor left-multiplying the world-frame channels, which
    is what a gyro-integrated heading without a magnetometer does.  Unlike the
    constant ``eval_input_yaw_deg`` offset (and unlike the per-device conjugation
    a T-pose calibration leaves behind), this one is *time varying*, so neither a
    heading-conjugation augmentation nor a single self-calibrated angle per
    device can absorb it.  ``sigma_end_deg`` is the std of the walk at the end of
    the window; 0 disables.  Used to test how much of the within-window error
    ramp is attributable to heading drift.
    """
    n_time, n_sensor = t.shape[0], t.shape[1]
    step = math.radians(float(sigma_end_deg)) / math.sqrt(max(n_time, 1))
    phi = np.cumsum(rng.normal(0.0, step, size=(n_time, n_sensor)), axis=0)
    c = torch.tensor(np.cos(phi), dtype=t.dtype, device=t.device)
    s_ = torch.tensor(np.sin(phi), dtype=t.dtype, device=t.device)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    Ry = torch.stack([
        torch.stack([c, zero, s_], dim=-1),
        torch.stack([zero, one, zero], dim=-1),
        torch.stack([-s_, zero, c], dim=-1),
    ], dim=-2)  # [T, S, 3, 3]
    out = t.clone()
    out[..., :3] = (Ry @ t[..., :3].unsqueeze(-1)).squeeze(-1)
    out[..., 3:6] = (Ry @ t[..., 3:6].unsqueeze(-1)).squeeze(-1)
    out[..., 6:15] = (Ry @ t[..., 6:15].reshape(n_time, n_sensor, 3, 3)).reshape(n_time, n_sensor, 9)
    return out


def _rotate_world_yaw(t: torch.Tensor, yaw_rad: float) -> torch.Tensor:
    """Rotate world-frame IMU channels about the world vertical (+Y) axis.

    The IMU channel layout is [specific-force(3), angular-velocity(3),
    flattened world orientation matrix(9)]. Rotating the world coordinate
    frame premultiplies every world vector/matrix by R_y; gravity (-Y) is
    invariant, so the accelerometer channel stays physically consistent.
    """
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    Ry = torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=t.dtype,
        device=t.device,
    )
    acc = (Ry @ t[..., :3].unsqueeze(-1)).squeeze(-1)
    gyr = (Ry @ t[..., 3:6].unsqueeze(-1)).squeeze(-1)
    rot = t[..., 6:15].reshape(*t.shape[:-1], 3, 3)
    rot = Ry @ rot
    return torch.cat([acc, gyr, rot.reshape(*t.shape[:-1], 9)], dim=-1)


def load_text_diverse(config) -> dict | None:
    """experiment.eval_text_diverse_*: sample several captions per clip (0 = off).

    refs_file is JSON {sample_id: {ref_name: [caption, ...]}}; those captions are
    scored teacher-forced from the same prefix, so the model's likelihood of a
    human caption can be read against its own samples.
    """
    num = int(config.experiment.get("eval_text_diverse_num", 0) or 0)
    if num <= 0:
        return None
    refs_file = config.experiment.get("eval_text_diverse_refs_file", None)
    refs = {}
    if refs_file not in (None, "", "None"):
        with open(refs_file, "r", encoding="utf-8") as f:
            refs = json.load(f)
    return {
        "num": num,
        "temperature": float(config.experiment.get("eval_text_diverse_temperature", 1.0)),
        "top_p": float(config.experiment.get("eval_text_diverse_top_p", 1.0)),
        "seed": int(config.experiment.get("eval_text_diverse_seed", 0)),
        "refs": refs,
    }


def load_scene_sample(config) -> dict | None:
    """experiment.eval_scene_sample_*: one stochastic scene decode per eval call.

    With a seed set, the caption is sampled (temperature / nucleus) instead of
    greedy, every object's instance is drawn from the identity head instead
    of its argmax, and the anchor flow (first-frame layout) draws its noise from
    the same per-sample generator instead of the fixed seed 0. Motion stays
    greedy, so the body is the same for every seed. The object-category tokens
    stay greedy (following the sampled caption) unless ``objects`` is on.
    ``temperature`` / ``top_p`` apply to the caption; ``object_temperature`` to
    the category tokens and the identity head. None = off.
    """
    seed = config.experiment.get("eval_scene_sample_seed", None)
    if seed in (None, "", "None"):
        return None
    return {
        "seed": int(seed),
        "temperature": float(config.experiment.get("eval_scene_sample_temperature", 1.0)),
        "top_p": float(config.experiment.get("eval_scene_sample_top_p", 1.0)),
        "objects": bool(config.experiment.get("eval_scene_sample_objects", False)),
        "object_temperature": float(config.experiment.get("eval_scene_sample_object_temperature", 1.0)),
        # Forced diversity (seed k -> rank k+1): the (k+1)-th most likely asset in
        # each category, and the (k+1)-th farthest-point pick from a fixed pool
        # of whole-scene anchor layouts (0 = off), with widened flow noise.
        "identity_rank": bool(config.experiment.get("eval_scene_sample_identity_rank", False)),
        "layout_candidates": int(config.experiment.get("eval_scene_sample_layout_candidates", 0) or 0),
        "layout_noise": float(config.experiment.get("eval_scene_sample_layout_noise", 1.0)),
    }


def farthest_layout(model, object_hidden, keep, rank, num_candidates, noise_scale):
    """The ``rank``-th farthest-point pick among ``num_candidates`` anchor layouts [N, 9].

    The pool is drawn with a fixed seed, so every rank indexes the same pool;
    rank 0 is the candidate nearest the pool mean (the typical layout), later
    ranks are pushed as far as possible from all earlier ones. Distances use the
    translations (plus a quarter-weighted 6D rotation) of the ``keep`` objects
    (non-ground).
    """
    generator = torch.Generator(device=object_hidden.device)
    generator.manual_seed(0)
    n = len(object_hidden)
    pool = model.sample_object_anchor(
        object_hidden.repeat(num_candidates, 1), generator=generator, noise_scale=noise_scale
    ).reshape(num_candidates, n, -1)
    kept = pool[:, keep] if keep.any() else pool
    feat = torch.cat([kept[..., 6:9], 0.25 * kept[..., 0:6]], dim=-1).reshape(num_candidates, -1).float()
    order = [int(torch.linalg.norm(feat - feat.mean(0), dim=-1).argmin())]
    dist = torch.linalg.norm(feat - feat[order[0]], dim=-1)
    for _ in range(rank):
        order.append(int(dist.argmax()))
        dist = torch.minimum(dist, torch.linalg.norm(feat - feat[order[-1]], dim=-1))
    return pool[order[rank]]


def sample_from_logits(logits, temperature=1.0, top_p=1.0, generator=None):
    """Temperature / nucleus sampling over the last dim of ``logits`` [bs, V] -> [bs, 1]."""
    probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    if top_p < 1.0:
        sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
        drop = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
    return torch.multinomial(probs / probs.sum(dim=-1, keepdim=True), 1, generator=generator)


def decode_text_candidates(
    model, past_key_values, hidden_state, attention_mask, cache_length,
    invalid_positions, eot_token, max_new_tokens, num_samples=1,
    temperature=1.0, top_p=1.0, generator=None, forced=None,
):
    """Batched caption decode from one prefix KV cache; the cache is not modified.

    Samples ``num_samples`` captions (temperature / nucleus), or scores the
    token lists in ``forced`` teacher-forced. Returns one (tokens, logprobs) per
    row, <|eot|> included; logprobs are always under the untempered T=1
    distribution, i.e. the model's own likelihood of the caption.
    """
    batch = num_samples if forced is None else len(forced)
    if isinstance(past_key_values, tuple):
        cache = tuple(tuple(t.repeat_interleave(batch, dim=0) for t in layer) for layer in past_key_values)
    else:
        cache = copy.deepcopy(past_key_values)
        cache.batch_repeat_interleave(batch)
    hidden = hidden_state.expand(batch, -1)
    n_steps = max_new_tokens if forced is None else max(len(f) for f in forced)
    tokens = [[] for _ in range(batch)]
    logprobs = [[] for _ in range(batch)]
    done = [False] * batch
    for it in range(n_steps):
        logits = model.showo.lm_head(hidden).float()  # [B, V]
        logp = torch.log_softmax(logits, dim=-1)
        if forced is None:
            scaled = logits / max(temperature, 1e-6)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                drop = cum - torch.softmax(sorted_logits, dim=-1) >= top_p  # keep the token that crosses top_p
                sorted_logits[drop] = -float("inf")
                scaled = torch.full_like(scaled, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
            idx = torch.multinomial(torch.softmax(scaled, dim=-1), 1, generator=generator)[:, 0]
        else:
            idx = torch.tensor(
                [f[it] if it < len(f) else eot_token for f in forced],
                device=logits.device, dtype=torch.long,
            )
        idx_lp = logp.gather(-1, idx[:, None])[:, 0]
        for b in range(batch):
            if done[b]:
                continue
            tokens[b].append(int(idx[b]))
            logprobs[b].append(float(idx_lp[b]))
            if int(idx[b]) == eot_token or (forced is not None and it + 1 >= len(forced[b])):
                done[b] = True
        if all(done) or it + 1 == n_steps:
            break
        cache_length += 1
        step_mask = cached_decode_attention_mask(attention_mask, cache_length, invalid_positions)
        step_mask = step_mask.expand(batch, *step_mask.shape[1:])
        hidden_states, cache = model.backbone_forward_cached(
            model.embed_tokens(idx[:, None]), step_mask, past_key_values=cache,
        )
        hidden = hidden_states[:, -1]
    return list(zip(tokens, logprobs))


def load_forced_decode(path, variant) -> dict | None:
    """experiment.eval_forced_decode_file: JSON {sample_id: {variant: {caption, objects}}}.

    Returns {sample_id: {caption, objects}} for the chosen variant; samples
    without that variant decode normally.
    """
    if path in (None, "", "None") or variant in (None, "", "None"):
        return None
    table = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    forced = {
        sample_id: variants[str(variant)]
        for sample_id, variants in table.items()
        if variants.get(str(variant)) is not None
    }
    if not forced:
        raise ValueError(f"eval_forced_decode_file {path} has no variant {variant!r}")
    logger.info("Forced decode (variant %s): %s", variant, forced)
    return forced


@torch.inference_mode()
def eval_model(
    model,
    time_series_quantizer,
    dataloader,
    uni_prompting,
    object_token_id_to_name,
    obj_name_to_id,
    object_token_bias,
    identity_catalog,
    accelerator,
    config,
    global_step,
    temperature=1.0,
    top_k:int=None,
    split: str = "val",
    generate_number: bool = False,
    invalid_imu_id: list[int]=None,
    postfix: str = "",
    dynamic_object: bool=False,
    motion_only: bool=False,
    text_only: bool=False,
    # Motion is teacher-forced ground truth rather than predicted (M2T/M2S/M2TS).
    gt_motion_input: bool=False,
    # Motion span is learned queries, neither given nor scored (uncond gen).
    uncond_motion_input: bool=False,
    decode_objects: bool=True,
    total_jobs=1,
    job_id=0,
    bidirectional_imu: bool=False,
    bidirectional_motion: bool=False,
    fps: int=None,
    save_sample: bool=False,
    eval_num: int=None,
    eval_full_dataset: bool=False,
    max_sample_keep_num: int=5,
    eval_output_root: str | None = None,
    structured_output: bool = False,
    rerun_export_job: dict | None = None,
    # Evaluation window length in frames (full eval only; recorded in summaries
    # and Rerun metadata so 60- and 480-frame passes stay distinguishable).
    eval_frames: int | None = None,
    # Cascade input: an evaluation.cascade_motion.CascadeMotionSource whose
    # predicted motion replaces the ground truth a motion-input profile would
    # otherwise read (i2m -> m2t). None keeps the ground-truth motion.
    cascade_motion_source=None,

    # Sample a caption after the motion. False feeds the empty caption span a
    # profile without a text loss trains on (<|sot|><|eot|>) instead.
    decode_text: bool = True,
    max_new_text_tokens: int=60,
    max_object_id_tokens: int=1,
    min_object_id_tokens: int=1,
    # Eval-only oracle: force the object-category tokens instead of decoding
    # them (evaluation/object_list_override.py). None decodes as usual.
    object_list_override: str | None = None,
    object_multilabel_threshold: float = 0.1,
    # Eval-only oracle: feed the GT caption tokens (then <|eot|>) instead of
    # sampling the caption, so objects are decoded conditioned on GT text.
    teacher_force_text: bool = False,
    # Feed <|soobj|> after the caption before decoding object ids, as in
    # training. False reproduces the legacy decode (first id off <|eot|>).
    feed_soobj: bool = True,
    # object_list_override='pred_text_verified': minimum object-head probability
    # for a category named by the model's own caption to be forced.
    caption_verify_threshold: float = 0.05,
    # Eval-only counterfactual: {sample_id: {"caption": str, "objects": [category, ...]}}.
    # A listed sample's caption and/or object categories are forced instead of
    # decoded; motion is decoded before both, so only the object heads react.
    forced_decode: dict | None = None,
    # Eval-only: load_text_diverse() settings; sample extra captions per clip
    # (plus their likelihood) into records' text_diverse. None = greedy only.
    text_diverse: dict | None = None,
    # Eval-only: load_scene_sample() settings; sampled caption + identity. None = greedy.
    scene_sample: dict | None = None,
):
    logger.info("Generating ...")

    assert fps is not None, "FPS must be specified for evaluation."
    assert invalid_imu_id is not None, "invalid_imu_id must be specified"

    logger.info('########################################################')
    logger.info("Start evaluation ...")
    logger.info(f"eval_num = {eval_num}")
    logger.info(f"invalid_imu_id = {invalid_imu_id}")
    if invalid_imu_id != "random":
        active_sensor_names = [
            sensor_name
            for sensor_id, sensor_name in enumerate(IMU_SENSOR_NAMES)
            if sensor_id not in invalid_imu_id
        ]
        logger.info(
            f"active IMUs ({len(active_sensor_names)}pt) = {active_sensor_names}"
        )
    logger.info(f"bidirectional_imu = {bidirectional_imu}")
    logger.info(f"bidirectional_motion = {bidirectional_motion}")
    logger.info(f"text_only = {text_only}")
    logger.info(f"gt_motion_input = {gt_motion_input}")
    logger.info(f"uncond_motion_input = {uncond_motion_input}")
    # Three different reasons the motion span is never rolled out: text-only
    # supervision, teacher-forced ground truth, and unconditional generation.
    # Everything downstream that asks "is there a motion prediction to work
    # with" means this, not gt_motion_input alone.
    motion_not_decoded = text_only or gt_motion_input or uncond_motion_input
    logger.info(f"decode_objects = {decode_objects}")
    logger.info(f"decode_text = {decode_text}")
    logger.info(f"fps = {fps}")
    logger.info(f"max_new_text_tokens = {max_new_text_tokens}")
    logger.info(f"max_object_id_tokens = {max_object_id_tokens}")
    logger.info(f"min_object_id_tokens = {min_object_id_tokens}")
    if eval_frames is not None:
        logger.info(f"eval_frames = {eval_frames}")
    logger.info('########################################################')

    if text_only and rerun_export_job is not None:
        raise ValueError(
            "Rerun export needs decoded motion, which text_only evaluation never produces"
        )
    if gt_motion_input and bidirectional_motion:
        raise ValueError(
            "Teacher-forced motion evaluation needs showo.bidirectional_motion=False"
        )

    if eval_full_dataset:
        eval_num = len(dataloader)
    else:
        assert eval_num is not None, "eval_num must be specified if eval_full_dataset is False"

    if hasattr(model, 'module'):
        mask_dtype = model.module.embed_tokens.weight.dtype
    else:
        mask_dtype = model.embed_tokens.weight.dtype

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    eot_token = uni_prompting.sptids_dict['<|eot|>'].item()

    static_embedder = time_series_quantizer.static_embedder
    mean_embedder = time_series_quantizer.mean_embedder
    std_embedder = time_series_quantizer.std_embedder
    time_embedder = time_series_quantizer.time_embedder

    n_dynamic_bins = model.time_series_dynamic_vocab_size
    n_static_bins = model.time_series_static_vocab_size

    # status_static = torch.cat([traj_static, orient_static, pose_static], dim=2)  # [n_token, n_var]
    # status_mean = torch.cat([traj_mean, orient_mean, pose_mean], dim=2)
    # status_std = torch.cat([traj_std, orient_std, pose_std], dim=2)
    # status_static_emb = torch.cat([traj_static_emb, orient_static_emb, pose_static_emb], dim=2)
    # status_dynamic_emb = torch.cat([traj_dynamic_emb, orient_dynamic_emb, pose_dynamic_emb], dim=2)
    # status_embeddings = model.status_aggregator(status_static_emb[0], status_dynamic_emb[0])

    accumulate = config.model.accumulate
    accumulate_orient = config.model.accumulate_orient
    accumulate_pose = config.model.accumulate_pose
    local_coordinate = config.model.local_coordinate

    pose_filter_config = config.experiment.get("eval_pose_filter", {}) or {}
    pose_filter_settings = {
        "enabled": bool(pose_filter_config.get("enabled", True)) and not motion_not_decoded,
        "cutoff_hz": float(pose_filter_config.get("cutoff_hz", 3.0)),
        "order": int(pose_filter_config.get("order", 4)),
        "method": "butterworth_quaternion_zero_phase",
    }
    logger.info("Evaluation pose filter: %s", pose_filter_settings)

    traj_errors = []
    orient_errors = []
    pose_errors = []

    status_top1_accs = []
    status_top5_accs = []
    status_ces = [] # cross-entropy

    text_top1_accs = []
    text_top5_accs = []
    text_ces = [] # cross-entropy

    mpjpes = []
    object_id_accuracies = []
    object_rot_l1_errors = []
    object_transl_l1_errors = []
    # One entry per configured sample id: matched ids, exported ids, errors.
    rerun_status = {
        "matched": [],
        "exported": [],
        "errors": {},
    }

    # Map-style datasets expose disable_random_cut(); the wds eval loader does not
    # (it is already built with random_cut=False), so guard the call.
    if split == 'train' and hasattr(getattr(dataloader, 'dataset', None), 'disable_random_cut'):
        dataloader.dataset.disable_random_cut()

    # Optional heading perturbation for evaluation: rotates the world-frame IMU
    # input about the vertical axis, simulating the yaw offset between a
    # deployment IMU world frame and the GT-pelvis canonical frame used during
    # training.
    #   experiment.eval_input_yaw_deg=K   -> constant K-degree offset
    #   experiment.eval_yaw_random=True    -> uniform random offset per sample
    #                                         (heading-unknown protocol)
    # GT supervision stays canonical in both cases. 0.0/False disables.
    eval_yaw_deg = float(config.experiment.get("eval_input_yaw_deg", 0.0) or 0.0)
    eval_yaw_random = bool(config.experiment.get("eval_yaw_random", False))
    # Undo the applied input yaw on the *prediction* before saving/aggregating
    # metrics, so results are reported in the GT canonical frame (heading-unknown
    # protocol). Without this, body metrics are still meaningful (they rigid-align),
    # but MTE/ATE includes the unobservable constant-yaw offset of the output frame.
    _align_pred_yaw = bool(config.experiment.get("eval_yaw_align_pred", False))
    if eval_yaw_random:
        random.seed(int(config.experiment.get("eval_yaw_rand_seed", 0)))
    eval_yaw_rad = math.radians(eval_yaw_deg)
    if eval_yaw_rad or eval_yaw_random:
        logger.info(
            "Applying world-yaw heading perturbation to IMU inputs: "
            "constant=%.1f deg, random_per_sample=%s, align_pred_yaw=%s.",
            eval_yaw_deg,
            eval_yaw_random,
            _align_pred_yaw,
        )

    n_evaluated = 0

    def _text_metrics(text_logits, labels, imu_batches):
        """Return (top1, top5, ce) over the GT caption tokens, or None if the
        sample has no caption. Only the text tokens count: the <|eostatus|>,
        <|sot|> prefix and the <|eot|>/<|soobj|>/<|eoobj|> + object suffix of
        the text label row are excluded."""
        if not text_logits:
            # No caption was sampled (decode_text=False).
            return None
        text_logits = torch.cat(text_logits, dim=0)  # [len, vocab_size]
        n_valid_object_gt = len(imu_batches[0]['objects'])
        gt_text_tokens = labels[3][0][None][:, 2:-3-n_valid_object_gt]
        if gt_text_tokens.shape[1] == 0:
            return None
        text_top1_acc = topk_accuracy(text_logits[None, :, :], gt_text_tokens, k=1)
        text_top5_acc = topk_accuracy(text_logits[None, :, :], gt_text_tokens, k=5)
        min_len = min(text_logits.shape[0], gt_text_tokens.shape[1])
        text_ce = F.cross_entropy(text_logits[0:min_len], gt_text_tokens[0, 0:min_len], reduction='mean')
        return text_top1_acc, text_top5_acc, text_ce

    def _write_sample_header(f, sample_idx, invalid_imu_id_list, imu_batches):
        f.write(f"sample_idx: {sample_idx}\n")
        f.write(f"invalid_imu_id_list: {invalid_imu_id_list}\n")
        f.write(f"imu_point_count: {NUM_IMU_SENSORS - len(invalid_imu_id_list)}\n")
        active_sensor_names = [
            sensor_name
            for sensor_id, sensor_name in enumerate(IMU_SENSOR_NAMES)
            if sensor_id not in invalid_imu_id_list
        ]
        f.write(f"active_imu_sensor_names: {active_sensor_names}\n")
        f.write(f"input imu length: {len(imu_batches[0]['imu_data'])}\n")

    def _write_text_lines(f, empty_gt_text, text_top1_acc, text_top5_acc, text_ce, text_result, imu_batches):
        if not empty_gt_text:
            f.write(f"text_top1_acc: {text_top1_acc.item()*100:.4f}%, text_top5_acc: {text_top5_acc.item()*100:.2f}%, text_ce: {text_ce.item():.4f}\n")
        f.write(f"pred_text: {text_result}\n")
        f.write(f"gt_text: {imu_batches[0]['description']}\n")

    # ------------------------------------------------------------------
    # Raw per-sample eval outputs (structured / full eval only).
    #
    # Running the model is the expensive part, so everything a later metric
    # could need is written verbatim next to the summaries: the generated and
    # reference captions, and the predicted / GT object tracks with both the
    # sample's own bbox and the mesh-derived catalog extent. Adding a metric
    # (an LLM caption judge, scene IoU at another threshold) is then an offline
    # pass over records-*.jsonl + tracks/, never another evaluation run.
    #
    # One records file per rank, since ranks evaluate a disjoint modulo
    # partition of the same stream; readers glob records-*.jsonl.
    # ------------------------------------------------------------------
    records_file = None
    tracks_dir = None
    dump_records = bool(config.experiment.get("full_eval_dump_records", True))
    if structured_output and eval_output_root is not None and dump_records:
        os.makedirs(eval_output_root, exist_ok=True)
        records_file = open(
            os.path.join(
                eval_output_root, f"records-{accelerator.process_index:02d}.jsonl"
            ),
            "w",
            encoding="utf-8",
        )
        tracks_dir = os.path.join(eval_output_root, "tracks")
        os.makedirs(tracks_dir, exist_ok=True)

    def _record_numpy(value):
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _record_object_entry(name, obj, asset_id):
        """Static per-object fields; the per-frame track goes to tracks/."""
        rot = _record_numpy(obj['rot']).reshape(-1, 6)
        transl = _record_numpy(obj['transl']).reshape(-1, 3)
        bbox = _record_numpy(obj['bbox']).reshape(-1, 3)
        asset = None
        if asset_id is not None:
            asset_index = identity_catalog.asset_id_to_index.get(str(asset_id))
            if asset_index is not None:
                asset = identity_catalog.assets[asset_index]
        entry = {
            'category': canonical_category(name),
            'asset_id': None if asset_id is None else str(asset_id),
            # The extent carried by the sample itself. HUMOTO ships a unit
            # placeholder here (bbox_source: legacy_unit_placeholder), which is
            # why the scene metrics use catalog_extent_m: the mesh-derived
            # extent from dataset_process/object_identity_catalog.json,
            # expressed in the same canonical frame as the tracks.
            'sample_extent_m': [float(v) for v in bbox[0]],
            'catalog_extent_m': None if asset is None else canonical_extent_m(asset),
            'frames': int(len(rot)),
            'rot6d_first': [float(v) for v in rot[0]],
            'transl_first': [float(v) for v in transl[0]],
        }
        moving = obj.get('moving') if isinstance(obj, dict) else None
        if moving is not None:
            entry['moving_fraction'] = float(_record_numpy(moving).mean())
        return entry

    def _base_eval_record(sample_idx, invalid_imu_id_list, imu_batches, text_result, mpjpe_mm):
        """Per-sample record fields that do not depend on decoded motion.

        Both the text_only early exit and the full record below write these, so
        a caption-only profile still lands in records-*.jsonl: BLEU/ROUGE/CIDEr
        /BERTScore are scored offline from that file (evaluation/
        score_saved_outputs.py), never from summary.json, whose text numbers are
        token-position diagnostics.
        """
        gt_description = imu_batches[0]['description']
        return {
            'sample_id': (
                str(sample_idx[0])
                if isinstance(sample_idx, (list, tuple)) and len(sample_idx) == 1
                else str(sample_idx)
            ),
            'postfix': postfix,
            'eval_frames': None if eval_frames is None else int(eval_frames),
            'step': int(global_step),
            'invalid_imu_id': [int(v) for v in invalid_imu_id_list],
            'imu_point_count': NUM_IMU_SENSORS - len(invalid_imu_id_list),
            'input_imu_frames': int(len(imu_batches[0]['imu_data'])),
            # None when no motion was decoded; a text-only pass scores no pose.
            'mpjpe_mm': mpjpe_mm,
            'pred_text': None if text_result is None else str(text_result),
            'gt_texts': (
                [str(gt_description)]
                if isinstance(gt_description, str)
                else [str(item) for item in (gt_description or [])]
            ),
        }

    for imu_batch_idx, imu_batches in enumerate(dataloader):
        if imu_batch_idx >= eval_num:
            break

        # Every rank reads the same deterministic eval stream, but only executes
        # its disjoint modulo partition.  Compose this with the existing manual
        # job sharding so multi-node/offline launches remain deterministic.
        distributed_total_jobs = total_jobs * accelerator.num_processes
        distributed_job_id = job_id * accelerator.num_processes + accelerator.process_index
        if imu_batch_idx % distributed_total_jobs != distributed_job_id:
            continue

        # Evaluation probe: per-sensor world-yaw random walk on the input.
        _drift_deg = float(config.experiment.get("eval_input_yaw_drift_deg", 0.0) or 0.0)
        if _drift_deg:
            _drift_rng = np.random.default_rng(
                int(config.experiment.get("eval_yaw_drift_seed", 0)) + imu_batch_idx
            )
            for _b in imu_batches:
                if _b.get("imu_data") is None:
                    continue
                _numpy = isinstance(_b["imu_data"], np.ndarray)
                _t = torch.from_numpy(_b["imu_data"]) if _numpy else _b["imu_data"]
                _t = _drift_world_yaw(_t.float(), _drift_deg, _drift_rng)
                _b["imu_data"] = _t.numpy() if _numpy else _t

        _rad = eval_yaw_rad
        if eval_yaw_random:
            _rad = math.radians(random.uniform(0.0, 360.0))
        if _rad:
            # Perturb the input world frame only; GT supervision stays canonical.
            for _b in imu_batches:
                if _b.get("imu_data") is None:
                    continue
                _numpy = isinstance(_b["imu_data"], np.ndarray)
                _t = (
                    torch.from_numpy(_b["imu_data"])
                    if _numpy
                    else _b["imu_data"]
                )
                _b["imu_data"] = _rotate_world_yaw(_t.float(), _rad)
                if _numpy:
                    _b["imu_data"] = _b["imu_data"].numpy()
                if _b.get("imu_positions") is not None:
                    _p = (
                        torch.from_numpy(_b["imu_positions"])
                        if isinstance(_b["imu_positions"], np.ndarray)
                        else _b["imu_positions"]
                    )
                    _pf = _p.float()
                    _c = math.cos(_rad)
                    _s = math.sin(_rad)
                    _Ry = torch.tensor(
                        [[_c, 0.0, _s], [0.0, 1.0, 0.0], [-_s, 0.0, _c]],
                        dtype=_pf.dtype,
                        device=_pf.device,
                    )
                    _b["imu_positions"] = (
                        _Ry @ _pf.unsqueeze(-1)
                    ).squeeze(-1)

        # Cascade: swap the ground-truth motion for an upstream run's
        # prediction before anything reads it. imu_to_input quantizes these
        # three fields into the status tokens a motion-input profile is
        # conditioned on, so overriding them here -- alongside the yaw probes
        # above, and for the same reason -- is what makes the caption numbers
        # end-to-end rather than oracle-motion numbers.
        if cascade_motion_source is not None:
            for _b in imu_batches:
                if len(_b) == 0:
                    continue
                if not cascade_motion_source.override(_b):
                    raise RuntimeError(
                        f"no cascade motion for sample {_b.get('sample_idx')!r} in "
                        f"{cascade_motion_source.path}; the upstream run and this pass "
                        "must evaluate the same sample set (reuse its "
                        "evaluated_sample_ids_frames-*.txt)"
                    )

        if eval_output_root is not None:
            result_name = postfix or split
            if structured_output:
                save_folder = os.path.join(eval_output_root, "per_sequence")
            else:
                save_folder = os.path.join(
                    eval_output_root, "per_sequence", result_name
                )
        else:
            if split == "val":
                save_folder = config.experiment.viz_dir
            elif split == "train":
                save_folder = config.experiment.viz_train_dir
            elif split == "test":
                save_folder = config.experiment.viz_test_dir
            if generate_number:
                save_folder = save_folder + "_generate_number"
            if postfix != "":
                save_folder = save_folder + "_" + postfix
        os.makedirs(save_folder, exist_ok=True)

        # txt_file = os.path.join(save_folder, f"id_{imu_batch_idx}_step_{global_step}.txt")
        # if os.path.exists(txt_file):
        #     try:
        #         with open(txt_file, 'r') as f:
        #             lines = f.readlines()
        #         if len(lines) >= 2:
        #             print(f"Skipping batch {imu_batch_idx + 1}/{eval_num}, already exists.")
        #             continue
        #     except:
        #         pass

        print(f"Evaluating batch {imu_batch_idx + 1}/{eval_num} ...")
        logger.info(f"Evaluating batch {imu_batch_idx + 1}/{eval_num} ...")
        input_dict = imu_to_input(
                model,
                time_series_quantizer,
                imu_batches,
                accelerator,
                uni_prompting,
                mask_dtype,
                obj_name_to_id,
                object_token_bias,
                identity_catalog,
                normalization_window_size=config.model.normalization_window_size,
                smooth_imu=config.model.smooth_imu,
                invalid_imu_id=invalid_imu_id,
                random_text=False, # TODO
                dynamic_object=dynamic_object,
                bidirectional_imu=bidirectional_imu,
                bidirectional_motion=bidirectional_motion,
                predict_objects=config.model.get('predict_objects', True),
                fps=config.dataset.params.fps,
            )
        imu_embeddings = input_dict['imu_embeddings_batch']
        query_embeddings = input_dict['query_embeddings_batch']
        labels = input_dict['labels']
        attention_mask = input_dict['seq_attention_mask']
        input_imu_lens = input_dict['input_imu_len_batch']
        invalid_imu_id_list = input_dict['invalid_imu_id_list']
        sample_idx = input_dict['sample_idx_batch']

        all_continuous_recon = config.model.showo.get('all_continuous_recon', False)
        partial_continuous_recon = config.model.showo.get('partial_continuous_recon', False)
        x_recon_precomputed = None  # when all_continuous_recon or partial_continuous_recon, filled by model forward
        motion_query_hidden_states = None

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"): # disable autocast for debugging

            input_imu_len = input_imu_lens[0]
            n_time_token = (input_imu_len - 2)//6-2  # 2 for <|soimu|> and <|eoimu|>, 3 for each imu device, -2 for <|sostatus|> and <|eostatus|>

            # Invalid IMU positions for attention isolation (only attend to self; others don't attend to them)
            invalid_positions = get_invalid_imu_positions(input_imu_len, invalid_imu_id_list) if invalid_imu_id_list else None

            # Direct prediction without rollout
            top_k = 1

            if text_only or uncond_motion_input:
                # Motion is never decoded here: text-only supervision, or
                # unconditional generation where the motion span is the learned
                # queries themselves. Either way the downstream heads read the
                # motion-position embeddings the sequence already carries --
                # ground-truth status embeddings in causal mode, learned queries
                # in bidirectional mode -- exactly as in training.
                # Text-only supervision: motion is never decoded. Condition the
                # text decoder on the motion-position embeddings the training
                # sequence carries (ground-truth status embeddings in causal
                # mode, learned queries in bidirectional mode), exactly as in
                # training.
                L = input_imu_len + n_time_token
                cur_input_embeddings = input_dict['seq_all_embeddings'][:, :L]
                cur_attention_mask = truncate_attn_mask(attention_mask, L)
                if uncond_motion_input:
                    # A text-only profile stops here, but an unconditional one
                    # can still predict OBJECTS (i2s, uncond_scene_strict), and
                    # the track decoder reads the motion queries as its
                    # status_hidden (models/modeling_showo_imu.py:
                    # predict_object_tracks). One forward pass over the full
                    # sequence supplies them; nothing is rolled out, exactly as
                    # in the gt_motion_input branch below.
                    output = model(
                        input_embeddings=input_dict['seq_all_embeddings'],
                        attention_mask=attention_mask,
                        labels=labels,
                        input_imu_len=input_imu_lens,
                    )
                    _mq = output.get('motion_query_hidden_states')
                    if _mq is None and decode_objects:
                        raise RuntimeError(
                            "Unconditional evaluation needs motion_query_hidden_states "
                            "to decode objects; enable model.dynamic_object"
                        )
                    motion_query_hidden_states = None if _mq is None else _mq[0]
                    # There is no motion prediction at all here -- the motion
                    # span IS the learned queries -- so the human pose written
                    # to the sample is all zeros and every motion metric is
                    # skipped through motion_not_decoded. The six token
                    # stand-ins are the same ones the gt_motion_input branch
                    # uses: nothing reads them once the metrics are gated off,
                    # but the token bundle is written unconditionally.
                    n_status_token = labels[0][0].shape[1]
                    n_var_all = labels[2][0][0].shape[-1]
                    _dummy_device = model.showo.lm_head.weight.device
                    x_recon_precomputed = torch.zeros(
                        n_status_token, model.compression_rate, n_var_all,
                        dtype=torch.float32, device=_dummy_device,
                    )
                    result_static_idx = torch.zeros(n_status_token, 1, dtype=torch.long, device=_dummy_device)
                    result_mean_idx = torch.zeros(n_status_token, 1, dtype=torch.long, device=_dummy_device)
                    result_std_idx = torch.zeros(n_status_token, 1, dtype=torch.long, device=_dummy_device)
                    result_static_logits = torch.zeros(n_status_token, 1, 1, device=_dummy_device)
                    result_mean_logits = torch.zeros(n_status_token, 1, 1, device=_dummy_device)
                    result_std_logits = torch.zeros(n_status_token, 1, 1, device=_dummy_device)
            elif gt_motion_input:
                # Motion is teacher-forced ground truth; the heads after it are
                # what this profile predicts (objects for M2S, caption +
                # objects for M2TS). The motion
                # positions already carry the teacher-forced ground-truth
                # status embeddings, so nothing is rolled out; one forward pass
                # supplies the motion queries the object-track head consumes,
                # and the human motion is rebuilt from the same ground-truth
                # tokens the sequence was built from (the quantizer round trip
                # keeps it in the representation the downstream code expects).
                n_status_token = labels[0][0].shape[1]
                all_embeddings = input_dict['seq_all_embeddings']
                output = model(
                    input_embeddings=all_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    input_imu_len=input_imu_lens,
                )
                # Only the object-track head consumes these, and the model
                # returns them only when dynamic_object or bidirectional_motion
                # is on. A profile that predicts no objects (supervise:[text])
                # takes the text_only branch above, but stay defensive.
                _mq = output.get('motion_query_hidden_states')
                if _mq is None and decode_objects:
                    raise RuntimeError(
                        "Teacher-forced evaluation needs motion_query_hidden_states "
                        "to decode objects; enable model.dynamic_object"
                    )
                motion_query_hidden_states = None if _mq is None else _mq[0]
                gt_static_idx = labels[2][0][0]  # [n_status_token, n_var]
                gt_mean_idx = labels[0][0][0]
                gt_std_idx = labels[1][0][0]
                n_var_all = gt_static_idx.shape[-1]
                x_mean = time_series_quantizer.mean_quantizer.decode(
                    gt_mean_idx.reshape(-1)
                ).reshape(n_status_token, 1, n_var_all).float()
                x_std = time_series_quantizer.std_quantizer.decode(
                    gt_std_idx.reshape(-1)
                ).reshape(n_status_token, 1, n_var_all).float()
                x_static = static_embedder(gt_static_idx)  # [n_status_token, n_var, emb]
                embedding_dim = x_static.shape[-1]
                x_static = x_static.reshape(-1, embedding_dim, 1)
                x_static = time_series_quantizer.model.decoder(x_static, model.compression_rate)
                x_static = x_static.reshape(
                    n_status_token, n_var_all, model.compression_rate
                ).permute(0, 2, 1).float()  # [n_status_token, comp, n_var]
                x_recon_precomputed = x_static * x_std + x_mean
                status_end = input_imu_len + n_status_token
                cur_input_embeddings = all_embeddings[:, :status_end]
                L = cur_input_embeddings.shape[1]
                cur_attention_mask = truncate_attn_mask(attention_mask, L)
                # Motion is not predicted here; the status metrics below are skipped.
                _dummy_device = model.showo.lm_head.weight.device
                result_static_idx = torch.zeros(n_status_token, 1, dtype=torch.long, device=_dummy_device)
                result_mean_idx = torch.zeros(n_status_token, 1, dtype=torch.long, device=_dummy_device)
                result_std_idx = torch.zeros(n_status_token, 1, dtype=torch.long, device=_dummy_device)
                result_static_logits = torch.zeros(n_status_token, 1, 1, device=_dummy_device)
                result_mean_logits = torch.zeros(n_status_token, 1, 1, device=_dummy_device)
                result_std_logits = torch.zeros(n_status_token, 1, 1, device=_dummy_device)
            elif bidirectional_motion:
                if all_continuous_recon:
                    # all_continuous_recon: direct regression via nn.Linear, no discrete logits
                    n_status_token = labels[0][0].shape[1]
                    all_embeddings = input_dict['seq_all_embeddings']
                    output = model(
                        input_embeddings=all_embeddings,
                        attention_mask=attention_mask,
                        labels=labels,
                        input_imu_len=input_imu_lens,
                    )
                    motion_query_hidden_states = output[
                        'motion_query_hidden_states'
                    ][0]
                    # Model returns [bs, n_frame, n_var], reshape to [n_status_token, compression_rate, n_var]
                    x_recon_precomputed = output['x_recon'][0].float().reshape(n_status_token, model.compression_rate, -1)
                    # Build cur_input_embeddings up to end of status for text/object generation
                    status_end = input_imu_len + n_status_token
                    cur_input_embeddings = all_embeddings[:, :status_end]
                    L = cur_input_embeddings.shape[1]
                    cur_attention_mask = truncate_attn_mask(attention_mask, L)
                    # Dummy values for status metrics (will be skipped)
                    result_static_idx = torch.zeros(n_time_token, 1, dtype=torch.long, device=model.showo.lm_head.weight.device)
                    result_mean_idx = torch.zeros(n_time_token, 1, dtype=torch.long, device=model.showo.lm_head.weight.device)
                    result_std_idx = torch.zeros(n_time_token, 1, dtype=torch.long, device=model.showo.lm_head.weight.device)
                    result_static_logits = torch.zeros(n_time_token, 1, 1, device=model.showo.lm_head.weight.device)
                    result_mean_logits = torch.zeros(n_time_token, 1, 1, device=model.showo.lm_head.weight.device)
                    result_std_logits = torch.zeros(n_time_token, 1, 1, device=model.showo.lm_head.weight.device)

                elif partial_continuous_recon:
                    # partial_continuous_recon: motion (transl+orient) discrete CE, pose continuous
                    n_status_token = labels[0][0].shape[1]
                    all_embeddings = input_dict['seq_all_embeddings']
                    output = model(
                        input_embeddings=all_embeddings,
                        attention_mask=attention_mask,
                        labels=labels,
                        input_imu_len=input_imu_lens,
                    )
                    motion_query_hidden_states = output[
                        'motion_query_hidden_states'
                    ][0]
                    x_mean_logits_all = output['x_mean_logits_all'][0:1].float()  # [1, n_status_token, 3+rot_dof, n_dynamic_bins]
                    x_std_logits_all = output['x_std_logits_all'][0:1].float()
                    x_value_logits_all = output['x_value_logits_all'][0:1].float()
                    x_pose = output['x_pose'][0:1].float()  # [1, n_frame, 21*rot_dof]
                    rot_dof = model.rot_dof
                    n_motion_var = 3 + rot_dof
                    # Decode motion from discrete logits
                    x_static_idx = last_dim_logit_to_index(x_value_logits_all, temperature=temperature, top_k=top_k)[0]  # [n_status_token, n_motion_var]
                    x_mean_idx = last_dim_logit_to_index(x_mean_logits_all, temperature=temperature, top_k=top_k)[0]
                    x_std_idx = last_dim_logit_to_index(x_std_logits_all, temperature=temperature, top_k=top_k)[0]
                    x_mean = time_series_quantizer.mean_quantizer.decode(x_mean_idx.reshape(-1)).reshape(n_status_token, 1, n_motion_var)
                    x_std = time_series_quantizer.std_quantizer.decode(x_std_idx.reshape(-1)).reshape(n_status_token, 1, n_motion_var)
                    x_static = static_embedder(x_static_idx)  # [n_status_token, n_motion_var, embedding_dim]
                    _, _, embedding_dim = x_static.shape
                    x_static = x_static.reshape(-1, embedding_dim, 1)
                    x_static = time_series_quantizer.model.decoder(x_static, model.compression_rate)
                    x_static = x_static.reshape(n_status_token, n_motion_var, model.compression_rate).permute(0, 2, 1)  # [n_status_token, comp, n_motion_var]
                    x_motion = x_static * x_std + x_mean  # [n_status_token, comp, n_motion_var]
                    x_pose = x_pose.reshape(n_status_token, model.compression_rate, 21 * rot_dof)
                    x_recon_precomputed = torch.cat([x_motion, x_pose], dim=-1)  # [n_status_token, comp, 3+22*rot_dof]
                    status_end = input_imu_len + n_status_token
                    cur_input_embeddings = all_embeddings[:, :status_end]
                    L = cur_input_embeddings.shape[1]
                    cur_attention_mask = truncate_attn_mask(attention_mask, L)
                    # For status metrics use motion logits (pose has no discrete)
                    result_static_idx = x_static_idx
                    result_mean_idx = x_mean_idx
                    result_std_idx = x_std_idx
                    result_static_logits = x_value_logits_all[0].reshape(n_status_token, n_motion_var, -1)
                    result_mean_logits = x_mean_logits_all[0].reshape(n_status_token, n_motion_var, -1)
                    result_std_logits = x_std_logits_all[0].reshape(n_status_token, n_motion_var, -1)

                else:
                    # All discrete
                    cur_input_embeddings = torch.cat([imu_embeddings, query_embeddings], dim=1) # [batch_size, length, d_model]
                    assert cur_input_embeddings.shape[1] == input_imu_len + n_time_token

                    cur_attention_mask = truncate_attn_mask(
                        attention_mask, input_imu_len + n_time_token
                    )
                    L = input_imu_len + n_time_token
                    bs = cur_input_embeddings.shape[0]
                    assert bs == 1

                    # bidirectional motion inference: one-shot predict all motion tokens from full context
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        hidden_states = model.backbone_forward(
                            cur_input_embeddings,
                            cur_attention_mask,
                        )

                    status_hidden_states = hidden_states[:, input_imu_len:, :]  # [batch_size, n_time_token, hidden_dim]
                    assert status_hidden_states.shape[1] == n_time_token
                    motion_query_hidden_states = status_hidden_states[0]

                    _x_static_logits = model.value_head(status_hidden_states)
                    _x_mean_logits = model.mean_head(status_hidden_states)
                    _x_std_logits = model.std_head(status_hidden_states)

                    _x_static_logits = _x_static_logits.reshape(bs, n_time_token, -1, n_static_bins)
                    _x_mean_logits = _x_mean_logits.reshape(bs, n_time_token, -1, n_dynamic_bins)
                    _x_std_logits = _x_std_logits.reshape(bs, n_time_token, -1, n_dynamic_bins)

                    _x_pose_static_logits = model.pose_value_head(status_hidden_states) # [batch_size, n_time_token, n_var*n_bins]
                    _x_pose_mean_logits = model.pose_mean_head(status_hidden_states)
                    _x_pose_std_logits = model.pose_std_head(status_hidden_states)

                    _x_pose_static_logits = _x_pose_static_logits.reshape(bs, n_time_token, -1, n_static_bins)
                    _x_pose_mean_logits = _x_pose_mean_logits.reshape(bs, n_time_token, -1, n_dynamic_bins)
                    _x_pose_std_logits = _x_pose_std_logits.reshape(bs, n_time_token, -1, n_dynamic_bins)

                    _x_static_logits = torch.cat([_x_static_logits, _x_pose_static_logits], dim=2) # [batch_size, n_time_token, n_var, n_bins]
                    _x_mean_logits = torch.cat([_x_mean_logits, _x_pose_mean_logits], dim=2) # [batch_size, n_time_token, n_var, n_bins]
                    _x_std_logits = torch.cat([_x_std_logits, _x_pose_std_logits], dim=2) # [batch_size, n_time_token, n_var, n_bins]

                    # sample indices for all time steps at once
                    _x_static_idx = last_dim_logit_to_index(_x_static_logits, temperature=temperature, top_k=top_k)[0] # [n_time_token, n_var]
                    _x_mean_idx = last_dim_logit_to_index(_x_mean_logits, temperature=temperature, top_k=top_k)[0] # [n_time_token, n_var]
                    _x_std_idx = last_dim_logit_to_index(_x_std_logits, temperature=temperature, top_k=top_k)[0] # [n_time_token, n_var]

                    # reshape to [n_time_token, n_var] to match casual branch output format
                    result_static_idx = _x_static_idx
                    result_mean_idx = _x_mean_idx
                    result_std_idx = _x_std_idx
                    assert len(result_static_idx.shape) == 2

                    result_static_logits = _x_static_logits.reshape(n_time_token, -1, n_static_bins) # [n_time_token, n_var, n_bins]
                    result_mean_logits = _x_mean_logits.reshape(n_time_token, -1, n_dynamic_bins) # [n_time_token, n_var, n_bins]
                    result_std_logits = _x_std_logits.reshape(n_time_token, -1, n_dynamic_bins) # [n_time_token, n_var, n_bins]

            else:
                # Auto-regressive
                if accelerator.is_main_process:
                    time_bar = tqdm.tqdm(range(n_time_token))
                else:
                    time_bar = range(n_time_token)
                cur_input_embeddings = imu_embeddings # [batch_size, length, d_model]
                assert cur_input_embeddings.shape[1] == input_imu_len
                cur_attention_mask = truncate_attn_mask(attention_mask, input_imu_len)
                L = input_imu_len

                result_static_idx = []
                result_mean_idx = []
                result_std_idx = []
                result_static_logits = []
                result_mean_logits = []
                result_std_logits = []
                causal_motion_query_hidden_states = []

                # loop to reconstruct the trajectory and human pose
                for t in range(len(time_bar)):

                    # Forward pass
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        hidden_states = model.backbone_forward(
                            cur_input_embeddings,
                            cur_attention_mask,
                        )
                    causal_motion_query_hidden_states.append(hidden_states[0, -1])

                    # Get logits
                    _x_static_logits = model.value_head(hidden_states[:, -1])
                    _x_mean_logits = model.mean_head(hidden_states[:, -1])
                    _x_std_logits = model.std_head(hidden_states[:, -1])

                    _x_static_logits = _x_static_logits.reshape(-1, n_static_bins)
                    _x_mean_logits = _x_mean_logits.reshape(-1, n_dynamic_bins)
                    _x_std_logits = _x_std_logits.reshape(-1, n_dynamic_bins)

                    _x_pose_static_logits = model.pose_value_head(hidden_states[:, -1])
                    _x_pose_mean_logits = model.pose_mean_head(hidden_states[:, -1])
                    _x_pose_std_logits = model.pose_std_head(hidden_states[:, -1])

                    _x_pose_static_logits = _x_pose_static_logits.reshape(-1, n_static_bins)
                    _x_pose_mean_logits = _x_pose_mean_logits.reshape(-1, n_dynamic_bins)
                    _x_pose_std_logits = _x_pose_std_logits.reshape(-1, n_dynamic_bins)

                    _x_static_logits = torch.cat([_x_static_logits, _x_pose_static_logits], dim=0)
                    _x_mean_logits = torch.cat([_x_mean_logits, _x_pose_mean_logits], dim=0)
                    _x_std_logits = torch.cat([_x_std_logits, _x_pose_std_logits], dim=0)

                    # Sample indices
                    _x_static_idx = last_dim_logit_to_index(_x_static_logits[None], temperature=temperature, top_k=top_k)[0]
                    _x_mean_idx = last_dim_logit_to_index(_x_mean_logits[None], temperature=temperature, top_k=top_k)[0]
                    _x_std_idx = last_dim_logit_to_index(_x_std_logits[None], temperature=temperature, top_k=top_k)[0]

                    # Store predictions
                    result_static_idx.append(_x_static_idx)
                    result_mean_idx.append(_x_mean_idx)
                    result_std_idx.append(_x_std_idx)
                    result_static_logits.append(_x_static_logits)
                    result_mean_logits.append(_x_mean_logits)
                    result_std_logits.append(_x_std_logits)

                    # Create embeddings for next step
                    _x_static_embed = static_embedder(_x_static_idx).to(weight_dtype)
                    _x_mean_embed = mean_embedder(_x_mean_idx).to(weight_dtype)
                    _x_std_embed = std_embedder(_x_std_idx).to(weight_dtype)

                    _x_dynamic_embed = torch.stack([_x_mean_embed, _x_std_embed], dim=-1)
                    idx_next_embeddings = model.status_aggregator(_x_static_embed[None], _x_dynamic_embed[None])
                    idx_next_embeddings += time_embedder.get_pe_at_index(t, fps=fps).to(weight_dtype)

                    # Update state for next iteration
                    L = L + 1
                    cur_input_embeddings = torch.cat([cur_input_embeddings, idx_next_embeddings[None]], dim=1)
                    cur_attention_mask = extend_attn_mask(attention_mask, L, attention_mask.dtype, invalid_positions=invalid_positions)

                result_static_idx = torch.cat(result_static_idx, dim=0)  # [L, n_var]
                result_mean_idx = torch.cat(result_mean_idx, dim=0)  # [L, n_var]
                result_std_idx = torch.cat(result_std_idx, dim=0)  # [L, n_var]
                result_static_logits = torch.cat(result_static_logits, dim=0)  # [L, n_var, n_bins]
                result_mean_logits = torch.cat(result_mean_logits, dim=0)  # [L, n_var, n_bins]
                result_std_logits = torch.cat(result_std_logits, dim=0)  # [L, n_var, n_bins]
                motion_query_hidden_states = torch.stack(
                    causal_motion_query_hidden_states, dim=0
                )
                # import pdb; pdb.set_trace()

            # loop to reconstruct the text
            text_result = []
            text_logits = []

            # min_new_before_eot = 2
            # neg_inf = -100
            # skip_first_eot_token = True
            # import pdb;pdb.set_trace()

            # The caption span follows the motion as <|eostatus|> <|sot|> text
            # <|eot|> (imu_to_input), and the object decoder below continues
            # from whatever ends it.
            eostatus_token = uni_prompting.sptids_dict['<|eostatus|>'].item()
            sot_token = uni_prompting.sptids_dict['<|sot|>'].item()
            if decode_text:
                text_prefix_tokens = [eostatus_token, sot_token]
            else:
                # A profile that does not train the text head drops the
                # caption from its sequence, so <|sot|><|eot|> adjacent is the
                # layout it was trained on. Feeding <|eot|> directly, instead
                # of sampling up to max_new_text_tokens from the untrained
                # lm_head (which never emits <|eot|>), skips ~60 cached
                # forwards per sample -- 78% of i2m's evaluation time -- and
                # gives the object decoder its training-time context.
                text_prefix_tokens = [eostatus_token, sot_token, eot_token]
            text_prefix = model.embed_tokens(
                torch.tensor(
                    text_prefix_tokens,
                    device=cur_input_embeddings.device,
                    dtype=torch.long,
                )
            )[None]
            cur_input_embeddings = torch.cat([cur_input_embeddings, text_prefix], dim=1)
            L += len(text_prefix_tokens)
            cur_attention_mask = extend_attn_mask(
                attention_mask, L, attention_mask.dtype,
                invalid_positions=invalid_positions,
            )
            cache_length = L
            if decode_text or decode_objects:
                # Prefill the KV cache once. All following text/object steps
                # feed one embedding and one mask row.
                hidden_states, past_key_values = model.backbone_forward_cached(
                    cur_input_embeddings,
                    cur_attention_mask,
                )
                cached_hidden_state = hidden_states[:, -1]
            else:
                # Nothing is decoded after the motion, so the prefill forward
                # over the whole sequence would be discarded.
                cached_hidden_state = None
                past_key_values = None

            sample_forced = None
            if forced_decode:
                sample_forced = forced_decode.get(
                    str(sample_idx[0])
                    if isinstance(sample_idx, (list, tuple)) and len(sample_idx) == 1
                    else str(sample_idx)
                )
            forced_text_tokens = None
            if decode_text and sample_forced is not None and sample_forced.get('caption'):
                forced_text_tokens = list(
                    uni_prompting.text_tokenizer(sample_forced['caption'])['input_ids']
                ) + [eot_token]
            elif decode_text and teacher_force_text:
                # Same slice as _text_metrics: <|eostatus|> <|sot|> text <|eot|>
                # <|soobj|> objects <|eoobj|> -> text.
                n_gt_objects = len(imu_batches[0]['objects'])
                forced_text_tokens = labels[3][0][2:len(labels[3][0]) - 3 - n_gt_objects].tolist()
                forced_text_tokens = forced_text_tokens + [eot_token]
            n_text_steps = (
                (max_new_text_tokens if decode_text else 0)
                if forced_text_tokens is None else len(forced_text_tokens)
            )
            text_diverse_result = None
            if text_diverse is not None and decode_text and forced_text_tokens is None:
                sample_key = (
                    str(sample_idx[0])
                    if isinstance(sample_idx, (list, tuple)) and len(sample_idx) == 1
                    else str(sample_idx)
                )
                generator = torch.Generator(device=cached_hidden_state.device)
                generator.manual_seed(text_diverse['seed'] * 1000003 + zlib.crc32(sample_key.encode()))
                decode_args = (
                    model, past_key_values, cached_hidden_state, attention_mask,
                    cache_length, invalid_positions, eot_token, max_new_text_tokens,
                )

                def _caption_entry(tokens, logprobs):
                    text_ids = [t for t in tokens if t != eot_token]
                    return {
                        'text': uni_prompting.text_tokenizer.decode(text_ids, skip_special_tokens=True).strip(),
                        'n_tokens': len(tokens),
                        'logprob': float(sum(logprobs)),
                        'ended': bool(tokens and tokens[-1] == eot_token),
                    }

                text_diverse_result = {
                    'num': text_diverse['num'],
                    'temperature': text_diverse['temperature'],
                    'top_p': text_diverse['top_p'],
                    'samples': [
                        _caption_entry(*row) for row in decode_text_candidates(
                            *decode_args, num_samples=text_diverse['num'],
                            temperature=text_diverse['temperature'],
                            top_p=text_diverse['top_p'], generator=generator,
                        )
                    ],
                    'refs': {},
                }
                for ref_name, captions in (text_diverse['refs'].get(sample_key) or {}).items():
                    forced_rows = [
                        list(uni_prompting.text_tokenizer(c)['input_ids']) + [eot_token] for c in captions
                    ]
                    if forced_rows:
                        scored = decode_text_candidates(*decode_args, forced=forced_rows)
                        text_diverse_result['refs'][ref_name] = [
                            dict(_caption_entry(*row), text=c) for c, row in zip(captions, scored)
                        ]
            scene_generator = None
            if scene_sample is not None and (decode_text or decode_objects):
                scene_key = (
                    str(sample_idx[0])
                    if isinstance(sample_idx, (list, tuple)) and len(sample_idx) == 1
                    else str(sample_idx)
                )
                scene_generator = torch.Generator(device=cached_hidden_state.device)
                scene_generator.manual_seed(scene_sample['seed'] * 1000003 + zlib.crc32(scene_key.encode()))
            for it in range(n_text_steps):
                logits = model.showo.lm_head(cached_hidden_state)  # [bs, vocab_size]
                # if it < min_new_before_eot:
                #     # prevent generating <|eot|> token at the beginning
                #     logits[:, eot_token] = neg_inf
                if scene_generator is not None:
                    idx_next = sample_from_logits(
                        logits, scene_sample['temperature'], scene_sample['top_p'], scene_generator
                    )
                else:
                    idx_next = last_dim_logit_to_index(logits[:, None, :], temperature=temperature, top_k=1)  # greedy: full-vocabulary T=1 sampling made every text metric irreproducible
                if forced_text_tokens is not None:
                    idx_next = torch.full_like(idx_next, int(forced_text_tokens[it]))
                # idx_next = last_dim_logit_to_index(logits[:, None, :], temperature=1.2, top_k=500)  # [bs, 1]
                idx_next_embeddings = model.embed_tokens(idx_next)

                # input_imu_len[0]-1 is <|sostatus|> token, so it will decode the first status token
                # n_status_token = labels[0][0].shape[0]
                # traj_hidden_states = hidden_states[0, input_imu_len-1:input_imu_len+n_status_token-1]

                text_logits.append(logits)
                text_result.append(idx_next[0][0])

                cache_length += 1
                step_attention_mask = cached_decode_attention_mask(
                    attention_mask, cache_length, invalid_positions
                )
                hidden_states, past_key_values = model.backbone_forward_cached(
                    idx_next_embeddings,
                    step_attention_mask,
                    past_key_values=past_key_values,
                )
                cached_hidden_state = hidden_states[:, -1]

                if eot_token is not None and idx_next.item() == eot_token:
                    break

                # if eot_token is not None and idx_next.item() == eot_token:
                #     if skip_first_eot_token and it == 0:
                #         skip_first_eot_token = False
                #     else:
                        # break

            # import pdb;pdb.set_trace()
            # text_top1_acc = topk_accuracy(torch.cat(text_logits, dim=0)[None, :, :], labels[3][0][None][:, :len(text_logits)], k=1)
            # print(f"text_top1_acc = {text_top1_acc}")

            object_id_result = []
            object_id_logits = []
            max_object_id_tokens = max_object_id_tokens if decode_objects else 0
            soobj_token = uni_prompting.sptids_dict['<|soobj|>'].item()
            eoobj_token = uni_prompting.sptids_dict['<|eoobj|>'].item()
            valid_object_ids = list(object_token_id_to_name.keys())
            if decode_objects and model.identity_head is not None:
                # Some taxonomy labels intentionally have no geometry asset.
                # They may remain in the language-level vocabulary, but scene
                # decoding must only emit categories for which the identity
                # head can retrieve a concrete object. This is especially
                # important while freshly initialized object logits are random.
                identity_category_ids = set(
                    model.identity_head.asset_category_ids.detach().cpu().tolist()
                )
                valid_object_ids = [
                    token_id
                    for token_id in valid_object_ids
                    if obj_name_to_id[object_token_id_to_name[token_id]]
                    in identity_category_ids
                ]
                if not valid_object_ids:
                    raise ValueError(
                        "Object identity catalog has no decodable taxonomy categories"
                    )

            if decode_objects and feed_soobj and cached_hidden_state is not None:
                # Training lays out <|eot|> <|soobj|> ids <|eoobj|>: the <|eot|>
                # position is supervised to predict <|soobj|> and the first id
                # is predicted from the <|soobj|> position. Without this step
                # the first id is read off the <|eot|> hidden state and every
                # later object token sits one position early.
                soobj_ids = torch.tensor([[soobj_token]], device=cached_hidden_state.device, dtype=torch.long)
                cache_length += 1
                step_attention_mask = cached_decode_attention_mask(
                    attention_mask, cache_length, invalid_positions
                )
                hidden_states, past_key_values = model.backbone_forward_cached(
                    model.embed_tokens(soobj_ids),
                    step_attention_mask,
                    past_key_values=past_key_values,
                )
                cached_hidden_state = hidden_states[:, -1]

            # append <|soobj|> token
            # soobj_embedding = model.embed_tokens(torch.tensor([soobj_token], device=cur_input_embeddings.device, dtype=torch.long)).to(weight_dtype).reshape(1, 1, -1)
            # cur_input_embeddings = torch.cat([cur_input_embeddings, soobj_embedding], dim=1)
            # temp = torch.ones([1, 1, cur_attention_mask.shape[2] + 1, cur_attention_mask.shape[3] + 1], device=cur_attention_mask.device) * torch.finfo(cur_attention_mask.dtype).min
            # temp[:, :, :-1, :-1] = cur_attention_mask
            # temp[:, :, -1] = 0
            # cur_attention_mask = temp

            forced_object_ids = None
            forced_prefix_ids = []
            object_decode_info = {}
            decodable_categories = [object_token_id_to_name[t] for t in valid_object_ids]
            ground_token_ids = {
                t for t in valid_object_ids if is_static_ground(object_token_id_to_name[t])
            }
            if decode_objects:
                # First-step category distribution (non-ground, renormalised),
                # dumped so the multilabel threshold can be swept offline.
                with torch.no_grad():
                    first_logits = model.showo.lm_head(cached_hidden_state)[0].float()
                    cand = [t for t in valid_object_ids if t not in ground_token_ids]
                    cand_probs = torch.softmax(first_logits[cand], dim=-1)
                    all_probs = torch.softmax(first_logits[valid_object_ids], dim=-1)
                order = torch.argsort(cand_probs, descending=True).tolist()
                object_decode_info['first_step_ground_prob'] = float(sum(
                    all_probs[i].item() for i, t in enumerate(valid_object_ids) if t in ground_token_ids
                ))
                object_decode_info['first_step_probs'] = {
                    object_token_id_to_name[cand[i]]: round(cand_probs[i].item(), 5) for i in order[:15]
                }
            if decode_objects and object_list_override == 'set_head':
                # Plan A: the trained set head picks the categories in one shot;
                # they are forced in the training (canonical) order, ground last.
                if getattr(model, 'object_set_classifier', None) is None or not feed_soobj:
                    raise ValueError("set_head decoding needs model.object_set_head and eval_feed_soobj")
                with torch.no_grad():
                    set_probs = torch.sigmoid(model.object_set_classifier(cached_hidden_state)[0].float())
                set_ids = [t for t in valid_object_ids if t not in ground_token_ids]
                set_p = {t: set_probs[obj_name_to_id[object_token_id_to_name[t]]].item() for t in set_ids}
                threshold = float(object_multilabel_threshold)
                picked = [t for t in set_ids if set_p[t] >= threshold] or [max(set_p, key=set_p.get)]
                picked = sorted(
                    picked, key=lambda t: canonical_object_key(object_token_id_to_name[t], obj_name_to_id)
                )[: max(0, max_object_id_tokens - 1)]
                forced_object_ids = picked + sorted(ground_token_ids) + [eoobj_token]
                object_decode_info['set_head_probs'] = {
                    object_token_id_to_name[t]: round(p, 5)
                    for t, p in sorted(set_p.items(), key=lambda kv: -kv[1])[:15]
                }
            elif decode_objects and object_list_override == 'pred_text_verified':
                caption = uni_prompting.text_tokenizer.decode(
                    torch.tensor([int(t) for t in text_result], dtype=torch.long),
                    skip_special_tokens=True,
                ) if len(text_result) else ''
                verified = verified_caption_categories(
                    caption,
                    {object_token_id_to_name[cand[i]]: cand_probs[i].item() for i in range(len(cand))},
                    decodable_categories,
                    caption_verify_threshold,
                    max_object_id_tokens,
                )
                forced_prefix_ids = [obj_name_to_id[c] + object_token_bias for c in verified]
                object_decode_info['caption_used'] = caption
                object_decode_info['caption_verified'] = verified
            elif decode_objects and object_list_override == 'multilabel':
                # Every category above the threshold (at least the top one),
                # most probable first, ground last as in training.
                threshold = float(object_multilabel_threshold)
                picked = [cand[i] for i in order if cand_probs[i].item() >= threshold] or [cand[order[0]]]
                picked = picked[: max(0, max_object_id_tokens - 1)]
                forced_object_ids = picked + sorted(ground_token_ids) + [eoobj_token]
            elif decode_objects and object_list_override not in (None, 'no_early_stop', 'pred_text_verified'):
                if object_list_override == 'pred_text':
                    # The caption is decoded before <|soobj|>, so the model's
                    # own sentence is available here.
                    descriptions = uni_prompting.text_tokenizer.decode(
                        torch.tensor([int(t) for t in text_result], dtype=torch.long),
                        skip_special_tokens=True,
                    ) if len(text_result) else ''
                    object_decode_info['caption_used'] = descriptions
                else:
                    descriptions = imu_batches[0].get('description')
                forced_categories = object_list_override_categories(
                    'gt_text' if object_list_override == 'pred_text' else object_list_override,
                    descriptions=descriptions,
                    gt_object_names=imu_batches[0]['objects'].keys(),
                    decodable_categories=decodable_categories,
                    max_objects=max_object_id_tokens,
                )
                forced_object_ids = [
                    obj_name_to_id[c] + object_token_bias for c in forced_categories
                ] + [eoobj_token]
            if decode_objects and sample_forced is not None and sample_forced.get('objects'):
                # Forced categories in the given order, ground last as in training.
                unknown = [c for c in sample_forced['objects'] if c not in decodable_categories]
                if unknown:
                    raise ValueError(f"eval_forced_decode: undecodable categories {unknown}")
                forced_object_ids = [
                    obj_name_to_id[c] + object_token_bias for c in sample_forced['objects']
                ] + sorted(ground_token_ids) + [eoobj_token]
                forced_prefix_ids = []
            if sample_forced is not None:
                object_decode_info['forced_decode'] = dict(sample_forced)
            # Local count: max_object_id_tokens is the per-call cap and is
            # re-read for every sample of this loop.
            n_object_steps = (
                max_object_id_tokens if forced_object_ids is None else len(forced_object_ids)
            )

            for it in range(n_object_steps):
                logits = model.showo.lm_head(cached_hidden_state)  # [bs, vocab_size]
                # Set non-object logits to -100, only allow object class tokens and eoobj (when appropriate)
                if it >= min_object_id_tokens:
                    valid_token_ids = valid_object_ids + [eoobj_token]
                else:
                    valid_token_ids = valid_object_ids
                valid_logits = logits[:, valid_token_ids].clone()
                # This is a hard vocabulary constraint, not a training ignore
                # value.  A finite sentinel such as -100 can still win when a
                # freshly initialized/poorly calibrated object's valid logits
                # are even smaller, which lets taxonomy-only categories reach
                # the identity head despite the filtering above.
                logits[:, :] = -float('inf')
                logits[:, valid_token_ids] = valid_logits
                if object_list_override in ('no_early_stop', 'pred_text_verified') and not any(
                    int(t) not in ground_token_ids for t in object_id_result
                ):
                    # No stop and no ground before the first real object.
                    logits[:, [eoobj_token, *ground_token_ids]] = -float('inf')
                if scene_generator is not None and scene_sample['objects']:
                    # Invalid tokens are already -inf above.
                    idx_next = sample_from_logits(
                        logits, scene_sample['object_temperature'], 1.0, scene_generator
                    )
                else:
                    idx_next = last_dim_logit_to_index(logits[:, None, :], temperature=temperature, top_k=1)  # [bs, 1] randomly sample FIXME
                if forced_object_ids is not None:
                    idx_next = torch.full_like(idx_next, forced_object_ids[it])
                elif it < len(forced_prefix_ids):
                    idx_next = torch.full_like(idx_next, forced_prefix_ids[it])
                idx_next_embeddings = model.embed_tokens(idx_next)

                object_id_logits.append(logits)
                object_id_result.append(idx_next[0][0])

                cache_length += 1
                step_attention_mask = cached_decode_attention_mask(
                    attention_mask, cache_length, invalid_positions
                )
                hidden_states, past_key_values = model.backbone_forward_cached(
                    idx_next_embeddings,
                    step_attention_mask,
                    past_key_values=past_key_values,
                )
                cached_hidden_state = hidden_states[:, -1]

                if eoobj_token is not None and idx_next.item() == eoobj_token:
                    break

            object_id_valid = []
            for temp in object_id_result:
                if temp.item() in object_token_id_to_name.keys():
                    object_id_valid.append(temp.item())
            n_valid_object = len(object_id_valid)

            category_ids = torch.tensor(
                [
                    obj_name_to_id[object_token_id_to_name[token_id]]
                    for token_id in object_id_valid
                ],
                dtype=torch.long,
                device=cached_hidden_state.device,
            )
            use_geometry = getattr(model, 'object_geometry_embed', None) is not None
            object_status_result = []
            object_geometry_result = []
            for t in range(n_valid_object):
                head_hidden_state = cached_hidden_state
                if use_geometry:
                    # Retrieve this object's asset first, then condition its
                    # pose on that asset's shape (GT asset during training).
                    asset_index, _ = model.identity_head.predict(
                        cached_hidden_state, category_ids[t:t + 1]
                    )
                    geometry_term = model.object_geometry_term(asset_index, cached_hidden_state)
                    object_geometry_result.append(geometry_term)
                    head_hidden_state = cached_hidden_state + geometry_term
                _x_obj_mean_logits = model.object_mean_head(head_hidden_state) # [1, 12]
                object_status_result.append(cached_hidden_state)

                # _x_obj_mean_logits = _x_obj_mean_logits.reshape(-1, n_dynamic_bins)
                # _x_obj_mean_idx = torch.argmax(_x_obj_mean_logits, dim=-1)
                # _x_obj_mean_embed = mean_embedder(_x_obj_mean_idx).to(weight_dtype)  # [n_var]
                # idx_next_embeddings = model.object_aggregator(_x_obj_mean_embed[None], None).reshape(1,1,-1)
                # The checkpoint retains three legacy scale outputs.  Replace
                # them with the same neutral constants used during training so
                # scale predictions cannot affect autoregressive generation.
                object_status_for_embedding = _x_obj_mean_logits.clone()
                object_status_for_embedding[..., model.object_pose_dim:] = 1
                idx_next_embeddings = model.object_aggregator(
                    object_status_for_embedding[None]
                ) # [1, 1, 2048]

                cache_length += 1
                step_attention_mask = cached_decode_attention_mask(
                    attention_mask, cache_length, invalid_positions
                )
                hidden_states, past_key_values = model.backbone_forward_cached(
                    idx_next_embeddings,
                    step_attention_mask,
                    past_key_values=past_key_values,
                )
                cached_hidden_state = hidden_states[:, -1]


            pred_objects = {}
            if len(object_status_result) > 0:

                object_status_result = torch.cat(object_status_result, dim=0)
                object_geometry_result = (
                    torch.cat(object_geometry_result, dim=0) if use_geometry else None
                )
                object_head_hidden = (
                    object_status_result if object_geometry_result is None
                    else object_status_result + object_geometry_result
                )
                # x_object_mean_logits = model.object_mean_head(object_status_result).reshape(1, -1, n_dynamic_bins)
                # x_object_mean_idx = last_dim_logit_to_index(x_object_mean_logits, temperature=temperature, top_k=1) # [n_obj, 9]
                # x_object_mean = time_series_quantizer.mean_quantizer.decode(x_object_mean_idx.reshape(-1)).reshape(len(object_status_result), 9)
                x_object_mean = model.object_mean_head(object_head_hidden).reshape(
                    len(object_status_result), model.object_checkpoint_dim
                )[:, :model.object_pose_dim] # scale compatibility channels are intentionally ignored
                if getattr(model, 'object_anchor_head', 'regression') == 'flow':
                    # One deterministic sample per eval call (fixed seed) so
                    # metrics are reproducible; the regression mean above is
                    # kept only as a fallback / diagnostic.
                    anchor_generator = torch.Generator(device=object_status_result.device)
                    anchor_generator.manual_seed(0)
                    if scene_generator is not None:
                        anchor_generator = scene_generator  # scene sampling: the layout varies per seed too
                    if scene_generator is not None and scene_sample['layout_candidates'] > 0:
                        keep = torch.tensor(
                            [t not in ground_token_ids for t in object_id_valid],
                            device=object_head_hidden.device,
                        )
                        x_object_mean = farthest_layout(
                            model, object_head_hidden, keep, scene_sample['seed'] + 1,
                            scene_sample['layout_candidates'], scene_sample['layout_noise'],
                        ).to(x_object_mean.dtype)
                    else:
                        x_object_mean = model.sample_object_anchor(
                            object_head_hidden, generator=anchor_generator,
                            noise_scale=scene_sample['layout_noise'] if scene_generator is not None else 1.0,
                        ).to(x_object_mean.dtype)

                identity_indices, identity_probabilities = model.identity_head.predict(
                    object_status_result, category_ids
                )
                if scene_generator is not None and scene_sample['identity_rank']:
                    # (seed+1)-th most likely asset of the category, wrapping when
                    # the category has fewer assets.
                    # Rank by logits: at logit_scale ~100 a valid asset's softmax can underflow to 0.
                    identity_logits = model.identity_head(object_status_result, category_ids)
                    ranked = identity_logits.argsort(dim=-1, descending=True)
                    n_assets = (
                        category_ids[:, None] == model.identity_head.asset_category_ids
                    ).sum(dim=-1).clamp_min(1)
                    pick = (scene_sample['seed'] + 1) % n_assets
                    identity_indices = ranked.gather(-1, pick[:, None])[:, 0]
                elif scene_generator is not None:
                    # Masked (other-category) assets have probability 0 -> -inf.
                    identity_indices = sample_from_logits(
                        torch.log(identity_probabilities.float()),
                        scene_sample['object_temperature'], 1.0, scene_generator,
                    )[:, 0]

                if not dynamic_object:
                    for t in range(n_valid_object):
                        obj_name = object_token_id_to_name[object_id_valid[t]]
                        instance_name = unique_instance_key(obj_name, pred_objects)
                        asset = identity_catalog.assets[identity_indices[t].item()]
                        extent = canonical_extent_m(asset) or [1.0, 1.0, 1.0]
                        pred_objects[instance_name] = {
                            'rot': x_object_mean[t, 0:6],
                            'transl': x_object_mean[t, 6:9],
                            'bbox': torch.tensor(
                                extent,
                                dtype=x_object_mean.dtype,
                                device=x_object_mean.device,
                            ),
                            'asset_id': asset['asset_id'],
                            'mesh_path': asset.get('mesh_path'),
                            'identity_probability': identity_probabilities[
                                t, identity_indices[t]
                            ].item(),
                        } # imu_batches[0]['objects']

            # decode motion tokens (skip when all_continuous_recon, x_recon_precomputed is used)
            if not text_only and x_recon_precomputed is None:
                x_static_idx = result_static_idx.reshape(n_time_token, -1)
                x_mean_idx = result_mean_idx
                x_std_idx = result_std_idx

                x_static_logits = result_static_logits.reshape(n_time_token, -1, n_static_bins)
                x_mean_logits = result_mean_logits.reshape(n_time_token, -1, n_dynamic_bins)
                x_std_logits = result_std_logits.reshape(n_time_token, -1, n_dynamic_bins)

                x_mean = time_series_quantizer.mean_quantizer.decode(x_mean_idx.reshape(-1)).reshape(n_time_token, 1, -1)
                x_std = time_series_quantizer.std_quantizer.decode(x_std_idx.reshape(-1)).reshape(n_time_token, 1, -1)

        # Text metrics and the decoded caption do not depend on motion; compute
        # them first so text-only evaluation can stop right after.
        text_metrics = _text_metrics(text_logits, labels, imu_batches)
        empty_gt_text = text_metrics is None
        text_top1_acc = text_top5_acc = text_ce = None
        if not empty_gt_text:
            text_top1_acc, text_top5_acc, text_ce = text_metrics
        if text_diverse_result is not None and text_logits:
            greedy_lp = [
                float(torch.log_softmax(lg[0].float(), dim=-1)[int(t)])
                for lg, t in zip(text_logits, text_result)
            ]
            text_diverse_result['greedy'] = {
                'n_tokens': len(greedy_lp),
                'logprob': float(sum(greedy_lp)),
                'ended': bool(int(text_result[-1]) == eot_token),
            }
        text_result = torch.tensor(text_result, device=accelerator.device, dtype=torch.long) # decode token to string
        text_result = uni_prompting.text_tokenizer.decode(text_result, skip_special_tokens=True)
        n_evaluated += 1

        if structured_output:
            sample_file_id = str(sample_idx[0]).rsplit('/', 1)[-1]
            sample_file_id = re.sub(r"[^A-Za-z0-9._-]", "_", sample_file_id)
            txt_name = f"{sample_file_id}.txt"
        else:
            txt_name = f"id_{imu_batch_idx}_step_{global_step}.txt"
        txt_path = os.path.join(save_folder, txt_name)

        if text_only:
            # Text is the only output: no motion/object metrics and no .npy
            # sample (it would only hold motion and objects).
            if generate_number and not empty_gt_text:
                text_top1_accs.append(text_top1_acc.item())
                text_top5_accs.append(text_top5_acc.item())
                text_ces.append(text_ce.item())
            with open(txt_path, "w") as f:
                _write_sample_header(f, sample_idx, invalid_imu_id_list, imu_batches)
                _write_text_lines(f, empty_gt_text, text_top1_acc, text_top5_acc, text_ce, text_result, imu_batches)
            print(f"Saved metrics to {txt_path}")
            # The shared record writer sits past this `continue` and needs
            # motion/object state a text-only pass never produces, so emit the
            # caption record here. Without it records-*.jsonl stays empty and
            # the offline caption metrics have nothing to score -- there are no
            # objects and no tracks/ entry to write either.
            if records_file is not None:
                record = _base_eval_record(
                    sample_idx, invalid_imu_id_list, imu_batches, text_result, None
                )
                if text_diverse_result is not None:
                    record['text_diverse'] = text_diverse_result
                record['object_matches'] = []
                record['objects'] = {'pred': {}, 'gt': {}}
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_file.flush()
            continue

        # calculate metrics
        if motion_not_decoded or (x_recon_precomputed is not None and not partial_continuous_recon):
            # Teacher-forced motion: the motion tokens are ground truth, so there is
            # no motion prediction to score.
            status_top1_acc = torch.tensor(0.0)
            status_top5_acc = torch.tensor(0.0)
            status_ce = torch.tensor(0.0)
        elif partial_continuous_recon:
            # partial: motion (transl+orient) discrete, compute acc on motion part only
            n_motion_var = 3 + model.rot_dof
            labels_mean_motion = labels[0][0][0][:, :n_motion_var]
            labels_std_motion = labels[1][0][0][:, :n_motion_var]
            labels_static_motion = labels[2][0][0][:, :n_motion_var]
            status_top1_acc_static = topk_accuracy(result_static_logits, labels_static_motion, k=1)
            status_top1_acc_mean = topk_accuracy(result_mean_logits, labels_mean_motion, k=1)
            status_top1_acc_std = topk_accuracy(result_std_logits, labels_std_motion, k=1)
            status_top1_acc = (status_top1_acc_mean + status_top1_acc_std + status_top1_acc_static) / 3.0

            status_top5_acc_mean = topk_accuracy(result_mean_logits, labels_mean_motion, k=5)
            status_top5_acc_std = topk_accuracy(result_std_logits, labels_std_motion, k=5)
            status_top5_acc_static = topk_accuracy(result_static_logits, labels_static_motion, k=5)
            status_top5_acc = (status_top5_acc_mean + status_top5_acc_std + status_top5_acc_static) / 3.0

            # Eval-side token CE; a sample whose whole token-head channel group is masked
            # (motion_supervise) has only -100 targets -> report 0 instead of NaN.
            def _ce_or_zero(logits, targets):
                if not (targets != -100).any():
                    return logits.sum() * 0.0
                return F.cross_entropy(logits, targets, reduction='mean', ignore_index=-100)
            status_ce_mean = _ce_or_zero(result_mean_logits.reshape(-1, n_dynamic_bins), labels_mean_motion.reshape(-1))
            status_ce_std = _ce_or_zero(result_std_logits.reshape(-1, n_dynamic_bins), labels_std_motion.reshape(-1))
            status_ce_static = _ce_or_zero(result_static_logits.reshape(-1, n_static_bins), labels_static_motion.reshape(-1))
            status_ce = (status_ce_mean + status_ce_std + status_ce_static) / 3.0
        else:
            status_top1_acc_static = topk_accuracy(x_static_logits, labels[2][0][0], k=1)
            status_top1_acc_mean = topk_accuracy(x_mean_logits, labels[0][0][0], k=1)
            status_top1_acc_std = topk_accuracy(x_std_logits, labels[1][0][0], k=1)
            status_top1_acc = (status_top1_acc_mean + status_top1_acc_std + status_top1_acc_static) / 3.0

            status_top5_acc_mean = topk_accuracy(x_mean_logits, labels[0][0][0], k=5)
            status_top5_acc_std = topk_accuracy(x_std_logits, labels[1][0][0], k=5)
            status_top5_acc_static = topk_accuracy(x_static_logits, labels[2][0][0], k=5)
            status_top5_acc = (status_top5_acc_mean + status_top5_acc_std + status_top5_acc_static) / 3.0

            status_ce_mean = F.cross_entropy(x_mean_logits.reshape(-1, n_dynamic_bins), labels[0][0][0].reshape(-1), reduction='mean')
            status_ce_std = F.cross_entropy(x_std_logits.reshape(-1, n_dynamic_bins), labels[1][0][0].reshape(-1), reduction='mean')
            status_ce_static = F.cross_entropy(x_static_logits.reshape(-1, n_static_bins), labels[2][0][0].reshape(-1), reduction='mean')
            status_ce = (status_ce_mean + status_ce_std + status_ce_static) / 3.0

        if x_recon_precomputed is not None:
            x_recon = x_recon_precomputed  # [n_time_token, chunck_size, 3+22*rot_dof]
        else:
            x_static = static_embedder(x_static_idx) # [n_time_token, n_var, embedding_dim]
            _, n_var, embedding_dim = x_static.shape
            x_static = x_static.reshape(-1, embedding_dim, 1) # [n_time_token * n_var, embedding_dim, 1]
            x_static = time_series_quantizer.model.decoder(x_static, model.compression_rate)  # [n_time_token * n_var, chunck_size]
            x_static = x_static.reshape(n_time_token, -1, model.compression_rate) # [n_time_token, n_var, chunck_size]
            x_static = x_static.permute(0, 2, 1) # [n_time, chunck_size, n_var]

            # import pdb; pdb.set_trace()
            x_recon = x_static * x_std + x_mean  # [n_window, chunck_size, 3+22*rot_dof]

        rot_rep = '6d'
        rot_dof = 6
        ntime = n_time_token * model.compression_rate

        pred_transl = x_recon[:, :, 0:3].float().reshape(ntime, 3) # [n_time, 3], local space, relative translation
        pred_orient = x_recon[:, :, 3:3+rot_dof].float().reshape(ntime, rot_dof) # [n_time, 6], local space, relative rotation
        pred_pose = x_recon[:, :, 3+rot_dof:3+22*rot_dof].float() # [n_time, 21, 6]


        if dynamic_object and n_valid_object > 0:
            # Use the same temporal motion queries that drive human-motion
            # prediction. This avoids GT-motion leakage and train/test mismatch.
            if motion_query_hidden_states is None:
                raise RuntimeError("Missing motion query hidden states")
            if motion_query_hidden_states.shape != (
                n_time_token,
                model.llm_hidden_size,
            ):
                raise RuntimeError(
                    "Unexpected motion query shape: "
                    f"{tuple(motion_query_hidden_states.shape)}"
                )

            n_obj_token = len(object_status_result)
            root_features = None
            if getattr(model, 'object_track_root_feature', False):
                # Decoded human root trajectory (same representation as the
                # GT labels used for teacher forcing in training).
                root_features = model.root_trajectory_features(
                    x_recon.reshape(1, ntime, -1)[..., :3 + rot_dof], relative=True
                ).expand(n_obj_token, -1, -1)
            use_state_gate = bool(getattr(model, 'object_track_state_head', False))
            object_moving = None
            if use_state_gate:
                object_residual, object_state_logits = model.predict_object_tracks(
                    object_status_result,
                    motion_query_hidden_states[None].expand(n_obj_token, -1, -1),
                    root=root_features,
                    return_state=True,
                    geometry=object_geometry_result,
                )
            else:
                object_residual = model.predict_object_tracks(
                    object_status_result,
                    motion_query_hidden_states[None].expand(n_obj_token, -1, -1),
                    root=root_features,
                    geometry=object_geometry_result,
                )
            if object_residual.shape != (n_obj_token, ntime, 9):
                raise RuntimeError(
                    f"Unexpected object track shape: {tuple(object_residual.shape)}"
                )
            if use_state_gate:
                # Predicted per-frame motion state: static runs hold the world
                # pose, moving runs continue the body-frame residual from the
                # held pose (same gating as the teacher-forced training loss).
                object_moving = model.decode_motion_state(object_state_logits)
                x_obj_dynamic_mean_pred = model.gated_compose_object_tracks(
                    x_object_mean, object_residual, root_features, object_moving
                ).to(x_object_mean.dtype)
            else:
                # Training supervises frame-0-relative tracks; anchor them at the
                # predicted first-frame pose (rotation composed or added per
                # model.object_track_rotation_residual).
                x_obj_dynamic_mean_pred = model.compose_object_tracks(
                    x_object_mean, object_residual, root=root_features
                ).to(x_object_mean.dtype)

            # The constant ground plane is excluded from the track and
            # motion-state losses (dataset_process/object_taxonomy.py
            # is_static_ground), so its residual and state logit are never
            # trained. Hold it at the anchor pose the flow head placed it at
            # instead of letting an untrained logit gate the floor into motion.
            static_rows = [
                t for t in range(n_valid_object)
                if is_static_ground(object_token_id_to_name[object_id_valid[t]])
            ]
            if static_rows:
                static_rows = torch.tensor(
                    static_rows, dtype=torch.long, device=x_obj_dynamic_mean_pred.device
                )
                x_obj_dynamic_mean_pred[static_rows] = x_object_mean[
                    static_rows, None, :
                ].to(x_obj_dynamic_mean_pred.dtype)
                if object_moving is not None:
                    object_moving[static_rows] = False

            for t in range(n_valid_object):
                obj_name = object_token_id_to_name[object_id_valid[t]]
                instance_name = unique_instance_key(obj_name, pred_objects)
                asset = identity_catalog.assets[identity_indices[t].item()]
                extent = torch.tensor(
                    canonical_extent_m(asset) or [1.0, 1.0, 1.0],
                    dtype=x_obj_dynamic_mean_pred.dtype,
                    device=x_obj_dynamic_mean_pred.device,
                ).expand(ntime, -1)
                pred_objects[instance_name] = {
                    'rot': x_obj_dynamic_mean_pred[t, :, 0:6],
                    'transl': x_obj_dynamic_mean_pred[t, :, 6:9],
                    'bbox': extent,
                    'asset_id': asset['asset_id'],
                    'mesh_path': asset.get('mesh_path'),
                    'identity_probability': identity_probabilities[
                        t, identity_indices[t]
                    ].item(),
                } # imu_batches[0]['objects']
                if object_moving is not None:
                    pred_objects[instance_name]['moving'] = object_moving[t].float()

        gt_object_source_names = [
            (canonical_category(name), name)
            for name in imu_batches[0]['objects'].keys()
        ]
        pred_object_instance_names = list(pred_objects)
        pred_objects_names = [
            canonical_category(name) for name in pred_object_instance_names
        ]
        unmatched_pred = list(enumerate(pred_objects_names))
        object_matches = []
        for category, source_name in gt_object_source_names:
            match_position = next(
                (
                    position
                    for position, (_, predicted_category) in enumerate(unmatched_pred)
                    if predicted_category == category
                ),
                None,
            )
            if match_position is None:
                continue
            pred_index, _ = unmatched_pred.pop(match_position)
            object_matches.append(
                (category, source_name, pred_object_instance_names[pred_index])
            )
        # The ground plane is excluded from every object metric: every sample
        # carries it and it is matched almost always, so counting it inflated
        # object_id_accuracy (and made ground-only MotionMillion samples score
        # 100%). A sample without non-ground GT objects scores None and drops
        # out of the mean. record['object_matches'] still lists ground; the
        # offline scorer excludes it too.
        gt_objects_names = [
            category for category, _ in gt_object_source_names if not is_static_ground(category)
        ]
        object_matches_no_ground = [
            match for match in object_matches if not is_static_ground(match[0])
        ]
        object_id_accuracy = (
            len(object_matches_no_ground) / len(gt_objects_names) if len(gt_objects_names) > 0 else None
        )
        object_rot_L1 = None
        object_transl_L1 = None
        pose_matches = [match for match in object_matches_no_ground
                        if imu_batches[0].get('object_anchor_valid', {}).get(match[1], True)]
        if len(pose_matches) > 0:
            object_rot_L1 = 0.0
            object_transl_L1 = 0.0
            for _, source_name, pred_instance_name in pose_matches:
                gt_obj_rot = imu_batches[0]['objects'][source_name]['rot'] # numpy array
                gt_obj_transl = imu_batches[0]['objects'][source_name]['transl']
                pred_obj_rot = pred_objects[pred_instance_name]['rot'].float().detach().cpu().numpy()
                pred_obj_transl = pred_objects[pred_instance_name]['transl'].float().detach().cpu().numpy()
                pred_obj_bbox = pred_objects[pred_instance_name]['bbox'].float().detach().cpu().numpy()
                object_rot_L1 += np.linalg.norm(gt_obj_rot - pred_obj_rot)
                object_transl_L1 += np.linalg.norm(gt_obj_transl - pred_obj_transl)
            object_rot_L1 /= len(pose_matches)
            object_transl_L1 /= len(pose_matches)
            if dynamic_object:
                object_rot_L1 /= ntime
                object_transl_L1 /= ntime

        if accumulate_pose:
            # recover pose
            # won't use it for now
            assert False, "Not implemented yet."
            pred_pose = pred_pose.reshape(1, ntime, 21, rot_dof)
            pred_pose = recover_absolute_rotation(pred_pose, rot_rep=rot_rep)

        if accumulate_orient:
            # recover orient
            pred_orient = pred_orient.reshape(1, ntime, 1, rot_dof)
            pred_orient = recover_absolute_rotation(pred_orient, rot_rep=rot_rep).reshape(ntime, rot_dof) # [n_time, rot_dof]

        pred_orient = convert_rotation(pred_orient, rot_rep, 'mat') # [n_time, 3, 3], world space, absolute rotation

        # recover traj, accumulate since second window
        if local_coordinate:

            raise ValueError("Not implemented yet.")

            final_abs_transl = []

            for i in range(pred_transl.shape[0]):

                if i == 0:
                    ref_R = torch.eye(3).float().to(pred_transl.device) # [3, 3]
                    ref_T = torch.zeros(3).float().to(pred_transl.device) # [3]
                else:
                    ref_R = pred_orient[i-1] # [3, 3]

                rel_transl = pred_transl[i]
                abs_transl = (ref_R @ rel_transl) + ref_T

                final_abs_transl.append(abs_transl)
                ref_T = abs_transl.clone()

            pred_transl = torch.stack(final_abs_transl, dim=0) # [n_time, 3]

        else:
            if accumulate:
                # pred_transl[1:] = pred_transl[1:] + torch.cumsum(pred_transl[:-1], dim=0).mean(dim=1, keepdim=True) # [n_window, chunck_size, 3]
                pred_transl = pred_transl.reshape(ntime, 3)
                pred_transl = torch.cumsum(pred_transl, dim=0)

        # convert to aa for visualization
        pred_orient = convert_rotation(pred_orient, 'mat', 'aa')
        pose_raw = pred_pose.reshape(-1, rot_dof)
        pred_pose = convert_rotation(pose_raw, rot_rep, 'aa')

        pred_transl = pred_transl.reshape(ntime, 3).cpu().numpy()
        pred_orient = pred_orient.reshape(ntime, 3).cpu().numpy()
        pred_pose = pred_pose.reshape(ntime, 21*3).cpu().numpy()

        if pose_filter_settings["enabled"]:
            pred_pose = lowpass_pose(
                pred_pose,
                fps=float(imu_batches[0].get("fps", fps)),
                cutoff_hz=pose_filter_settings["cutoff_hz"],
                order=pose_filter_settings["order"],
            )
            # Keep the exported 6D pose consistent with the axis-angle pose
            # used by metrics, SMPL-X, and Rerun. Training targets stay untouched.
            pose_raw = convert_rotation(
                torch.from_numpy(pred_pose.reshape(-1, 3)).to(pose_raw.device),
                "aa", "6d",
            )

        if _rad and _align_pred_yaw:
            # Heading-unknown eval: the model predicts in the (yaw-rotated)
            # input world frame, so undo the applied yaw on the prediction to
            # report metrics in the GT canonical frame. Only the root
            # orient/transl move; local joint pose is frame-independent.
            _ca, _sa = math.cos(-_rad), math.sin(-_rad)
            _Ryi = np.asarray(
                [[_ca, 0.0, _sa], [0.0, 1.0, 0.0], [-_sa, 0.0, _ca]],
                dtype=np.float32,
            )
            pred_transl = np.einsum("ij,tj->ti", _Ryi, pred_transl)
            _m = convert_rotation(
                torch.from_numpy(np.ascontiguousarray(pred_orient, np.float32)),
                "aa",
                "mat",
            ).numpy()
            _m = np.einsum("ij,tjk->tik", _Ryi, _m)
            pred_orient = convert_rotation(
                torch.from_numpy(np.ascontiguousarray(_m, np.float32)), "mat", "aa"
            ).numpy()

        gt_transl = imu_batches[0]['transl'].cpu().numpy()
        gt_orient = convert_rotation(imu_batches[0]['orient'], '6d', 'aa').cpu().numpy()
        gt_pose = convert_rotation(imu_batches[0]['pose'].reshape(-1, 6), '6d', 'aa').reshape(-1, 21*3).cpu().numpy()
        gt_pose_raw = imu_batches[0]['pose'].reshape(-1, 6).cpu().numpy()

        # (text metrics and the decoded caption were computed above)

        # Preserve the actual generator decisions for paired tokenizer studies.
        # Re-encoding ``pred`` would only produce round-trip proxy tokens and can
        # hide precisely the substitutions/bursts that a robust decoder must see.
        token_bundle = None
        if not all_continuous_recon:
            pred_static_tokens = result_static_idx.reshape(n_time_token, -1).detach().to("cpu", torch.long)
            pred_mean_tokens = result_mean_idx.reshape(n_time_token, -1).detach().to("cpu", torch.long)
            pred_std_tokens = result_std_idx.reshape(n_time_token, -1).detach().to("cpu", torch.long)
            oracle_static_tokens = labels[2][0][0].detach().to("cpu", torch.long)
            oracle_mean_tokens = labels[0][0][0].detach().to("cpu", torch.long)
            oracle_std_tokens = labels[1][0][0].detach().to("cpu", torch.long)
            if partial_continuous_recon:
                # Only trajectory + root orientation are discrete in this mode.
                n_discrete_vars = pred_static_tokens.shape[-1]
                oracle_static_tokens = oracle_static_tokens[:, :n_discrete_vars]
                oracle_mean_tokens = oracle_mean_tokens[:, :n_discrete_vars]
                oracle_std_tokens = oracle_std_tokens[:, :n_discrete_vars]
            token_bundle = {
                "layout": "window_variable",
                "stream_order": ["static", "mean", "std"],
                "compression_factor": int(model.compression_rate),
                "partial_continuous": bool(partial_continuous_recon),
                "pred": {
                    "static": pred_static_tokens.numpy(),
                    "mean": pred_mean_tokens.numpy(),
                    "std": pred_std_tokens.numpy(),
                },
                "oracle": {
                    "static": oracle_static_tokens.numpy(),
                    "mean": oracle_mean_tokens.numpy(),
                    "std": oracle_std_tokens.numpy(),
                },
                # Raw absolute motion in generator variable order.  The paired
                # evaluator applies the same traj/orient preprocessing as the
                # frozen tokenizer before computing distortion.
                "target_layout": "absolute_traj_absolute_orient6d_pose6d",
                "target_motion": labels[10][0].detach().float().cpu().numpy(),
            }

        # save the reconstructed trajectory
        sample_pred = {
            'sample_idx': sample_idx,
            'tokens': token_bundle,
            'input':{
                'imu': imu_batches[0]['imu_data'],
                'imu_positions': imu_batches[0].get('imu_positions'),
                'invalid_imu_id_list': invalid_imu_id_list,
                'imu_point_count': NUM_IMU_SENSORS - len(invalid_imu_id_list),
                'active_imu_sensor_names': [
                    sensor_name
                    for sensor_id, sensor_name in enumerate(IMU_SENSOR_NAMES)
                    if sensor_id not in invalid_imu_id_list
                ],
            },
            'pred':{
                'transl': pred_transl,
                'orient': pred_orient,
                'pose': pred_pose,
                'pose_raw': pose_raw,
                'description': text_result,
                'objects': pred_objects,
            },
            'gt':{
                'transl': gt_transl,
                'orient': gt_orient,
                'pose': gt_pose,
                'pose_raw': gt_pose_raw.reshape(-1, 6),
                'description': imu_batches[0]['description'],
                'objects': imu_batches[0]['objects']
            },
            'metadata': {
                'dataset': (
                    rerun_export_job['dataset']
                    if rerun_export_job is not None
                    else str(imu_batches[0].get('source', 'unknown'))
                ),
                'source': str(imu_batches[0].get('source', 'unknown')),
                'motion_id': str(
                    imu_batches[0].get(
                        'motion_id', str(sample_idx[0]).rsplit('/', 1)[-1]
                    )
                ),
                'actor_id': str(imu_batches[0].get('actor_id', 'unknown')),
                'fps': float(imu_batches[0].get('fps', fps)),
                'step': int(global_step),
                'layout': (
                    rerun_export_job['layout']
                    if rerun_export_job is not None
                    else postfix or None
                ),
                'eval_frames': None if eval_frames is None else int(eval_frames),
                'object_metadata': imu_batches[0].get('object_metadata', {}),
                'object_anchor_valid': imu_batches[0].get('object_anchor_valid', {}),
                'ground_plane': imu_batches[0].get('ground_plane'),
                'pose_filter': dict(pose_filter_settings),
            },
        }

        # Foot-contact trajectory correction.  The LM emits the trajectory as
        # per-frame deltas that are integrated with nothing tying the body to the
        # ground: on measured IMU the pelvis can leave the floor and stay there,
        # and walking clips travel too short.  The decoded pose is accurate, so
        # the stance foot is the better velocity source *there*; on the synthetic
        # test sets the LM trajectory is already an order of magnitude better
        # than any integrator of the pose, hence the "auto" default, which
        # restricts the pass to real-IMU sources.  experiment.eval_footlock:
        #   enabled: auto | true | false, plus footlock_translation kwargs
        #   (guard_m, stance_floor, cutoff_hz).
        _footlock_cfg = config.experiment.get("eval_footlock", None)
        _footlock_cfg = {} if _footlock_cfg is None else {
            str(_k): _v for _k, _v in dict(_footlock_cfg).items()
        }
        _footlock_mode = _footlock_cfg.pop("enabled", "auto")
        if not motion_not_decoded and footlock_enabled_for(
            sample_pred['metadata']['source'], _footlock_mode
        ):
            try:
                _footlock_transl = footlock_sample(sample_pred, **_footlock_cfg)
                if torch.is_tensor(pred_transl):
                    _footlock_transl = torch.as_tensor(
                        _footlock_transl, dtype=pred_transl.dtype, device=pred_transl.device
                    )
                pred_transl = _footlock_transl
                sample_pred['pred']['transl'] = pred_transl
                sample_pred['metadata']['footlock'] = {'applied': True, **_footlock_cfg}
            except Exception as _footlock_error:  # never let it break an evaluation
                logger.warning("Foot-lock trajectory correction failed: %s", _footlock_error)
                sample_pred['metadata']['footlock'] = {
                    'applied': False, 'error': f"{type(_footlock_error).__name__}: {_footlock_error}"
                }

        mpjpe = compute_mpjpe(sample_pred)[0]
        sample_pred['metadata'].update({
            'mpjpe_mm': float(mpjpe),
            'object_id_accuracy': (
                None if object_id_accuracy is None else float(object_id_accuracy)
            ),
            'object_rotation_l1': (
                None if object_rot_L1 is None else float(object_rot_L1)
            ),
            'object_translation_l1': (
                None if object_transl_L1 is None else float(object_transl_L1)
            ),
        })

        if rerun_export_job is not None:
            current_sample_id = (
                str(sample_idx[0])
                if isinstance(sample_idx, (list, tuple)) and len(sample_idx) == 1
                else str(sample_idx)
            )
            if (
                current_sample_id in rerun_export_job['target_sample_ids']
                and current_sample_id not in rerun_status['matched']
            ):
                rerun_status['matched'].append(current_sample_id)
                output_dir = Path(rerun_export_job['output_dirs'][current_sample_id])
                try:
                    from visualize.eval_rerun import export_eval_pair

                    export_eval_pair(
                        sample_pred,
                        output_dir,
                        smplx_model=rerun_export_job['smplx_model'],
                        asset_roots=rerun_export_job['asset_roots'],
                        record_fps=rerun_export_job['record_fps'],
                        max_seconds=rerun_export_job['max_seconds'],
                        device="cpu",
                    )
                    rerun_status['exported'].append(current_sample_id)
                    logger.info(
                        "Exported full-eval Rerun pair for %s to %s",
                        current_sample_id,
                        output_dir,
                    )
                except Exception as error:
                    rerun_status['errors'][current_sample_id] = (
                        f"{type(error).__name__}: {error}"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    error_path = output_dir / "error.json"
                    error_path.write_text(
                        json.dumps(
                            {
                                "status": "error",
                                "sample_id": current_sample_id,
                                "dataset": rerun_export_job['dataset'],
                                "step": int(global_step),
                                "layout": rerun_export_job['layout'],
                                "eval_frames": rerun_export_job.get('eval_frames'),
                                "error": rerun_status['errors'][current_sample_id],
                            },
                            indent=2,
                            sort_keys=True,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    logger.exception(
                        "Failed to export full-eval Rerun pair for %s",
                        current_sample_id,
                    )

        if generate_number:
            # L1 loss
            traj_error = np.abs(pred_transl - gt_transl).mean()
            orient_error = np.abs(pred_orient - gt_orient).mean()
            pose_error = np.abs(pred_pose - gt_pose).mean()

            traj_errors.append(traj_error)
            orient_errors.append(orient_error)
            pose_errors.append(pose_error)

            status_top1_accs.append(status_top1_acc.item())
            status_top5_accs.append(status_top5_acc.item())
            status_ces.append(status_ce.item())

            if not empty_gt_text:
                text_top1_accs.append(text_top1_acc.item())
                text_top5_accs.append(text_top5_acc.item())
                text_ces.append(text_ce.item())

            mpjpes.append(mpjpe)
            if object_id_accuracy is not None:
                object_id_accuracies.append(float(object_id_accuracy))
            if object_rot_L1 is not None:
                object_rot_l1_errors.append(float(object_rot_L1))
                object_transl_l1_errors.append(float(object_transl_L1))

        if save_sample:
            # save current sample
            np.save(os.path.join(save_folder, f"id_{imu_batch_idx}_step_{global_step}.npy"), sample_pred)
            sample_pred['step'] = global_step
            print(f"Saved sample {imu_batch_idx} at step {global_step}")

            # keep at most max_sample_keep_num npy samples for this imu_batch_idx
            pattern = f"id_{imu_batch_idx}_step_"
            npy_files = [
                fname for fname in os.listdir(save_folder)
                if fname.startswith(pattern) and fname.endswith(".npy")
            ]
            if len(npy_files) > max_sample_keep_num:
                # parse global_step from filename "id_{imu_batch_idx}_step_{global_step}.npy"
                def _extract_step(name: str) -> int:
                    try:
                        base = os.path.splitext(name)[0]
                        step_str = base.split("_step_")[-1]
                        return int(step_str)
                    except Exception:
                        return float("inf")

                npy_files.sort(key=_extract_step)
                files_to_remove = npy_files[:-max_sample_keep_num]
                for old_name in files_to_remove:
                    old_path = os.path.join(save_folder, old_name)
                    try:
                        os.remove(old_path)
                        print(f"Removed old sample file: {old_path}")
                    except OSError as e:
                        print(f"Failed to remove old sample file {old_path}: {e}")

        # record accuracy and ce
        with open(txt_path, "w") as f:
            _write_sample_header(f, sample_idx, invalid_imu_id_list, imu_batches)
            f.write(f"status_top1_acc: {status_top1_acc.item()*100:.4f}%, status_top5_acc: {status_top5_acc.item()*100:.4f}%, status_ce: {status_ce.item():.4f}\n")
            f.write(f"mpjpe: {mpjpe:.4f} mm\n")
            _write_text_lines(f, empty_gt_text, text_top1_acc, text_top5_acc, text_ce, text_result, imu_batches)
            # write the information of objects
            f.write(f"object_id_accuracy: {object_id_accuracy*100 if object_id_accuracy is not None else 'N/A'}%\n")
            if object_rot_L1 is not None:
                f.write(f"object_rot_L1: {object_rot_L1:.4f}, object_transl_L1: {object_transl_L1:.4f}\n")
            f.write(f"pred_objects:\n")
            for obj_name in pred_objects.keys():
                f.write(f"  {obj_name}: rot: {pred_objects[obj_name]['rot'].tolist()}, transl: {pred_objects[obj_name]['transl'].tolist()}\n")
            f.write(f"gt_objects:\n")
            if len(imu_batches[0]['objects']) > 0:
                for obj_key, obj in imu_batches[0]['objects'].items():
                    obj = imu_batches[0]['objects'][obj_key]
                    f.write(f"  {obj_key}: rot: {obj['rot'].tolist()}, transl: {obj['transl'].tolist()}\n")
        print(f"Saved metrics to {txt_path}")

        if records_file is not None:
            gt_object_metadata = imu_batches[0].get('object_metadata', {}) or {}
            gt_entries, pred_entries, track_arrays = {}, {}, {}
            for obj_key, obj in imu_batches[0]['objects'].items():
                gt_entries[obj_key] = _record_object_entry(
                    obj_key,
                    obj,
                    identity_catalog.resolve_asset_id(
                        obj_key, gt_object_metadata.get(obj_key, {})
                    ),
                )
                track_arrays[f"gt|{obj_key}|rot"] = _record_numpy(obj['rot'])
                track_arrays[f"gt|{obj_key}|transl"] = _record_numpy(obj['transl'])
                valid_mask = imu_batches[0].get('object_valid_mask', {}).get(obj_key)
                motion_mask = imu_batches[0].get('object_motion_mask', {}).get(obj_key)
                if valid_mask is not None:
                    track_arrays[f"gt|{obj_key}|valid_mask"] = _record_numpy(
                        valid_mask
                    ).astype(bool)
                if motion_mask is not None:
                    track_arrays[f"gt|{obj_key}|motion_mask"] = _record_numpy(
                        motion_mask
                    ).astype(bool)
            for obj_key, obj in pred_objects.items():
                pred_entries[obj_key] = _record_object_entry(
                    obj_key, obj, obj.get('asset_id')
                )
                track_arrays[f"pred|{obj_key}|rot"] = _record_numpy(obj['rot'])
                track_arrays[f"pred|{obj_key}|transl"] = _record_numpy(obj['transl'])
            record = _base_eval_record(
                sample_idx, invalid_imu_id_list, imu_batches, text_result, float(mpjpe)
            )
            if text_diverse_result is not None:
                record['text_diverse'] = text_diverse_result
            if scene_sample is not None:
                record['scene_sample'] = dict(scene_sample)
            record_sample_id = record['sample_id']
            # Category matching as the summary scores it, so an offline
            # scorer can reproduce or replace the matching rule.
            record['object_matches'] = [list(match) for match in object_matches]
            record['objects'] = {'pred': pred_entries, 'gt': gt_entries}
            if object_decode_info:
                record['object_decode'] = object_decode_info
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_file.flush()
            # Ground-only samples (MotionMillion) carry no object information
            # worth a file; every HOI sample gets its full tracks.
            if any(
                not is_static_ground(key.split('|')[1]) for key in track_arrays
            ):
                np.savez_compressed(
                    os.path.join(
                        tracks_dir,
                        re.sub(r'[^A-Za-z0-9._-]', '_', record_sample_id) + '.npz',
                    ),
                    **track_arrays,
                )

        if (
            rerun_export_job is not None
            and rerun_export_job.get('stop_after_export', False)
            and set(rerun_status['matched']) == set(rerun_export_job['target_sample_ids'])
        ):
            logger.info("Rerun-only pass: stopping after the configured samples.")
            break

    if records_file is not None:
        records_file.close()

    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    # gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)

    if rerun_export_job is not None:
        gathered_rerun_status = (
            gather_object([rerun_status])
            if accelerator.num_processes > 1
            else [rerun_status]
        )
        matched_ids = set()
        export_errors = {}
        for status in gathered_rerun_status:
            matched_ids.update(status['matched'])
            export_errors.update(status['errors'])
        missing_ids = [
            sample_id
            for sample_id in rerun_export_job['target_sample_ids']
            if sample_id not in matched_ids
        ]
        rerun_error = None
        if missing_ids:
            rerun_error = (
                "Configured full-eval Rerun sample(s) were not evaluated: "
                f"{missing_ids}"
            )
            if accelerator.is_main_process:
                logger.error(rerun_error)
                for sample_id in missing_ids:
                    output_dir = Path(rerun_export_job['output_dirs'][sample_id])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "error.json").write_text(
                        json.dumps(
                            {
                                "status": "error",
                                "sample_id": sample_id,
                                "dataset": rerun_export_job['dataset'],
                                "step": int(global_step),
                                "layout": rerun_export_job['layout'],
                                "eval_frames": rerun_export_job.get('eval_frames'),
                                "error": rerun_error,
                            },
                            indent=2,
                            sort_keys=True,
                        ) + "\n",
                        encoding="utf-8",
                    )
        elif export_errors:
            rerun_error = next(iter(export_errors.values()))
        if rerun_error and rerun_export_job['fail_on_error']:
            raise RuntimeError(rerun_error)

    if generate_number:
        # Lists can differ by one or more samples across ranks, so use object
        # gather instead of tensor gather (which requires equal first dims).
        if accelerator.num_processes > 1:
            local_metrics = {
                "traj_errors": traj_errors,
                "orient_errors": orient_errors,
                "pose_errors": pose_errors,
                "status_top1_accs": status_top1_accs,
                "status_top5_accs": status_top5_accs,
                "status_ces": status_ces,
                "text_top1_accs": text_top1_accs,
                "text_top5_accs": text_top5_accs,
                "text_ces": text_ces,
                "mpjpes": mpjpes,
                "object_id_accuracies": object_id_accuracies,
                "object_rot_l1_errors": object_rot_l1_errors,
                "object_transl_l1_errors": object_transl_l1_errors,
                "evaluated": [n_evaluated],
            }
            gathered_metrics = gather_object([local_metrics])
            merged_metrics = {
                key: list(itertools.chain.from_iterable(rank[key] for rank in gathered_metrics))
                for key in local_metrics
            }
            traj_errors = merged_metrics["traj_errors"]
            orient_errors = merged_metrics["orient_errors"]
            pose_errors = merged_metrics["pose_errors"]
            status_top1_accs = merged_metrics["status_top1_accs"]
            status_top5_accs = merged_metrics["status_top5_accs"]
            status_ces = merged_metrics["status_ces"]
            text_top1_accs = merged_metrics["text_top1_accs"]
            text_top5_accs = merged_metrics["text_top5_accs"]
            text_ces = merged_metrics["text_ces"]
            mpjpes = merged_metrics["mpjpes"]
            object_id_accuracies = merged_metrics["object_id_accuracies"]
            object_rot_l1_errors = merged_metrics["object_rot_l1_errors"]
            object_transl_l1_errors = merged_metrics["object_transl_l1_errors"]
            n_evaluated = sum(merged_metrics["evaluated"])

        # All ranks participate in the collectives above; only rank zero writes
        # the shared summary and tracker record.
        if not accelerator.is_main_process:
            return

        # Samples without a GT caption (e.g. motion-only training) contribute
        # no text metrics. Skip the text keys entirely when nothing was scored,
        # instead of logging NaN from an empty mean.
        n_text_scored = len(text_top5_accs)
        has_text_metrics = n_text_scored > 0
        if has_text_metrics:
            mean_text_top1_acc = float(np.mean(text_top1_accs))
            mean_text_top5_acc = float(np.mean(text_top5_accs))
            mean_text_ce = float(np.mean(text_ces))
        else:
            mean_text_top1_acc = mean_text_top5_acc = mean_text_ce = None

        # add to wandb
        metric_split = f"{split}_{postfix}" if postfix else split
        full_eval_metrics = {}
        if has_text_metrics:
            full_eval_metrics.update({
                f"metric/token_acc/{metric_split}_text_top1_acc": mean_text_top1_acc,
                f"metric/token_acc/{metric_split}_text_top5_acc": mean_text_top5_acc,
                f"metric/cross_entropy/{metric_split}_text_ce": mean_text_ce,
            })
        summary = {
            "split": split,
            "step": int(global_step),
            "layout": postfix or None,
            "eval_frames": None if eval_frames is None else int(eval_frames),
            "evaluated_samples": int(n_evaluated),
            "text_only": bool(text_only),
            "gt_motion_input": bool(gt_motion_input),
            "pose_filter": dict(pose_filter_settings),
            "text_scored_samples": int(n_text_scored),
            "mean_text_top1_acc": mean_text_top1_acc,
            "mean_text_top5_acc": mean_text_top5_acc,
            "mean_text_ce": mean_text_ce,
        }

        if not text_only:
            # save the errors
            traj_errors = np.array(traj_errors)
            orient_errors = np.array(orient_errors)
            pose_errors = np.array(pose_errors)

            status_top1_accs = np.array(status_top1_accs)
            status_top5_accs = np.array(status_top5_accs)
            status_ces = np.array(status_ces)

            mean_traj_error = traj_errors.mean()
            mean_orient_error = orient_errors.mean()
            mean_pose_error = pose_errors.mean()

            mean_status_top1_acc = status_top1_accs.mean()
            mean_status_top5_acc = status_top5_accs.mean()
            mean_status_ce = status_ces.mean()

            mean_mpjpe = np.array(mpjpes).mean()
            mean_object_id_accuracy = (
                float(np.mean(object_id_accuracies))
                if object_id_accuracies
                else None
            )
            mean_object_rot_l1 = (
                float(np.mean(object_rot_l1_errors))
                if object_rot_l1_errors
                else None
            )
            mean_object_transl_l1 = (
                float(np.mean(object_transl_l1_errors))
                if object_transl_l1_errors
                else None
            )

            # logger.info(f"Split {split} step {global_step}: mean traj error: {mean_traj_error:.4f}, mean orient error: {mean_orient_error:.4f}, mean pose error: {mean_pose_error:.4f}")

            full_eval_metrics.update({
                f"metric/L1_error/{metric_split}_traj": mean_traj_error,
                f"metric/L1_error/{metric_split}_orient": mean_orient_error,
                f"metric/L1_error/{metric_split}_pose": mean_pose_error,
                f"metric/token_acc/{metric_split}_status_top1_acc": mean_status_top1_acc,
                f"metric/token_acc/{metric_split}_status_top5_acc": mean_status_top5_acc,
                f"metric/cross_entropy/{metric_split}_status_ce": mean_status_ce,
                f"metric/mpjpe/{metric_split}_mpjpe": mean_mpjpe,
            })
            if mean_object_id_accuracy is not None:
                full_eval_metrics[
                    f"metric/object/{metric_split}_id_accuracy"
                ] = mean_object_id_accuracy
            if mean_object_rot_l1 is not None:
                full_eval_metrics[
                    f"metric/object/{metric_split}_rotation_l1"
                ] = mean_object_rot_l1
                full_eval_metrics[
                    f"metric/object/{metric_split}_translation_l1"
                ] = mean_object_transl_l1

            summary.update({
                "mean_traj_error": float(mean_traj_error),
                "mean_orient_error": float(mean_orient_error),
                "mean_pose_error": float(mean_pose_error),
                "mean_status_top1_acc": float(mean_status_top1_acc),
                "mean_status_top5_acc": float(mean_status_top5_acc),
                "mean_status_ce": float(mean_status_ce),
                "mean_mpjpe_mm": float(mean_mpjpe),
                "object_evaluated_samples": len(object_id_accuracies),
                "object_matched_samples": len(object_rot_l1_errors),
                "mean_object_id_accuracy": mean_object_id_accuracy,
                "mean_object_rotation_l1": mean_object_rot_l1,
                "mean_object_translation_l1": mean_object_transl_l1,
            })

        accelerator.log(full_eval_metrics, step=global_step)

        if eval_output_root is not None:
            os.makedirs(eval_output_root, exist_ok=True)
            if structured_output:
                summary_path = os.path.join(eval_output_root, "summary.txt")
            else:
                summary_stem = postfix or split
                summary_path = os.path.join(
                    eval_output_root, f"{summary_stem}_summary.txt"
                )
        else:
            summary_path = os.path.join(
                save_folder, f"{split}_step_{global_step}_summary.txt"
            )

        if invalid_imu_id != "random":
            summary["invalid_imu_id_list"] = list(invalid_imu_id)
            summary["active_imu_sensor_names"] = [
                sensor_name
                for sensor_id, sensor_name in enumerate(IMU_SENSOR_NAMES)
                if sensor_id not in invalid_imu_id
            ]
            summary["imu_point_count"] = len(summary["active_imu_sensor_names"])

        with open(summary_path, "w") as f:
            if invalid_imu_id != "random":
                f.write(f"imu_point_count: {summary['imu_point_count']}\n")
                f.write(f"active_imu_sensor_names: {summary['active_imu_sensor_names']}\n")
                f.write(f"invalid_imu_id_list: {invalid_imu_id}\n")
            f.write(f"evaluated_samples: {n_evaluated}\n")
            if gt_motion_input:
                f.write("gt_motion_input: True (motion is teacher-forced ground truth, not predicted)\n")
            if text_only:
                f.write("text_only: True (motion and objects are not decoded)\n")
            else:
                f.write(f"mean_traj_error: {mean_traj_error:.4f}, mean_orient_error: {mean_orient_error:.4f}, mean_pose_error: {mean_pose_error:.4f}\n")
                f.write(f"mean_status_top1_acc: {mean_status_top1_acc*100:.2f}%, mean_status_top5_acc: {mean_status_top5_acc*100:.2f}%, mean_status_ce: {mean_status_ce:.4f}\n")
            if has_text_metrics:
                f.write(f"text_scored_samples: {n_text_scored}\n")
                f.write(f"mean_text_top1_acc: {mean_text_top1_acc*100:.2f}%, mean_text_top5_acc: {mean_text_top5_acc*100:.2f}%, mean_text_ce: {mean_text_ce:.4f}\n")
            else:
                f.write("text metrics: n/a (no evaluated sample has a GT caption)\n")
            if not text_only:
                f.write(f"mean_mpjpe: {mean_mpjpe:.2f} mm\n")
                if mean_object_id_accuracy is not None:
                    f.write(
                        f"object_evaluated_samples: {len(object_id_accuracies)}, "
                        f"object_matched_samples: {len(object_rot_l1_errors)}\n"
                    )
                    f.write(
                        f"mean_object_id_accuracy: {mean_object_id_accuracy*100:.2f}%\n"
                    )
                if mean_object_rot_l1 is not None:
                    f.write(
                        f"mean_object_rotation_l1: {mean_object_rot_l1:.4f}, "
                        f"mean_object_translation_l1: {mean_object_transl_l1:.4f}\n"
                    )

        with open(os.path.splitext(summary_path)[0] + ".json", "w") as f:
            json.dump(summary, f, indent=2)

        # print the file content to console
        with open(summary_path, "r") as f:
            print(f.read())

def save_checkpoint(model, optimizer, lr_scheduler, config, accelerator, global_step, epoch=None, loss=None, ema=None):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # Handle checkpoint limit cleanup
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # This single call saves everything: model, optimizer, scheduler, RNG states
    accelerator.save_state(save_path)

    # Save unwrapped model weights to a single .bin file
    if accelerator.is_main_process:
        unwrapped_model_dir = save_path / "unwrapped_model"
        unwrapped_model_dir.mkdir(parents=True, exist_ok=True)

        # Get the unwrapped model (handles DDP/FSDP wrapping)
        unwrapped_model = accelerator.unwrap_model(model)

        # Get state dict and save to single file.  With LoRA the adapters are
        # merged into the base weights first, so the file stays a plain
        # full-model checkpoint; the raw adapters are kept next to it.
        if lora_utils.has_lora(unwrapped_model):
            state_dict = lora_utils.merged_state_dict(unwrapped_model)
            adapter_path = unwrapped_model_dir / "lora_adapter.bin"
            torch.save(lora_utils.lora_state_dict(unwrapped_model), adapter_path)
            logger.info(f"Saved LoRA adapters to {adapter_path}")
        else:
            state_dict = unwrapped_model.state_dict()
        model_path = unwrapped_model_dir / "pytorch_model.bin"
        torch.save(state_dict, model_path)

        # Calculate and print file size
        file_size_bytes = model_path.stat().st_size
        file_size_gb = file_size_bytes / (1024 ** 3)

        logger.info(f"Saved unwrapped model to {model_path}")
        logger.info(f"Model file size: {file_size_gb:.3f} GB")
        logger.info(f"State dict keys ({len(state_dict)}):")
        for key in state_dict.keys():
            logger.info(f"  - {key}")

    # Save additional metadata if needed
    if accelerator.is_main_process:
        metadata = {
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
            "model_config": {
                "vocab_size": getattr(model, 'vocab_size', None),
                "codebook_size": getattr(model.config, 'codebook_size', None) if hasattr(model, 'config') else None,
            }
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved complete checkpoint to {save_path}")

        # Persist the EMA shadow so a full-state resume can keep the smoothed
        # weights in sync with the live (optimizer) weights.
        if ema is not None:
            ema_path = save_path / "ema_model.pt"
            torch.save(ema.state_dict(), ema_path)
            logger.info(f"Saved EMA state to {ema_path}")

    return save_path

def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

if __name__ == "__main__":
    main()
