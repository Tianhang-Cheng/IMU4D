import os
import re

# disable wandb
# os.environ["WANDB_MODE"] = "disabled"

# Force GPU selection before any torch/CUDA import (overrides accelerate's device selection).
# Set TRAIN_GPU_ID=4 (or desired GPU id) in the shell or in the launch script.
if "TRAIN_GPU_ID" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TRAIN_GPU_ID"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import shutil
import time
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

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from safetensors.torch import load_file, save_file

from training.imu_dataset import IMUDataset, set_cut_length
from models import get_mask_chedule, ShowoIMU

from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from utils.rotation2 import recover_absolute_rotation, convert_rotation
from models.time_series_quant import DiscreteQuantizer

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from utils.metrics import compute_mpjpe
from dataset_process.custom_path import obj_name_path, pretrained_showo_path, imu_data_path

SYSTEM_PROMPT_LEN = 28

IMU_device_names = ['left_hip', 'right_hip', 'left_ear', 'right_ear', 'left_elbow', 'right_elbow']

from training.utils import get_config, flatten_omega_conf, AverageMeter

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

def imu_to_input(model,
                time_series_quantizer,
                imu_batches,
                accelerator,
                uni_prompting,
                mask_dtype,
                obj_name_to_id,
                object_token_bias,
                normalization_window_size:int,
                smooth_imu: bool,
                random_text: bool,
                invalid_imu_id: "random", # "random" or list[int]
                dynamic_object: bool=False,
                bidirectional_imu: bool=False,
                bidirectional_motion: bool=False,
                fps=None,
                inference_mode: bool=False,
                ):
    """
    This function is used to convert IMU batch data into input format for the model
    Argument:
        invalid_imu_id: list[int], use the invalid imu ids to generate input embeddings
    """
    # assert invalid_imu_id == "random" or isinstance(invalid_imu_id, list), 'got invalid_imu_id: {invalid_imu_id}'
    if not (invalid_imu_id == "random" or isinstance(invalid_imu_id, list)):
        import pdb; pdb.set_trace()
    if isinstance(invalid_imu_id, list):
        assert all(isinstance(x, int) and 0 <= x <= 5 for x in invalid_imu_id)

    imu_embeddings_batch = []
    query_embeddings_batch = []
    output_embeddings_batch = []
    all_embeddings_batch = []
    label_mean_batch = []
    label_std_batch = []
    label_static_batch = []
    label_text_batch = []
    label_object_batch = []
    input_imu_len_batch = []
    motion_len_batch = []

    label_mean_value_batch = []
    label_std_value_batch = []
    label_object_value_batch = []
    label_object_dynamic_value_batch = []
    label_motion_value_batch = []
    label_gt_motion_batch = []

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

    max_text_len = 0
    text_select_idxs = []
    for imu_batch in imu_batches: 

        all_description = imu_batch['description']

        if len(imu_batch) == 0 or len(all_description) == 0:
            text_select_idxs.append(-1)
            continue
            
        if random_text:
            text_select_idx = random.randrange(len(all_description))
        else:
            text_select_idx = 0
        text_select_idxs.append(text_select_idx)
        description = all_description[text_select_idx]
        text_tokens = uni_prompting.text_tokenizer(description)['input_ids']
        max_text_len = max(max_text_len, len(text_tokens))
    
    # import pdb; pdb.set_trace()

    max_obj_len = 0 # number of objects
    for imu_batch in imu_batches:
        if 'objects' in imu_batch:
            max_obj_len = max(max_obj_len, len(imu_batch['objects']))

    pad_token = uni_prompting.sptids_dict['<|pad|>']
    pad_embedding = get_tokens_embed(pad_token.to(device))  # [d_model]

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

        # randomly mask some of the imu data
        invalid_imu_id_list = invalid_imu_id
        if invalid_imu_id == "random":
            # select k invalid imu ids, k is random between 0 and 5
            # invalid_imu_id_list = random.sample([0, 1, 2, 3, 4, 5], k=random.randint(0, 5))
            # only keep certain combinations of invalid imu ids
            # IMU_device_names = ['left_hip', 'right_hip', 'left_ear', 'right_ear', 'left_elbow', 'right_elbow']
            valid_combinations = [
                # [0, 1, 2, 3, 4, 5], # 6pt
                [0, 1, 2, 4, 5], # 5pt, no left_ear
                # [0, 1, 3, 4, 5], # 5pt, no right_ear
                [0, 2, 4], # 3pt, left_hip, left_ear, left_elbow
                [1, 2, 4], # 3pt, right_hip, left_ear, left_elbow
                [0, 2, 5], # 3pt, left_hip, left_ear, right_elbow
                [1, 2, 5], # 3pt, right_hip, left_ear, right_elbow
            ]
            full_list = [0, 1, 2, 3, 4, 5]
            invalid_imu_id_list = [list(set(full_list) - set(combination)) for combination in valid_combinations]
            invalid_imu_id_list = random.choice(invalid_imu_id_list)

        mask_token_id = torch.tensor(accelerator.unwrap_model(model).mask_token_id, dtype=torch.long, device=device)

        if 0 in invalid_imu_id_list:
            imu_0_embeddings = get_tokens_embed(mask_token_id) # [1, d_model]
            imu_0_embeddings = torch.repeat_interleave(imu_0_embeddings[None], n_token, dim=0) # [n_token, d_model]
        else:
            imu_0 = imu_data[:, 0].to(mask_dtype).to(device)  # [len, 15] # 15 = 3 (acc) + 3 (gyr) + 9 (orient 3x3)
            imu_0 = imu_0.reshape(-1, normalization_window_size * 15) # [n_token, window_size * 15]
            imu_0_embeddings = model.imu_aggregator(imu_0) # [n_token, d_model]

        if 1 in invalid_imu_id_list:
            imu_1_embeddings = get_tokens_embed(mask_token_id) # [1, d_model]
            imu_1_embeddings = torch.repeat_interleave(imu_1_embeddings[None], n_token, dim=0) # [n_token, d_model]
        else:
            imu_1 = imu_data[:, 1].to(mask_dtype).to(device)  # [len, 15]
            imu_1 = imu_1.reshape(-1, normalization_window_size * 15) # [n_token, window_size * 15]
            imu_1_embeddings = model.imu_aggregator(imu_1) # [n_token, d_model]
        
        if 2 in invalid_imu_id_list:
            imu_2_embeddings = get_tokens_embed(mask_token_id) # [1, d_model]
            imu_2_embeddings = torch.repeat_interleave(imu_2_embeddings[None], n_token, dim=0) # [n_token, d_model]
        else:
            imu_2 = imu_data[:, 2].to(mask_dtype).to(device)  # [len, 15]
            imu_2 = imu_2.reshape(-1, normalization_window_size * 15) # [n_token, window_size * 15]
            imu_2_embeddings = model.imu_aggregator(imu_2) # [n_token, d_model]

        if 3 in invalid_imu_id_list:
            imu_3_embeddings = get_tokens_embed(mask_token_id) # [1, d_model]
            imu_3_embeddings = torch.repeat_interleave(imu_3_embeddings[None], n_token, dim=0) # [n_token, d_model]
        else:
            imu_3 = imu_data[:, 3].to(mask_dtype).to(device)  # [len, 15]
            imu_3 = imu_3.reshape(-1, normalization_window_size * 15) # [n_token, window_size * 15]
            imu_3_embeddings = model.imu_aggregator(imu_3) # [n_token, d_model]

        if 4 in invalid_imu_id_list:
            imu_4_embeddings = get_tokens_embed(mask_token_id) # [1, d_model]
            imu_4_embeddings = torch.repeat_interleave(imu_4_embeddings[None], n_token, dim=0) # [n_token, d_model]
        else:
            imu_4 = imu_data[:, 4].to(mask_dtype).to(device)  # [len, 15]
            imu_4 = imu_4.reshape(-1, normalization_window_size * 15) # [n_token, window_size * 15]
            imu_4_embeddings = model.imu_aggregator(imu_4) # [n_token, d_model]

        if 5 in invalid_imu_id_list:
            imu_5_embeddings = get_tokens_embed(mask_token_id) # [1, d_model]
            imu_5_embeddings = torch.repeat_interleave(imu_5_embeddings[None], n_token, dim=0) # [n_token, d_model]
        else:
            imu_5 = imu_data[:, 5].to(mask_dtype).to(device)  # [len, 15]
            imu_5 = imu_5.reshape(-1, normalization_window_size * 15) # [n_token, window_size * 15]
            imu_5_embeddings = model.imu_aggregator(imu_5) # [n_token, d_model]

        ref_R = convert_rotation(imu_batch["orient"], src_rep='6d', tgt_rep='mat')

        kwargs = {'normalization_window_size': normalization_window_size,
                  'return_seperate_indices': True,
                  'data_type': 'traj',
                  'debug': False,
                  'ref_R': ref_R,
                  }

        with torch.no_grad():
            _, traj_static, traj_mean_idx, traj_std_idx, traj_mean_value, traj_std_value = time_series_quantizer(imu_batch["transl"], **kwargs)
            traj_static_emb = get_token_embed_static(traj_static) # [1, n_token, n_var, d_model]
            traj_dynamic_mean_emb = get_token_embed_mean(traj_mean_idx)  # [1, n_token, n_var]
            traj_dynamic_std_emb = get_token_embed_std(traj_std_idx)  # [1, n_token, n_var]
            traj_dynamic_emb = torch.stack([traj_dynamic_mean_emb, traj_dynamic_std_emb], dim=-1) # [1, n_token, n_var, 2]
        
        kwargs = {'normalization_window_size': normalization_window_size,
                'return_seperate_indices': True,
                'data_type': 'orient',
                'debug': False,
                'ref_R': ref_R,
                }

        with torch.no_grad():
            _, orient_static, orient_mean_idx, orient_std_idx, orient_mean_value, orient_std_value = time_series_quantizer(imu_batch["orient"], **kwargs)
            orient_static_emb = get_token_embed_static(orient_static)
            orient_dynamic_mean_emb = get_token_embed_mean(orient_mean_idx)
            orient_dynamic_std_emb = get_token_embed_std(orient_std_idx)
            orient_dynamic_emb = torch.stack([orient_dynamic_mean_emb, orient_dynamic_std_emb], dim=-1)


        kwargs = {'normalization_window_size': normalization_window_size,
                'return_seperate_indices': True,
                'data_type': 'pose',
                'debug': False}

        # Also add pose information
        with torch.no_grad():
            _, pose_static, pose_mean_idx, pose_std_idx, pose_mean_value, pose_std_value = time_series_quantizer(imu_batch["pose"], **kwargs)
            pose_static_emb = get_token_embed_static(pose_static)
            pose_dynamic_mean_emb = get_token_embed_mean(pose_mean_idx)
            pose_dynamic_std_emb = get_token_embed_std(pose_std_idx)
            pose_dynamic_emb = torch.stack([pose_dynamic_mean_emb, pose_dynamic_std_emb], dim=-1)
        
        # import pdb; pdb.set_trace()
        status_static = torch.cat([traj_static, orient_static, pose_static], dim=2)
        status_mean = torch.cat([traj_mean_idx, orient_mean_idx, pose_mean_idx], dim=2)
        status_std = torch.cat([traj_std_idx, orient_std_idx, pose_std_idx], dim=2)
        status_static_emb = torch.cat([traj_static_emb, orient_static_emb, pose_static_emb], dim=2) # [1, n_token, 2*n_var, d_model]
        status_dynamic_emb = torch.cat([traj_dynamic_emb, orient_dynamic_emb, pose_dynamic_emb], dim=2) # [1, n_token, 2*n_var, 2]

        status_mean_value = torch.cat([traj_mean_value[:, :, 0], orient_mean_value[:, :, 0], pose_mean_value[:, :, 0]], dim=2) # [1, n_token, 2*n_var]
        status_std_value = torch.cat([traj_std_value[:, :, 0], orient_std_value[:, :, 0], pose_std_value[:, :, 0]], dim=2) # [1, n_token, 2*n_var]

        # Build gt_motion_recon [n_time_token, chunk_size, 3+22*rot_dof] for L2 reconstruction loss (order: traj, orient, pose)
        compression_rate = time_series_quantizer.compression_rate
        traj_t = imu_batch["transl"].float().to(device)
        orient_t = imu_batch["orient"].float().to(device)
        pose_t = imu_batch["pose"].float().to(device)
        gt_motion = torch.cat([traj_t, orient_t, pose_t], dim=1)  # [ntime, 3+22*6]
        ntime_trim = n_token * compression_rate
        gt_motion = gt_motion[:ntime_trim] # [n_time, 3+22*6]

        if status_mean_value.shape[1] != n_token:
            import pdb; pdb.set_trace()

        status_static_emb = status_static_emb.to(mask_dtype)
        status_dynamic_emb = status_dynamic_emb.to(mask_dtype)

        if not inference_mode:
            # use teacher forcing to generate the status embeddings, so input embedding is ground truth
            status_embeddings = model.status_aggregator(status_static_emb[0], status_dynamic_emb[0]) # [n_token, d_model]
        else:
            status_embeddings = torch.zeros([status_static_emb.shape[1], model.llm_hidden_size], dtype=mask_dtype).to(device)
        
        if bidirectional_motion:
            # generate from learnable motion embeddings
            query_embeddings = model.status_learnable_embeddings.repeat(status_static_emb.shape[1], 1) # [n_token, d_model]

        # Process text description
        if len(imu_batch['description']) == 0:
            description = None
            text_tokens = []
            text_embeddings = torch.zeros([0, get_tokens_embed.weight.shape[1]], dtype=mask_dtype).to(device)
        else:
            # randomly select one
            description = imu_batch['description'][text_select_idxs[batch_idx]]
            text_tokens = uni_prompting.text_tokenizer(description)['input_ids']
            text_embeddings = get_tokens_embed(torch.tensor(text_tokens, dtype=torch.long).to(device))  # [length, d_model]
        
        # Process object poses
        if len(imu_batch['objects']) == 0:
            object_id_tokens = []
            # object_status_tokens = []
            object_mean_values = []
            object_dynamic_mean_values = []
            object_id_embeddings = torch.zeros([0, get_tokens_embed.weight.shape[1]], dtype=mask_dtype).to(device)
            object_status_embeddings = torch.zeros([0, get_tokens_embed.weight.shape[1]], dtype=mask_dtype).to(device)
            cur_object_num = 0
        else:
            object_id_tokens = []
            # object_status_tokens = []
            object_mean_values = []
            object_dynamic_mean_values = []
            object_id_embeddings = []
            object_status_embeddings = []
            cur_object_num = 0
            for _, obj in enumerate(imu_batch['objects'].keys()):
                obj_id = obj_name_to_id.get(obj, None)
                assert obj_id is not None, f"Object name {obj} not found in obj_name_to_id mapping."

                obj_id_token = obj_id + object_token_bias
                obj_id_embedding = get_tokens_embed(torch.tensor([obj_id_token], dtype=torch.long).to(device))  # [d_model]

                if dynamic_object:
                    obj_rot = imu_batch['objects'][obj]['rot'] # [t, 6]
                    obj_transl = imu_batch['objects'][obj]['transl'] # [t,3]
                    obj_bbox = imu_batch['objects'][obj]['bbox'] # [t,3]
                    obj_dynamic_status = torch.from_numpy(np.concatenate([obj_rot, obj_transl, obj_bbox], axis=1)).float().to(device) # [t, 12]
                    obj_status = obj_dynamic_status[0:1] # first frame, [1, 12]
                    obj_status_embedding = model.object_aggregator(obj_status, allow_projection=True) # [1, 12] -> [1, d_model], only use the first frame
                else:
                    obj_rot = imu_batch['objects'][obj]['rot'] # [6]
                    obj_transl = imu_batch['objects'][obj]['transl'] # [3]
                    obj_bbox = imu_batch['objects'][obj]['bbox'] # [3] # FIXME: constant in the dataset
                    obj_bbox = np.ones_like(obj_transl)
                    try:
                        obj_status = torch.from_numpy(np.concatenate([obj_rot, obj_transl, obj_bbox], axis=0)).float().to(device)[None] # [1, 12] 
                        # obj_mean_indices, obj_quantized_mean = time_series_quantizer.mean_quantizer.quantize(obj_status)  # [9], [9], numpy array
                        # obj_mean_indices_embed = get_token_embed_mean(obj_mean_indices)
                        obj_status_embedding = model.object_aggregator(obj_status, allow_projection=True) # [1, 12] -> [1, d_model]
                    except:
                        import pdb; pdb.set_trace()

                object_id_embeddings.append(obj_id_embedding)
                object_status_embeddings.append(obj_status_embedding)
                object_id_tokens.append(obj_id_token)
                # object_status_tokens.append(obj_mean_indices.tolist())
                object_mean_values.append(obj_status)
                if dynamic_object:
                    # obj_dynamic_diff_status = obj_dynamic_status.clone()
                    # obj_dynamic_diff_status[1:] = obj_dynamic_diff_status[1:] - obj_dynamic_diff_status[:-1] # diff
                    object_dynamic_mean_values.append(obj_dynamic_status)

                cur_object_num += 1

            object_id_tokens = list(object_id_tokens)
            # object_status_tokens = torch.tensor(object_status_tokens)
            object_id_embeddings = torch.cat(object_id_embeddings, dim=0)  # [n_obj, d_model]
            object_status_embeddings = torch.cat(object_status_embeddings, dim=0)  # [n_obj, d_model]
            # import pdb; pdb.set_trace()
            object_mean_values = torch.cat(object_mean_values , dim=0)  # [n_obj, 12]
            if dynamic_object:
                object_dynamic_mean_values = torch.stack(object_dynamic_mean_values, dim=0)  # [n_obj, t, 12]

        n_time_step = len(imu_0_embeddings)

        _imu_embeddings = torch.cat([
            get_tokens_embed(uni_prompting.sptids_dict['<|imu|>'].to(device)), # 0

            get_tokens_embed(uni_prompting.sptids_dict['<|soimu_0|>'].to(device)), # 1
            apply_time_embed(imu_0_embeddings, fps=fps),
            get_tokens_embed(uni_prompting.sptids_dict['<|eoimu_0|>'].to(device)), # 2 + n

            get_tokens_embed(uni_prompting.sptids_dict['<|soimu_1|>'].to(device)), # 3 + n
            apply_time_embed(imu_1_embeddings, fps=fps),
            get_tokens_embed(uni_prompting.sptids_dict['<|eoimu_1|>'].to(device)), 

            get_tokens_embed(uni_prompting.sptids_dict['<|soimu_2|>'].to(device)),
            apply_time_embed(imu_2_embeddings, fps=fps),
            get_tokens_embed(uni_prompting.sptids_dict['<|eoimu_2|>'].to(device)), 

            get_tokens_embed(uni_prompting.sptids_dict['<|soimu_3|>'].to(device)),
            apply_time_embed(imu_3_embeddings, fps=fps),
            get_tokens_embed(uni_prompting.sptids_dict['<|eoimu_3|>'].to(device)), 

            get_tokens_embed(uni_prompting.sptids_dict['<|soimu_4|>'].to(device)),
            apply_time_embed(imu_4_embeddings, fps=fps),
            get_tokens_embed(uni_prompting.sptids_dict['<|eoimu_4|>'].to(device)),  

            get_tokens_embed(uni_prompting.sptids_dict['<|soimu_5|>'].to(device)),
            apply_time_embed(imu_5_embeddings, fps=fps),
            get_tokens_embed(uni_prompting.sptids_dict['<|eoimu_5|>'].to(device)),  

            get_tokens_embed(uni_prompting.sptids_dict['<|sostatus|>'].to(device)),
        ], dim=0).to(mask_dtype)  # [length, d_model]

        imu_embeddings_batch.append(_imu_embeddings[None]) # [1, length, d_model]


        if bidirectional_motion:
            query_embeddings_with_time = apply_time_embed(query_embeddings, fps=fps) # [n_token, d_model]
            query_embeddings_batch.append(query_embeddings_with_time[None]) # [1, n_token, d_model]
        else:
            status_embeddings_with_time = apply_time_embed(status_embeddings, fps=fps) # [n_token, d_model]

        _output_embeddings = torch.cat([
            status_embeddings_with_time if not bidirectional_motion else query_embeddings_with_time, # not use this one if bidirectional_motion
            get_tokens_embed(uni_prompting.sptids_dict['<|eostatus|>'].to(device)),
            get_tokens_embed(uni_prompting.sptids_dict['<|sot|>'].to(device)), 
            text_embeddings, 
            get_tokens_embed(uni_prompting.sptids_dict['<|eot|>'].to(device)),
            get_tokens_embed(uni_prompting.sptids_dict['<|soobj|>'].to(device)),
            object_id_embeddings,
            get_tokens_embed(uni_prompting.sptids_dict['<|eoobj|>'].to(device)),
            object_status_embeddings,
        ], dim=0).to(mask_dtype) # [length, d_model]

        output_embeddings_batch.append(_output_embeddings[None]) # [1, length, d_model]

        _all_embeddings = torch.cat([_imu_embeddings[None], _output_embeddings[None]], dim=1) # [1, length, d_model]
        all_embeddings_batch.append(_all_embeddings) # [1, length, d_model]

        ignore_token = torch.tensor([-100], dtype=torch.long, device=device)

        # import pdb; pdb.set_trace()

        _label_text_tokens = torch.cat([
            uni_prompting.sptids_dict['<|eostatus|>'].to(device),
            uni_prompting.sptids_dict['<|sot|>'].to(device),
            torch.tensor(text_tokens, dtype=torch.long, device=device),  # [length]
            (uni_prompting.sptids_dict['<|eot|>'] if len(text_tokens) > 0 else ignore_token).to(device), # if no text, set to -100, don't predict <|eot|>
            uni_prompting.sptids_dict['<|soobj|>'].to(device), # TODO: do we need this?
            torch.tensor(object_id_tokens, dtype=torch.long, device=device),
            (uni_prompting.sptids_dict['<|eoobj|>'] if cur_object_num > 1 else ignore_token).to(device), # if no object, set to -100, don't predict <|eoobj|>
        ], dim=0)

        label_mean_batch.append(status_mean) 
        label_std_batch.append(status_std)
        label_static_batch.append(status_static) 
        label_text_batch.append(_label_text_tokens)
        # label_object_batch.append(_label_object_tokens)

        label_mean_value_batch.append(status_mean_value)
        label_std_value_batch.append(status_std_value)
        label_object_value_batch.append(object_mean_values)
        label_gt_motion_batch.append(gt_motion)
        if dynamic_object:
            label_object_dynamic_value_batch.append(object_dynamic_mean_values)

            motion_value = torch.cat([imu_batch["orient"], imu_batch["transl"], imu_batch["pose"]], dim=1) # [len, d]
            label_motion_value_batch.append(motion_value)

        input_seq_len = 1 + (1 + n_time_step + 1) * 6 + 1 # input part of _imu_embeddings, until <|sostatus|> (included)
        n_status_token = status_embeddings.shape[0]
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
            
            # Create attention mask
            valid_position = torch.ones(cur_len, dtype=torch.bool)
            invalid_position = torch.zeros(pad_length, dtype=torch.bool)
            cur_mask = build_attention_mask(torch.cat([valid_position, invalid_position]))
            for k in range(cur_len, max_length):
                cur_mask[k, k] = 1 # self attention
        elif cur_len > max_length:
            raise ValueError
        cur_len = max_length
        cur_mask = build_attention_mask(torch.ones(cur_len, dtype=torch.bool))
        
        # Convert mask to the expected format
        inverted_mask = 1.0 - cur_mask.type(cur_embedding.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(mask_dtype).min
        )
        seq_all_embeddings.append(cur_embedding)
        seq_attention_mask.append(inverted_mask[None, None].to(mask_dtype))

    seq_all_embeddings = torch.cat(seq_all_embeddings, dim=0).to(device, non_blocking=True) # [batch_size, length, d_model]
    seq_attention_mask = torch.cat(seq_attention_mask, dim=0).to(device, non_blocking=True)  # [batch_size, 1, length, length]

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
              label_object_dynamic_value_batch, label_motion_value_batch, label_gt_motion_batch]
    
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

def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    
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
        assert train_selected_dataset in ["HUMOTO", "LINGO", "ParaHome", 'humanml', 'imuposer', 'dipimu']
    if eval_selected_dataset == 'None':
        eval_selected_dataset = None
    if eval_selected_dataset is not None:
        assert eval_selected_dataset in ["HUMOTO", "LINGO", "ParaHome", 'humanml', 'imuposer', 'dipimu']
    assert mode in ["train", "test"], "Mode must be train or test"
    if mode == "test":
        assert eval_selected_dataset is not None or eval_selected_imu_seq is not None, (
            "test mode requires experiment.eval_selected_dataset and/or experiment.eval_selected_imu_seq"
        )
    assert eval_selected_imu_seq is None or mode == "test", "eval_selected_imu_seq is only valid for mode=test"

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    config.experiment.viz_dir = str(Path(config.experiment.output_dir) / "viz_result_val")
    config.experiment.viz_test_dir = str(Path(config.experiment.output_dir) / "viz_test")
    config.experiment.viz_train_dir = str(Path(config.experiment.output_dir) / "viz_result_train")

    dynamic_object = config.model.dynamic_object
    if dynamic_object:
        assert train_selected_dataset == 'HUMOTO', "Dynamic object is only supported for HUMOTO"

    import wandb
    if mode == "test": # disable wandb for testing on full dataset
        os.environ["WANDB_MODE"] = "disabled"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=False, # the effective batch size is batch_size_imu * #GPU * #accumulation
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

    if accelerator.is_main_process:
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
    tokenizer_path = config.model.showo.llm_model_path
    if tokenizer_path is None:
        raise ValueError("config.model.showo.llm_model_path must be set")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")

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
    
    obj_name_list = np.loadtxt(obj_name_path, dtype=str).tolist()
    obj_name_list.sort() # TODO: find a easier way to expand the obj_name_list
    obj_name_to_id = {name: idx for idx, name in enumerate(obj_name_list)}
    n_object_class_token = len(obj_name_list)
    object_token_bias = token_bias + n_imu_special_token
    object_token_id_to_name = {v + object_token_bias: k for k, v in obj_name_to_id.items()}

    # add imu special tokens and times series tokens
    config.model.showo.vocab_size = token_bias + n_imu_special_token + n_object_class_token

    bidirectional_imu = config.model.showo.bidirectional_imu
    bidirectional_motion = config.model.showo.get('bidirectional_motion', False)
    print(f"Bidirectional IMU: {bidirectional_imu}")
    print(f"Bidirectional motion: {bidirectional_motion}")


    # load time series tokenizer
    use_rope_embed = config.model.get('use_rope_embed', False)
    time_series_quantizer = DiscreteQuantizer(
            compression_rate=config.model.normalization_window_size,
            d_model=2048 if base_model == 'showo' else 1024,
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
        if base_model == 'gpt2-medium':
            from transformers import GPT2Config
            llm_path = config.model.showo.llm_model_path
            showo_cfg = OmegaConf.to_container(config.model.showo, resolve=True)
            exclude_keys = {'bidirectional_motion', 'add_gate', 'all_continuous_recon', 'partial_continuous_recon'}
            showo_cfg_filtered = {k: v for k, v in showo_cfg.items() if k not in exclude_keys}
            model_config = {
                **GPT2Config.from_pretrained(llm_path).to_dict(),
                **showo_cfg_filtered,
                'base_model': 'gpt2-medium',
                'llm_model_path': llm_path,
            }
        else:
            model_config = PretrainedConfig.from_pretrained(
                os.path.join(pretrained_showo_path, 'config.json')
            ).to_dict()

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
                motion_only=config.model.motion_only,
                text_only=config.model.text_only,
                bidirectional_motion=config.model.showo.get('bidirectional_motion', False),
                add_gate=config.model.showo.get('add_gate', False),
                all_continuous_recon=config.model.showo.get('all_continuous_recon', False),
                partial_continuous_recon=config.model.showo.get('partial_continuous_recon', False),
                **model_config,
            )
        print(f"Show-o model size: {get_model_size_gb(model):.2f} GB")
        print("Initialized Show-o model.")

        if base_model == 'showo' and not config.experiment.resume_from_checkpoint:
            assert config.model.showo.load_from_showo, "load_from_showo must be True when resume_from_checkpoint is False"

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
        elif base_model == 'gpt2-medium':
            # GPT-2 backbone already loaded via from_pretrained in ShowoIMU.__init__
            logger.info("Using pretrained GPT-2 medium (loaded in model __init__).")

        old_embeddings_len = len(model.embed_tokens.weight.data) # 58498
        if base_model == 'showo':
            assert old_embeddings_len == 58498, f"Old embeddings length {old_embeddings_len} does not match the expected value 58498"
        elif base_model == 'gpt2-medium':
            print(f"Old embeddings length {old_embeddings_len}")

        if config.model.showo.vocab_size != model.vocab_size:
            print(f"Resizing model token embeddings from {model.vocab_size} to {config.model.showo.vocab_size}")
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size

    print(f"Done loading the showo model in {time.time() - start_time:.2f} seconds.")

    mask_id = model.mask_token_id
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

        logger.info(f"Set finetune mode, excluding parameters from optimization:")
        for n in no_opt:
            logger.info(f"- {n}")
        logger.info(f"###########################")

    if base_model == 'showo':
        # no_decay = ["bias", "layer_norm.weight","embeddings.weight"]
        no_decay = ["bias", "layernorm.weight","embed_tokens.weight", "status_learnable_embeddings"]
    elif base_model == 'gpt2-medium':
        no_decay = ["bias", "wte", "wpe", "ln_1", "ln_2", "ln_f", "embed_tokens.weight"]

    train_only = []
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
        if any(no in name for no in no_opt):
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

    assert config.training.batch_size_imu > 0, "Batch size must be greater than 0"

    
    def_dataset = partial(IMUDataset,
        root=imu_data_path,
        overfit=overfit,
        motion_only=config.model.motion_only,
        text_only=config.model.text_only,
        fps=config.dataset.params.fps,
        dynamic_object=dynamic_object,
        acc_scale=config.dataset.params.get('acc_scale', 1.0),
        gyro_scale=config.dataset.params.get('gyro_scale', 1.0),
    )

    # For training, disable per-sample random_cut inside the dataset and
    # instead perform a shared random cut at the batch level in
    # imu_to_input (controlled by random_cut above).
    train_dataset_imu = def_dataset(
        split="train",
        selected_dataset=train_selected_dataset,
        selected_imu_seq=eval_selected_imu_seq,
        random_cut=train_random_cut,
        random_mask_text=False,
        add_imu_noise=config.training.add_imu_noise,
        IMUSEQMAXLEN=config.experiment.max_train_imu_len,
    )
    train_sampler_imu = DistributedSampler(
        train_dataset_imu, 
        num_replicas=accelerator.num_processes, 
        rank=accelerator.process_index, 
        shuffle=True
    )
    train_dataloader_imu = DataLoader(
        train_dataset_imu,
        batch_size=config.training.batch_size_imu,
        sampler=train_sampler_imu,
        collate_fn=train_dataset_imu.collate_fn,
        num_workers=dataset_config.num_workers
    )
    assert len(train_dataloader_imu) > 0, "No training data found"
    
    # Data for Visualization, so no need shuffle
    train_seq_sampler_imu = SequentialSampler(train_dataset_imu)
    train_seq_dataloader_imu = DataLoader(
        train_dataset_imu,
        batch_size=1,
        sampler=train_seq_sampler_imu,
        collate_fn=train_dataset_imu.collate_fn,
        num_workers=dataset_config.num_workers
    )
    if overfit:
        val_dataset_imu = train_dataset_imu
    else:
        val_dataset_imu = def_dataset(
            split="val",
            selected_dataset=eval_selected_dataset,
            selected_imu_seq=eval_selected_imu_seq,
            random_mask_text=False,
            shuffle_list=True,
            add_imu_noise=False,
            IMUSEQMAXLEN=config.experiment.max_eval_imu_len,
        )
        assert len(val_dataset_imu) > 0, "No validation data found" 
    val_sampler_imu = SequentialSampler(val_dataset_imu)
    val_seq_dataloader_imu = DataLoader(
        val_dataset_imu,
        batch_size=1,
        shuffle=False, 
        sampler=val_sampler_imu,
        collate_fn=val_dataset_imu.collate_fn,
        num_workers=dataset_config.num_workers
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataset_imu) / config.training.batch_size_imu)
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)  # Ensure at least one step per epoch
    # num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    # num_train_epochs = max(num_train_epochs, 50000)

    num_train_epochs = config.training.get('num_train_epochs', 100)

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
            compatible_state_dict = {}
            missing_keys = []
            unexpected_keys = []
            shape_mismatches = []

            if not strict_resume:

                for key in model_state_dict.keys():
                    if key in checkpoint_state_dict:
                        model_param = model_state_dict[key]
                        ckpt_param = checkpoint_state_dict[key].to(model_param.device)

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
                logger.info(f"Shape mismatches handled: {len(shape_mismatches)}")
                logger.info(f"Loaded single GPU checkpoint from {checkpoint_path}.")

            else:
                model.load_state_dict(checkpoint_state_dict, strict=True)
                logger.info(f"Strictly loaded checkpoint from {checkpoint_path}.")

            logger.info("Preparing model, optimizer and dataloaders")
            model, optimizer, train_dataloader_imu, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader_imu, lr_scheduler, 
            )
            successful_resume = True

        # load from distributed checkpoint
        elif mode == "train" and not load_without_optimizer:
            checkpoint_path = os.path.join(config.experiment.ckpt_dir, path)

            logger.info("Preparing model, optimizer and dataloaders")
            model, optimizer, train_dataloader_imu, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader_imu, lr_scheduler, 
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
        model, optimizer, train_dataloader_imu, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader_imu, lr_scheduler, 
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

    eval_model_func = partial(eval_model,
        time_series_quantizer=time_series_quantizer,
        uni_prompting=uni_prompting,
        object_token_id_to_name=object_token_id_to_name,
        obj_name_to_id=obj_name_to_id,
        object_token_bias=object_token_bias,
        accelerator=accelerator,
        config=config,
        dynamic_object=dynamic_object,
        bidirectional_imu=bidirectional_imu,
        bidirectional_motion=bidirectional_motion,
        fps=config.dataset.params.fps,
        motion_only=config.model.motion_only,
        generate_number=True,
        max_new_text_tokens=config.experiment.max_new_text_tokens,
        max_object_id_tokens=config.experiment.max_object_id_tokens,
        min_object_id_tokens=config.experiment.min_object_id_tokens,
    )
    ##################################
    #         Evaluation Only        #
    ################################## 
    # After this block, we will exit the script
    if mode == "test":

        # delete optimizer and lr_scheduler
        optimizer = None
        lr_scheduler = None
        torch.cuda.empty_cache()

        # assert 1 GPU
        assert accelerator.num_processes == 1, "Testing on full dataset requires 1 GPU"
        shift_values = [0] if eval_selected_imu_seq is not None else [0, 2]
        
        start_time = time.time()
        for shift_value in shift_values:

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
            test_sampler_imu = SequentialSampler(test_dataset_imu)
            test_seq_dataloader_imu = DataLoader(
                test_dataset_imu, 
                batch_size=1,
                sampler=test_sampler_imu, 
                collate_fn=test_dataset_imu.collate_fn,
                shuffle=False,
                num_workers=dataset_config.num_workers
            )
            if eval_selected_imu_seq is not None:
                print(
                    f"Evaluating single IMU sequence {eval_selected_imu_seq} "
                    f"({len(test_dataset_imu)} sample(s)), shift={shift_value} frames"
                )
            else:
                print(f"Evaluating on {eval_selected_dataset} dataset with {len(test_dataset_imu)} samples and shifted {shift_value} frames")

            eval_model_func(
                model=accelerator.unwrap_model(model),
                eval_num=len(test_seq_dataloader_imu),
                dataloader=test_seq_dataloader_imu,
                global_step=0,
                split="test",
                postfix=f"shifted_{shift_value}",
                job_id=job_id,
                total_jobs=total_jobs,
                invalid_imu_id=eval_invalid_imu_id,
                save_sample=config.experiment.save_test_sample,
            ) 
        print(f"Evaluation done in {time.time() - start_time} seconds.")
        # copy the loaded checkpoint to the same dir as saved samples (output_dir)
        if config.experiment.save_ckpt_when_eval:
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
    logger.info(f"  Number of samples in train set = {len(train_dataset_imu)}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    is_start_eval = config.experiment.is_start_eval
    is_start_sample = config.experiment.is_start_sample

    print(f'total epoch = {num_train_epochs}')
    set_cut_length(train_dataloader_imu, config.experiment.min_train_imu_len, config.experiment.max_train_imu_len)
    for epoch in range(first_epoch, first_epoch + num_train_epochs): #FIXME: set a more reasonable way to set the number of epochs

        model.train()
        train_sampler_imu.set_epoch(epoch + 1)  # Shuffle the data at the beginning of each epoch
        # iterate over train_dataloader_imu        
        for imu_batch_idx, imu_batches in enumerate(train_dataloader_imu):

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
                        normalization_window_size=config.model.normalization_window_size,
                        smooth_imu=config.model.smooth_imu,
                        invalid_imu_id=train_invalid_imu_id,
                        random_text=not overfit,
                        dynamic_object=dynamic_object,
                        bidirectional_imu=bidirectional_imu,
                        bidirectional_motion=bidirectional_motion,
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

                # Gather the losses across all processes for logging (if we use distributed training).
                total_loss_global = accelerator.gather(total_loss).mean()
                loss_mean_global = accelerator.gather(loss_dict["loss_mean"]).mean()
                loss_std_global = accelerator.gather(loss_dict["loss_std"]).mean()
                loss_static_global = accelerator.gather(loss_dict["loss_static"]).mean()
                loss_recon_global = accelerator.gather(loss_dict["loss_recon"]).mean()
                loss_text_global = accelerator.gather(loss_dict["loss_text"]).mean()
                loss_object_pose_global = accelerator.gather(loss_dict["loss_object_pose"]).mean()
                loss_obj_dynamic_pose_global = accelerator.gather(loss_dict["loss_obj_dynamic_pose"]).mean()
                mean_accuracy_global = accelerator.gather(loss_dict["mean_accuracy"]).mean()
                std_accuracy_global = accelerator.gather(loss_dict["std_accuracy"]).mean()
                static_accuracy_global = accelerator.gather(loss_dict["static_accuracy"]).mean()
                obj_id_accuracy_global = accelerator.gather(loss_dict["obj_id_accuracy"]).mean()
                text_accuracy_global = accelerator.gather(loss_dict["text_accuracy"]).mean()
                # obj_pose_accuracy_global = accelerator.gather(loss_dict["obj_pose_accuracy"]).mean()
                obj_pose_accuracy_global = torch.tensor(0.0, device=accelerator.device)
                # obj_dynamic_pose_accuracy_global = accelerator.gather(loss_dict["obj_dynamic_pose_accuracy"]).mean()
                obj_dynamic_pose_accuracy_global = torch.tensor(0.0, device=accelerator.device)

                if accelerator.is_main_process:
                    print('epoch = {}, step = {}, total_loss = {:.2e}, loss_mean = {:.2e}, loss_std = {:.2e}, loss_static_status = {:.2e}, loss_recon = {:.2e}, loss_text = {:.2e}, loss_object_pose = {:.2e}, loss_obj_dynamic_pose = {:.2e}, mean_acc = {:.4f}, std_acc = {:.4f}, static_acc = {:.4f}, text_acc = {:.2f}, obj_id_acc = {:.4f}, obj_pose_acc = {:.4f}, obj_dynamic_pose_acc = {:.4f}, len = {}'.format(
                        epoch,
                        global_step, 
                        total_loss_global.item(),
                        loss_mean_global.item(),
                        loss_std_global.item(),
                        loss_static_global.item(),
                        loss_recon_global.item(),
                        loss_text_global.item(),
                        loss_object_pose_global.item(),
                        loss_obj_dynamic_pose_global.item(),
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
                embedding_weight_grad = model.embed_tokens.weight.grad
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
                end = time.time()
                if (global_step == 0 or (global_step + 1) % config.experiment.sample_every == 0) or is_start_sample:
                    accelerator.wait_for_everyone()
                    is_start_sample = False
                    model.eval()
                    if accelerator.is_main_process:
                        if not overfit:
                            eval_model_func(
                                model=accelerator.unwrap_model(model),
                                dataloader=val_seq_dataloader_imu,
                                global_step=global_step + 1,
                                split="val",
                                eval_num=3,
                                save_sample=True,
                                invalid_imu_id=eval_invalid_imu_id,
                                generate_number=False,
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
                        if config.experiment.train_random_cut:
                            train_sampler_imu.dataset.enable_random_cut()
                    model.train()
                    accelerator.wait_for_everyone()

                # generate numbers for train and val set
                if (global_step == 0 or (global_step + 1) % config.experiment.eval_every == 0) or is_start_eval:
                    accelerator.wait_for_everyone()
                    is_start_eval = False
                    model.eval()
                    if accelerator.is_main_process: 
                        eval_model_func(
                            model=accelerator.unwrap_model(model),
                            dataloader=val_seq_dataloader_imu,
                            global_step=global_step + 1,
                            split="val",
                            eval_num=config.experiment.max_eval_sample_num,
                            save_sample=False,
                            invalid_imu_id=eval_invalid_imu_id,
                        ) 
                    model.train()
                    accelerator.wait_for_everyone()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "epoch": epoch,
                        "step_total_loss": total_loss_global.item(),
                        "step_loss_mean": loss_mean_global.item(),
                        "step_loss_std": loss_std_global.item(),
                        "step_loss_static": loss_static_global.item(),
                        "step_loss_recon": loss_recon_global.item(),
                        "step_loss_text": loss_text_global.item(),
                        "step_loss_object_pose": loss_object_pose_global.item(),
                        "step_loss_obj_dynamic_pose": loss_obj_dynamic_pose_global.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
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
                        f"Loss_object_pose: {loss_object_pose_global.item():0.4f} "
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
                    save_checkpoint(model, optimizer, lr_scheduler, config, accelerator, global_step + 1, epoch=epoch)
                    print(f"Saving checkpoint at step {global_step + 1}")
                    accelerator.wait_for_everyone()
                    
                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break

            set_cut_length(train_dataloader_imu, config.experiment.min_train_imu_len, config.experiment.max_train_imu_len)

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, optimizer, lr_scheduler, config, accelerator, global_step, epoch=epoch)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
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

    if len_pred < len_gt:
        # Pad with False (incorrect) for missing predictions
        pad_len = len_gt - len_pred
        pad_false = torch.zeros(bs, pad_len, dtype=torch.bool, device=logits.device)
        correct = torch.cat([correct_cut, pad_false], dim=1)
    else:
        correct = correct_cut  # Already cut to target length

    # Compute mean accuracy
    return correct.float().mean()

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



@torch.inference_mode()
def eval_model(
    model,
    time_series_quantizer,
    dataloader,
    uni_prompting,
    object_token_id_to_name,
    obj_name_to_id,
    object_token_bias,
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
    total_jobs=1,
    job_id=0,
    bidirectional_imu: bool=False,
    bidirectional_motion: bool=False,
    fps: int=None,
    save_sample: bool=False,
    eval_num: int=None,
    eval_full_dataset: bool=False,
    max_sample_keep_num: int=5,

    max_new_text_tokens: int=60,
    max_object_id_tokens: int=1,
    min_object_id_tokens: int=1
):
    logger.info("Generating ...")

    assert fps is not None, "FPS must be specified for evaluation."
    assert invalid_imu_id is not None, "invalid_imu_id must be specified"

    logger.info('########################################################')
    logger.info("Start evaluation ...")
    logger.info(f"eval_num = {eval_num}")
    logger.info(f"invalid_imu_id = {invalid_imu_id}")
    logger.info(f"bidirectional_imu = {bidirectional_imu}")
    logger.info(f"bidirectional_motion = {bidirectional_motion}")
    logger.info(f"fps = {fps}")
    logger.info(f"max_new_text_tokens = {max_new_text_tokens}")
    logger.info(f"max_object_id_tokens = {max_object_id_tokens}")
    logger.info(f"min_object_id_tokens = {min_object_id_tokens}")
    logger.info('########################################################')

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

    if split == 'train':
        dataloader.dataset.disable_random_cut()

    for imu_batch_idx, imu_batches in enumerate(dataloader):
        if imu_batch_idx >= eval_num:
            break

        if imu_batch_idx % total_jobs != job_id:
            continue

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
                normalization_window_size=config.model.normalization_window_size,
                smooth_imu=config.model.smooth_imu,
                invalid_imu_id=invalid_imu_id,
                random_text=False, # TODO
                dynamic_object=dynamic_object,
                bidirectional_imu=bidirectional_imu,
                bidirectional_motion=bidirectional_motion,
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

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"): # disable autocast for debugging

            input_imu_len = input_imu_lens[0]
            n_time_token = (input_imu_len - 2)//6-2  # 2 for <|soimu|> and <|eoimu|>, 3 for each imu device, -2 for <|sostatus|> and <|eostatus|>

            # Invalid IMU positions for attention isolation (only attend to self; others don't attend to them)
            invalid_positions = get_invalid_imu_positions(input_imu_len, invalid_imu_id_list) if invalid_imu_id_list else None

            # Direct prediction without rollout
            top_k = 1

            if bidirectional_motion:
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
                    # Model returns [bs, n_frame, n_var], reshape to [n_status_token, compression_rate, n_var]
                    x_recon_precomputed = output['x_recon'][0].float().reshape(n_status_token, model.compression_rate, -1)
                    # Build cur_input_embeddings up to end of status for text/object generation
                    status_end = input_imu_len + n_status_token
                    cur_input_embeddings = all_embeddings[:, :status_end]
                    L = cur_input_embeddings.shape[1]
                    cur_attention_mask = attention_mask[:, :, :L, :L]
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
                    cur_attention_mask = attention_mask[:, :, :L, :L]
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

                    cur_attention_mask = attention_mask[:, :, :input_imu_len+n_time_token, :input_imu_len+n_time_token]
                    L = input_imu_len + n_time_token
                    bs = cur_input_embeddings.shape[0]
                    assert bs == 1

                    # bidirectional motion inference: one-shot predict all motion tokens from full context
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if config.model.showo.base_model == 'showo':
                            hidden_states = model.showo(inputs_embeds=cur_input_embeddings, attention_mask=cur_attention_mask)
                        else:
                            hidden_states = model.showo(inputs_embeds=cur_input_embeddings, attention_mask=cur_attention_mask,
                                                        output_hidden_states=True).hidden_states[-1]

                    status_hidden_states = hidden_states[:, input_imu_len:, :]  # [batch_size, n_time_token, hidden_dim]
                    assert status_hidden_states.shape[1] == n_time_token

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
                cur_attention_mask = attention_mask[:, :, :input_imu_len, :input_imu_len]
                L = input_imu_len

                result_static_idx = []
                result_mean_idx = []
                result_std_idx = []
                result_static_logits = []
                result_mean_logits = []
                result_std_logits = []

                # loop to reconstruct the trajectory and human pose
                for t in range(len(time_bar)):
                    
                    # Forward pass
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if config.model.showo.base_model == 'showo':
                            hidden_states = model.showo(inputs_embeds=cur_input_embeddings, attention_mask=cur_attention_mask)
                        else:
                            hidden_states = model.showo(inputs_embeds=cur_input_embeddings, output_hidden_states=True).hidden_states[-1]
                    
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
                # import pdb; pdb.set_trace()

            # loop to reconstruct the text
            text_result = []
            text_logits = []
            
            # min_new_before_eot = 2
            # neg_inf = -100
            # skip_first_eot_token = True
            # import pdb;pdb.set_trace()

            # append <|eostatus|> token
            eostatus_token = uni_prompting.sptids_dict['<|eostatus|>'].item()
            idx_next_embeddings = model.embed_tokens(torch.tensor([eostatus_token], device=cur_input_embeddings.device, dtype=torch.long))
            cur_input_embeddings = torch.cat([cur_input_embeddings, idx_next_embeddings[None]], dim=1)
            L = L + 1
            cur_attention_mask = extend_attn_mask(attention_mask, L, attention_mask.dtype, invalid_positions=invalid_positions)

            # append <|sot|> token
            sot_token = uni_prompting.sptids_dict['<|sot|>'].item()
            idx_next_embeddings = model.embed_tokens(torch.tensor([sot_token], device=cur_input_embeddings.device, dtype=torch.long))
            cur_input_embeddings = torch.cat([cur_input_embeddings, idx_next_embeddings[None]], dim=1)
            L = L + 1
            cur_attention_mask = extend_attn_mask(attention_mask, L, attention_mask.dtype, invalid_positions=invalid_positions)

            for it in range(max_new_text_tokens):

                if config.model.showo.base_model == 'showo':
                    hidden_states = model.showo(inputs_embeds=cur_input_embeddings, attention_mask=cur_attention_mask)
                else:
                    hidden_states = model.showo(inputs_embeds=cur_input_embeddings, output_hidden_states=True).hidden_states[-1]
                logits = model.showo.lm_head(hidden_states[:, -1])  # [bs, vocab_size]
                # if it < min_new_before_eot:
                #     # prevent generating <|eot|> token at the beginning
                #     logits[:, eot_token] = neg_inf
                idx_next = last_dim_logit_to_index(logits[:, None, :], temperature=temperature, top_k=None)  # [bs, 1] # FIXME
                # idx_next = last_dim_logit_to_index(logits[:, None, :], temperature=1.2, top_k=500)  # [bs, 1]
                idx_next_embeddings = model.embed_tokens(idx_next)

                # input_imu_len[0]-1 is <|sostatus|> token, so it will decode the first status token
                # n_status_token = labels[0][0].shape[0]
                # traj_hidden_states = hidden_states[0, input_imu_len-1:input_imu_len+n_status_token-1]

                cur_input_embeddings = torch.cat([cur_input_embeddings, idx_next_embeddings], dim=1) # 
                # cur_attention_mask: [1,1,L,L] -> [1,1,L+1,L+1]
                cur_attention_mask = extend_attn_mask(cur_attention_mask, cur_attention_mask.shape[2]+1, cur_attention_mask.dtype, invalid_positions=invalid_positions)

                text_logits.append(logits)
                text_result.append(idx_next[0][0])

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
            max_object_id_tokens = 0 if motion_only else max_object_id_tokens
            soobj_token = uni_prompting.sptids_dict['<|soobj|>'].item()
            eoobj_token = uni_prompting.sptids_dict['<|eoobj|>'].item()
            
            # append <|soobj|> token
            # soobj_embedding = model.embed_tokens(torch.tensor([soobj_token], device=cur_input_embeddings.device, dtype=torch.long)).to(weight_dtype).reshape(1, 1, -1)
            # cur_input_embeddings = torch.cat([cur_input_embeddings, soobj_embedding], dim=1)
            # temp = torch.ones([1, 1, cur_attention_mask.shape[2] + 1, cur_attention_mask.shape[3] + 1], device=cur_attention_mask.device) * torch.finfo(cur_attention_mask.dtype).min
            # temp[:, :, :-1, :-1] = cur_attention_mask
            # temp[:, :, -1] = 0
            # cur_attention_mask = temp

            skip_first_eoobj_token = True
            for it in range(max_object_id_tokens):
                if config.model.showo.base_model == 'showo':
                    hidden_states = model.showo(inputs_embeds=cur_input_embeddings, attention_mask=cur_attention_mask)
                else:
                    hidden_states = model.showo(inputs_embeds=cur_input_embeddings, output_hidden_states=True).hidden_states[-1]
                logits = model.showo.lm_head(hidden_states[:, -1])  # [bs, vocab_size]
                # Set non-object logits to -100, only allow object class tokens and eoobj (when appropriate)
                valid_object_ids = list(object_token_id_to_name.keys())
                if it >= min_object_id_tokens:
                    valid_token_ids = valid_object_ids + [eoobj_token]
                else:
                    valid_token_ids = valid_object_ids
                valid_logits = logits[:, valid_token_ids].clone()
                logits[:, :] = -100
                logits[:, valid_token_ids] = valid_logits
                idx_next = last_dim_logit_to_index(logits[:, None, :], temperature=temperature, top_k=1)  # [bs, 1] randomly sample FIXME
                idx_next_embeddings = model.embed_tokens(idx_next)

                cur_input_embeddings = torch.cat([cur_input_embeddings, idx_next_embeddings], dim=1)
                # cur_attention_mask: [1,1,L,L] -> [1,1,L+1,L+1]
                cur_attention_mask = extend_attn_mask(cur_attention_mask, cur_attention_mask.shape[2]+1, cur_attention_mask.dtype, invalid_positions=invalid_positions)

                object_id_logits.append(logits)
                object_id_result.append(idx_next[0][0])

                if eoobj_token is not None and idx_next.item() == eoobj_token:
                    if skip_first_eoobj_token and it == 0:
                        skip_first_eoobj_token = False
                    else:
                        break
            
            object_id_valid = []
            for temp in object_id_result:
                if temp.item() in object_token_id_to_name.keys():
                    object_id_valid.append(temp.item())
            n_valid_object = len(object_id_valid)

            object_status_result = []
            for t in range(n_valid_object):
                if config.model.showo.base_model == 'showo':
                    hidden_states = model.showo(inputs_embeds=cur_input_embeddings, attention_mask=cur_attention_mask)
                else:
                    hidden_states = model.showo(inputs_embeds=cur_input_embeddings, output_hidden_states=True).hidden_states[-1]

                _x_obj_mean_logits = model.object_mean_head(hidden_states[:, -1]) # [1, 12]
                object_status_result.append(hidden_states[:, -1])

                # _x_obj_mean_logits = _x_obj_mean_logits.reshape(-1, n_dynamic_bins)
                # _x_obj_mean_idx = torch.argmax(_x_obj_mean_logits, dim=-1)
                # _x_obj_mean_embed = mean_embedder(_x_obj_mean_idx).to(weight_dtype)  # [n_var]
                # idx_next_embeddings = model.object_aggregator(_x_obj_mean_embed[None], None).reshape(1,1,-1)
                idx_next_embeddings = model.object_aggregator(_x_obj_mean_logits[None]) # [1, 1, 2048]

                cur_input_embeddings = torch.cat([cur_input_embeddings, idx_next_embeddings], dim=1)
                # cur_attention_mask: [1,1,L,L] -> [1,1,L+1,L+1]
                cur_attention_mask = extend_attn_mask(cur_attention_mask, cur_attention_mask.shape[2]+1, cur_attention_mask.dtype, invalid_positions=invalid_positions)

            
            pred_objects = {}
            if len(object_status_result) > 0:

                object_status_result = torch.cat(object_status_result, dim=0)
                # x_object_mean_logits = model.object_mean_head(object_status_result).reshape(1, -1, n_dynamic_bins)
                # x_object_mean_idx = last_dim_logit_to_index(x_object_mean_logits, temperature=temperature, top_k=1) # [n_obj, 9]
                # x_object_mean = time_series_quantizer.mean_quantizer.decode(x_object_mean_idx.reshape(-1)).reshape(len(object_status_result), 9)
                x_object_mean = model.object_mean_head(object_status_result).reshape(len(object_status_result), 12) # [n_obj, 12]

                if not dynamic_object: 
                    for t in range(n_valid_object):
                        # {'object_name': {'rot': [6], 'transl': [3]}}
                        obj_name = object_token_id_to_name[object_id_valid[t]]
                        pred_objects[obj_name] = {
                            'rot': x_object_mean[t, 0:6],
                            'transl': x_object_mean[t, 6:9],
                            'bbox': x_object_mean[t, 9:12]
                        } # imu_batches[0]['objects'] 
 
            # decode object information (skip when all_continuous_recon, x_recon_precomputed is used)
            if x_recon_precomputed is None:
                x_static_idx = result_static_idx.reshape(n_time_token, -1)
                x_mean_idx = result_mean_idx
                x_std_idx = result_std_idx

                x_static_logits = result_static_logits.reshape(n_time_token, -1, n_static_bins)
                x_mean_logits = result_mean_logits.reshape(n_time_token, -1, n_dynamic_bins)
                x_std_logits = result_std_logits.reshape(n_time_token, -1, n_dynamic_bins)

                x_mean = time_series_quantizer.mean_quantizer.decode(x_mean_idx.reshape(-1)).reshape(n_time_token, 1, -1)
                x_std = time_series_quantizer.std_quantizer.decode(x_std_idx.reshape(-1)).reshape(n_time_token, 1, -1)

        # calculate metrics
        if x_recon_precomputed is not None and not partial_continuous_recon:
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

            status_ce_mean = F.cross_entropy(result_mean_logits.reshape(-1, n_dynamic_bins), labels_mean_motion.reshape(-1), reduction='mean', ignore_index=-100)
            status_ce_std = F.cross_entropy(result_std_logits.reshape(-1, n_dynamic_bins), labels_std_motion.reshape(-1), reduction='mean', ignore_index=-100)
            status_ce_static = F.cross_entropy(result_static_logits.reshape(-1, n_static_bins), labels_static_motion.reshape(-1), reduction='mean', ignore_index=-100)
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


        if dynamic_object:
            # object_status_result [n_obj, 12]
            n_obj_token = len(object_status_result)
            object_hidden_states_expand = object_status_result[:,None].expand(-1, n_time_token * 4, -1) # [n_obj_token, n_time, d_model]
            pred_motion = labels[9][0] # [n_time, d]
            pred_motion_expand = pred_motion[None].expand(n_obj_token, -1, -1) # [n_obj_token, n_time, d]
            pred_motion_expand = pred_motion_expand.to(object_hidden_states_expand.device)
            dynamic_diff_embeddings = torch.cat([object_hidden_states_expand, pred_motion_expand], dim=2) # [n_obj_token, n_time, 2*d_model]
            x_obj_dynamic_mean_pred = model.object_dynamic_mean_head(dynamic_diff_embeddings) # [n_obj_token, n_time, 12]
            x_obj_dynamic_mean_pred = x_obj_dynamic_mean_pred.reshape(n_obj_token, n_time_token * 4, 12) # [n_obj, n_time, 12]
            x_obj_dynamic_mean_pred = x_obj_dynamic_mean_pred + pred_motion_expand[:, 0:1] # add the first frame

            # x_obj_dynamic_concat = torch.cat([x_object_mean[:, None], x_obj_dynamic_mean_pred[:, 1:]], dim=1) # [n_obj, n_time, 12]
            # x_obj_dynamic_concat = torch.cumsum(x_obj_dynamic_concat, dim=1) # [n_obj, n_time, 12]
        
            for t in range(n_valid_object):
                # {'object_name': {'rot': [6], 'transl': [3]}}
                obj_name = object_token_id_to_name[object_id_valid[t]]
                pred_objects[obj_name] = {
                    'rot': x_obj_dynamic_mean_pred[t, :, 0:6],
                    'transl': x_obj_dynamic_mean_pred[t, :, 6:9],
                    'bbox': x_obj_dynamic_mean_pred[t, :, 9:12]
                } # imu_batches[0]['objects']

        gt_objects_names = [name for name in imu_batches[0]['objects'].keys()]  # list of object ids
        pred_objects_names = [name for name in pred_objects.keys()]
        object_id_correct = set(gt_objects_names).intersection(set(pred_objects_names))
        object_id_accuracy = len(object_id_correct) / len(gt_objects_names) if len(gt_objects_names) > 0 else None
        object_rot_L1 = None
        object_transl_L1 = None
        if len(object_id_correct) > 0:
            object_rot_L1 = 0.0
            object_transl_L1 = 0.0
            for obj_name in object_id_correct:
                gt_obj_rot = imu_batches[0]['objects'][obj_name]['rot'] # numpy array
                gt_obj_transl = imu_batches[0]['objects'][obj_name]['transl']
                pred_obj_rot = pred_objects[obj_name]['rot'].float().detach().cpu().numpy()
                pred_obj_transl = pred_objects[obj_name]['transl'].float().detach().cpu().numpy()
                pred_obj_bbox = pred_objects[obj_name]['bbox'].float().detach().cpu().numpy()
                object_rot_L1 += np.linalg.norm(gt_obj_rot - pred_obj_rot)
                object_transl_L1 += np.linalg.norm(gt_obj_transl - pred_obj_transl)
            object_rot_L1 /= len(object_id_correct)
            object_transl_L1 /= len(object_id_correct)
            if dynamic_object:
                object_rot_L1 /= (4 * n_time_token)
                object_transl_L1 /= (4 * n_time_token)

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

        gt_transl = imu_batches[0]['transl'].cpu().numpy()
        gt_orient = convert_rotation(imu_batches[0]['orient'], '6d', 'aa').cpu().numpy()
        gt_pose = convert_rotation(imu_batches[0]['pose'].reshape(-1, 6), '6d', 'aa').reshape(-1, 21*3).cpu().numpy()
        gt_pose_raw = imu_batches[0]['pose'].reshape(-1, 6).cpu().numpy()

        # post-process the text logits
        text_logits = torch.cat(text_logits, dim=0)  # [len, vocab_size]
        if len(imu_batches[0]['objects']) > 0:
            n_valid_object_gt = len(imu_batches[0]['objects'])
        else:
            n_valid_object_gt = 0
        # only consider the text tokens, exclude the object tokens and <|sostatus|> token, <|sot|> token, <|eot|> token etc.
        gt_text_tokens = labels[3][0][None][:, 2:-3-n_valid_object_gt]
        empty_gt_text = True
        if gt_text_tokens.shape[1] > 0:
            empty_gt_text = False
            text_top1_acc = topk_accuracy(text_logits[None, :, :], labels[3][0][None][:, 2:-3-n_valid_object_gt], k=1)
            text_top5_acc = topk_accuracy(text_logits[None, :, :], labels[3][0][None][:, 2:-3-n_valid_object_gt], k=5)
            min_len = min(text_logits.shape[0], labels[3][0][None][:, 2:-3-n_valid_object_gt].shape[1])
            text_ce = F.cross_entropy(text_logits[0:min_len], labels[3][0][None][:, 2:-3-n_valid_object_gt][0, 0:min_len], reduction='mean')

        # import pdb; pdb.set_trace()
        # object_id_top1_acc = topk_accuracy(text_logits[None, -1-n_valid_object_gt:-1, :], labels[4][0][0, :n_valid_object_gt], k=1)

        text_result = torch.tensor(text_result, device=accelerator.device, dtype=torch.long) # decode token to string
        text_result = uni_prompting.text_tokenizer.decode(text_result, skip_special_tokens=True)

        # save the reconstructed trajectory
        sample_pred = {
            'sample_idx': sample_idx,
            'input':{
                'imu': imu_batches[0]['imu_data'],
                'invalid_imu_id_list': invalid_imu_id_list,
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
        }

        mpjpe = compute_mpjpe(sample_pred)[0]

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
        txt_path = os.path.join(save_folder, f"id_{imu_batch_idx}_step_{global_step}.txt")
        with open(txt_path, "w") as f:
            f.write(f"sample_idx: {sample_idx}\n")
            f.write(f"invalid_imu_id_list: {invalid_imu_id_list}\n")
            f.write(f"input imu length: {len(imu_batches[0]['imu_data'])}\n")
            f.write(f"status_top1_acc: {status_top1_acc.item()*100:.4f}%, status_top5_acc: {status_top5_acc.item()*100:.4f}%, status_ce: {status_ce.item():.4f}\n")
            if not empty_gt_text:
                f.write(f"text_top1_acc: {text_top1_acc.item()*100:.4f}%, text_top5_acc: {text_top5_acc.item()*100:.2f}%, text_ce: {text_ce.item():.4f}\n")
            f.write(f"mpjpe: {mpjpe:.4f} mm\n")
            f.write(f"pred_text: {text_result}\n")
            f.write(f"gt_text: {imu_batches[0]['description']}\n")
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

    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    # gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)

    if generate_number:
        # save the errors
        traj_errors = np.array(traj_errors)
        orient_errors = np.array(orient_errors)
        pose_errors = np.array(pose_errors)

        status_top1_accs = np.array(status_top1_accs)
        status_top5_accs = np.array(status_top5_accs)
        status_ces = np.array(status_ces)

        text_top1_accs = np.array(text_top1_accs)
        text_top5_accs = np.array(text_top5_accs)
        text_ces = np.array(text_ces)

        mean_traj_error = traj_errors.mean()
        mean_orient_error = orient_errors.mean()
        mean_pose_error = pose_errors.mean()

        mean_status_top1_acc = status_top1_accs.mean()
        mean_status_top5_acc = status_top5_accs.mean()
        mean_status_ce = status_ces.mean()

        mean_text_top1_acc = text_top1_accs.mean()
        mean_text_top5_acc = text_top5_accs.mean()
        mean_text_ce = text_ces.mean()

        mean_mpjpe = np.array(mpjpes).mean()

        # logger.info(f"Split {split} step {global_step}: mean traj error: {mean_traj_error:.4f}, mean orient error: {mean_orient_error:.4f}, mean pose error: {mean_pose_error:.4f}")

        # add to wandb
        accelerator.log(
            {
                f"metric/L1_error/{split}_traj": mean_traj_error,
                f"metric/L1_error/{split}_orient": mean_orient_error,
                f"metric/L1_error/{split}_pose": mean_pose_error,
                f"metric/token_acc/{split}_status_top1_acc": mean_status_top1_acc,
                f"metric/token_acc/{split}_status_top5_acc": mean_status_top5_acc,
                f"metric/cross_entropy/{split}_status_ce": mean_status_ce,
                f"metric/token_acc/{split}_text_top1_acc": mean_text_top1_acc,
                f"metric/token_acc/{split}_text_top5_acc": mean_text_top5_acc,
                f"metric/cross_entropy/{split}_text_ce": mean_text_ce,
                f"metric/mpjpe/{split}_mpjpe": mean_mpjpe,
            },
            step=global_step,
        )

        with open(os.path.join(save_folder, f"{split}_step_{global_step}_summary.txt"), "w") as f:
            f.write(f"mean_traj_error: {mean_traj_error:.4f}, mean_orient_error: {mean_orient_error:.4f}, mean_pose_error: {mean_pose_error:.4f}\n")
            f.write(f"mean_status_top1_acc: {mean_status_top1_acc*100:.2f}%, mean_status_top5_acc: {mean_status_top5_acc*100:.2f}%, mean_status_ce: {mean_status_ce:.4f}\n")
            f.write(f"mean_text_top1_acc: {mean_text_top1_acc*100:.2f}%, mean_text_top5_acc: {mean_text_top5_acc*100:.2f}%, mean_text_ce: {mean_text_ce:.4f}\n")
            f.write(f"mean_mpjpe: {mean_mpjpe:.2f} mm\n")
        
        # print the file content to console
        with open(os.path.join(save_folder, f"{split}_step_{global_step}_summary.txt"), "r") as f:
            print(f.read())

def save_checkpoint(model, optimizer, lr_scheduler, config, accelerator, global_step, epoch=None, loss=None):
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
        
        # Get state dict and save to single file
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

    return save_path

def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

if __name__ == "__main__":
    main()