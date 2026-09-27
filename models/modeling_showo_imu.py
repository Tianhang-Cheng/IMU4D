# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import sys
# sys.path.append("..")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
# from .sampling import cosine_schedule, mask_by_random_topk
from .phi_imu import PhiForCausalLM
from .embed import IMUAggregator4,  SignalAggregator
from .object_identity_head import ObjectIdentityHead
from utils.rotation2 import recover_absolute_rotation, cont_6d_to_matrix, matrix_to_cont_6d


# The legacy implementation computed a batch-mean reconstruction loss and then
# divided it by the runtime batch size a second time. Keeping 32 as the explicit
# compatibility scale preserves the established batch-32 objective while making
# the loss independent of future micro-batch changes.
_RECONSTRUCTION_REFERENCE_BATCH_SIZE = 32



def _ce_ignoring_all_masked(logits, targets, ignore_index=-100):
    """cross_entropy(reduction='mean', ignore_index) that returns 0 (not NaN) when every
    target is ignored -- happens when a sample's whole channel group is untrusted
    (``motion_supervise``) and the token heads only cover that group."""
    if not (targets != ignore_index).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)

def _segmented_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a mean and count for each segment, using zero for empty ones."""
    totals = values.new_zeros(num_segments)
    if values.numel() > 0:
        totals.scatter_add_(0, segment_ids, values)
        counts = torch.bincount(segment_ids, minlength=num_segments).to(values.dtype)
    else:
        counts = values.new_zeros(num_segments)
    return totals / counts.clamp_min(1), counts


def supervised_mean(per_sample: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Count-weighted mean over the segments that carry a supervised element.

    ``_segmented_mean`` leaves a zero in every segment it saw no element for, so
    a plain ``.mean()`` divides by the full batch and scales the gradient by the
    fraction of samples that carry supervision. That fraction can be small for
    object-track losses in the mixed corpus, and dataset/crop-length filtering
    changes it from batch to batch -- a silent, fluctuating learning-rate cut
    on the object heads. Averaging over the supervised segments removes it.

    Weighting by ``counts`` makes the result the mean over supervised *elements*
    (objects) rather than over samples: a sample carrying one object no longer
    counts as much as a sample carrying ten, which is the second variance source
    on the object-track losses, where a micro-batch typically holds only one or
    two supervised samples (2-15% of the corpus carries object tracks).
    """
    valid = counts > 0
    if not bool(torch.any(valid)):
        return per_sample.new_zeros(())
    weights = counts[valid].to(per_sample.dtype)
    return (per_sample[valid] * weights).sum() / weights.sum()


def object_anchor_losses(pred, target, hidden, sample_ids, valid, batch_size,
                         flow_head=None, detach_latent=False, geometry=None):
    """Both anchor heads use only valid poses, with equal weight per sample.

    Select before evaluating the flow: an uncertain floor must not enter its
    noisy interpolation targets either. An all-invalid batch contributes zero.
    """
    pred, target = pred[valid], target[valid]
    hidden, sample_ids = hidden[valid], sample_ids[valid]
    pose = F.smooth_l1_loss(pred.float(), target.float(), beta=0.5, reduction='none').mean(-1) * 20
    pose_by_sample, counts = _segmented_mean(pose, sample_ids, batch_size)
    flow_by_sample = pose_by_sample.new_zeros(batch_size)
    if flow_head is not None and len(target):
        cond = hidden.detach() if detach_latent else hidden
        if geometry is not None:
            cond = cond + geometry[valid]
        flow_by_sample, _ = _segmented_mean(flow_head.loss(target, cond), sample_ids, batch_size)
    return pose_by_sample, flow_by_sample, counts


@torch.no_grad()
def masked_accuracy(logits, labels):
        mask = labels != -100
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)  # avoid nan
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == labels) & mask
        return correct.float().sum() / mask.sum()


def split_caption_object_targets(
    gt_text_token: torch.Tensor,
    text_token_count: int,
    bidirectional_motion: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split combined AR labels into disjoint caption and object targets.

    Labels are laid out as ``eostatus, sot, text..., eot, soobj,
    object_ids..., eoobj``. Empty captions intentionally receive no caption
    targets, while their object targets remain supervised.
    """
    if text_token_count < 0:
        raise ValueError("text_token_count must be non-negative")
    aligned = gt_text_token[1:] if bidirectional_motion else gt_text_token
    caption_targets = torch.full_like(aligned, -100)
    object_targets = torch.full_like(aligned, -100)
    offset = 1 if bidirectional_motion else 0
    caption_start = 0 if bidirectional_motion else 1
    object_start = 3 + text_token_count - offset
    if object_start < caption_start or object_start > len(aligned):
        raise ValueError("text_token_count is inconsistent with the combined labels")
    if text_token_count > 0:
        caption_targets[caption_start:object_start] = aligned[caption_start:object_start]
    object_targets[object_start:] = aligned[object_start:]
    return caption_targets, object_targets

class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = F.relu(self.linear0(x))
        x = F.elu(self.linear0(x))
        x = self.linear1(x)
        return x


class TemporalTrackDecoder(nn.Module):
    """Sequence-level object-track head.

    Replaces the per-window MLP (one motion query -> one independent
    ``compression_rate``-frame chunk) with a decoder that sees the whole
    status-token sequence: residual 1-D convolutions at token rate, nearest
    upsampling to frame rate, residual convolutions at frame rate, and a
    zero-initialised output projection so the initial residual track is
    exactly zero (constant anchor pose). Input ``[N, T, input_dim]`` ->
    output ``[N, T * compression_rate, output_dim]``.
    """

    def __init__(self, input_dim, output_dim, compression_rate, hidden_dim=256,
                 token_blocks=2, frame_blocks=2, kernel_size=5, root_dim=0,
                 state_head=False):
        super().__init__()
        self.compression_rate = compression_rate
        # Optional per-frame motion-state logit (moving vs static in the
        # world) from the same frame-rate features; separate 1x1 conv so a
        # checkpoint trained without it loads unchanged.
        self.state_out = nn.Conv1d(hidden_dim, 1, 1) if state_head else None
        if self.state_out is not None:
            nn.init.zeros_(self.state_out.weight)
            nn.init.zeros_(self.state_out.bias)
        # Optional frame-rate human root trajectory (absolute transl + 6D
        # orient in the body frame) injected after upsampling, so a carried
        # object can follow the body without re-deriving it from the queries.
        self.root_proj = nn.Linear(root_dim, hidden_dim) if root_dim > 0 else None
        if self.root_proj is not None:
            # Zero-init: a decoder fine-tuned without the root feature keeps its
            # behaviour exactly at load time (root translations reach metres).
            nn.init.zeros_(self.root_proj.weight)
            nn.init.zeros_(self.root_proj.bias)
        # Backbone hidden states carry large-magnitude outlier channels;
        # normalise before projecting so the zero-initialised output stays
        # small during LR warm-up.
        self.inp = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.token_blocks = nn.ModuleList(
            [self._block(hidden_dim, kernel_size) for _ in range(token_blocks)]
        )
        self.frame_blocks = nn.ModuleList(
            [self._block(hidden_dim, kernel_size) for _ in range(frame_blocks)]
        )
        self.out = nn.Conv1d(hidden_dim, output_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _block(dim, kernel_size):
        return nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x, root=None, return_state=False):
        h = self.inp(x).transpose(1, 2)  # [N, C, T]
        for block in self.token_blocks:
            h = h + block(h)
        h = h.repeat_interleave(self.compression_rate, dim=2)  # [N, C, T * cr]
        if self.root_proj is not None:
            if root is None:
                raise ValueError("TemporalTrackDecoder was built with root_dim > 0 but no root trajectory was given")
            h = h + self.root_proj(root.to(h.dtype)).transpose(1, 2)
        for block in self.frame_blocks:
            h = h + block(h)
        track = self.out(h).transpose(1, 2)  # [N, T * cr, out]
        if not return_state:
            return track
        if self.state_out is None:
            raise ValueError("TemporalTrackDecoder was built without a motion-state head")
        return track, self.state_out(h)[:, 0]  # logits [N, T * cr]


def motion_state_from_track(track, speed_threshold, angular_threshold, fps):
    """Per-frame moving flag ``[N, T]`` (bool) from a world pose track
    ``[N, T, 9]`` (6D rot | transl): moving when the translation speed exceeds
    ``speed_threshold`` (m/s) or the angular speed exceeds
    ``angular_threshold`` (deg/s). Frame 0 copies frame 1."""
    track = track.float()
    N, T, _ = track.shape
    if T < 2:
        return torch.ones(N, T, dtype=torch.bool, device=track.device)
    speed = torch.linalg.norm(track[:, 1:, 6:] - track[:, :-1, 6:], dim=-1) * fps
    rot = cont_6d_to_matrix(track[..., :6])
    rel = rot[:, 1:] @ rot[:, :-1].transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    ang_speed = torch.rad2deg(angle) * fps
    moving = (speed > speed_threshold) | (ang_speed > angular_threshold)
    return torch.cat([moving[:, :1], moving], dim=1)


def filter_motion_state(moving, window=5, min_run=0):
    """Majority vote over an odd ``window`` (edge-replicated), then flip every
    run shorter than ``min_run`` frames (never the whole sequence)."""
    moving = moving.bool()
    N, T = moving.shape
    if window > 1 and T >= 2:
        pad = window // 2
        x = F.pad(moving.float()[:, None], (pad, pad), mode='replicate')
        moving = F.avg_pool1d(x, window, stride=1)[:, 0] > 0.5
    if min_run > 1:
        out = moving.cpu().numpy().copy()
        for n in range(N):
            row = out[n]
            changed = True
            while changed:
                changed = False
                edges = np.flatnonzero(np.diff(row.astype(np.int8)))
                bounds = np.concatenate([[0], edges + 1, [T]])
                if len(bounds) <= 2:
                    break
                lengths = np.diff(bounds)
                shortest = int(np.argmin(lengths))
                if lengths[shortest] < min_run:
                    row[bounds[shortest]:bounds[shortest + 1]] = ~row[bounds[shortest]]
                    changed = True
        moving = torch.from_numpy(out).to(moving.device)
    return moving


def run_reference_index(moving):
    """Gate the frames of ``moving`` ``[N, T]`` (bool) into runs. Frame 0 is
    the anchor and never moving. Returns ``(moving, ref)`` where ``ref[n, t]``
    is the index of the frame before the current moving run (a held frame or
    frame 0) for moving frames and ``t`` itself for static frames."""
    moving = moving.bool().clone()
    moving[:, 0] = False
    N, T = moving.shape
    idx = torch.arange(T, device=moving.device)[None].expand(N, -1)
    prev = torch.cat([moving[:, :1] & False, moving[:, :-1]], dim=1)
    start = moving & ~prev
    candidate = torch.where(start, idx - 1, torch.full_like(idx, -1))
    ref = torch.cummax(candidate, dim=1).values
    ref = torch.where(moving, ref, idx)
    return moving, ref


def _gather_time(x, ref):
    """``x[n, ref[n, t]]`` for ``x`` ``[N, T, ...]`` and ``ref`` ``[N, T]``."""
    view = ref.reshape(*ref.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:])
    return torch.gather(x, 1, view)


def relative_to_reference(rot, transl, ref):
    """``R_t R_ref^T`` and ``t_t - t_ref`` for rot ``[N, T, 3, 3]``, transl ``[N, T, 3]``."""
    rel_rot = rot @ _gather_time(rot, ref).transpose(-1, -2)
    rel_transl = transl - _gather_time(transl, ref)
    return rel_rot, rel_transl


def gated_compose_tracks(anchor, residual, p, H, moving, identity_6d):
    """World pose track ``[N, T, 9]`` from the first-frame anchor ``[N, 9]``
    (world), the raw body-frame head output ``[N, T, 9]``, the body frames
    (``p`` ``[N, T, 3]``, ``H`` ``[N, T, 3, 3]``) and the per-frame moving flag
    ``[N, T]``. Static runs hold the world pose of the frame before them;
    moving runs continue in the body frame from that held pose."""
    anchor = anchor.float()
    residual = residual.float()
    N, T, _ = residual.shape
    rel_rot = cont_6d_to_matrix(residual[..., :6] + identity_6d)
    rel_transl = residual[..., 6:]
    out_rot = torch.zeros(N, T, 3, 3, dtype=torch.float32, device=residual.device)
    out_transl = torch.zeros(N, T, 3, dtype=torch.float32, device=residual.device)
    out_rot[:, 0] = cont_6d_to_matrix(anchor[:, :6])
    out_transl[:, 0] = anchor[:, 6:]
    moving, _ = run_reference_index(moving)
    flags = moving.cpu().numpy()
    for n in range(N):
        t = 1
        while t < T:
            b = t
            while b + 1 < T and flags[n, b + 1] == flags[n, t]:
                b += 1
            if not flags[n, t]:
                out_rot[n, t:b + 1] = out_rot[n, t - 1]
                out_transl[n, t:b + 1] = out_transl[n, t - 1]
            else:
                ref = t - 1
                ref_rot = H[n, ref].transpose(-1, -2) @ out_rot[n, ref]
                ref_transl = H[n, ref].transpose(-1, -2) @ (out_transl[n, ref] - p[n, ref])
                body_rot = rel_rot[n, t:b + 1] @ rel_rot[n, ref].transpose(-1, -2) @ ref_rot
                body_transl = ref_transl + rel_transl[n, t:b + 1] - rel_transl[n, ref]
                out_rot[n, t:b + 1] = H[n, t:b + 1] @ body_rot
                out_transl[n, t:b + 1] = p[n, t:b + 1] + torch.einsum('tij,tj->ti', H[n, t:b + 1], body_transl)
            t = b + 1
    return torch.cat([matrix_to_cont_6d(out_rot), out_transl], dim=-1)


IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class AnchorFlowHead(nn.Module):
    """Conditional flow matching over the 9-dim first-frame object pose
    (6D rotation | translation) given the object-token hidden state.

    The regression head collapses to the conditional mean when the first-frame
    pose is ambiguous from IMU + text (6D norms shrink, position -> dataset
    mean). Sampling a straight-line (rectified) flow from N(0, I) instead
    returns one plausible mode. Small MLP velocity field; Euler sampling.
    """

    def __init__(self, cond_dim, data_dim=9, hidden_dim=512, time_dim=64, n_layers=3):
        super().__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        self.cond = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.inp = nn.Linear(data_dim + time_dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        self.out = nn.Linear(hidden_dim, data_dim)

    def time_embedding(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        angles = t.float() * 1000.0 * freqs[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def velocity(self, x_t, t, cond):
        h = torch.cat([x_t, self.time_embedding(t), self.cond(cond)], dim=-1)
        h = self.inp(h)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)

    def loss(self, x1, cond):
        """Per-sample flow-matching MSE, ``x1`` ``[N, data_dim]``, ``cond`` ``[N, cond_dim]``."""
        x1 = x1.float()
        cond = cond.float()
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), 1, device=x1.device, dtype=torch.float32)
        x_t = (1.0 - t) * x0 + t * x1
        return F.mse_loss(self.velocity(x_t, t, cond), x1 - x0, reduction='none').mean(dim=-1)

    @torch.no_grad()
    def sample(self, cond, steps=20, generator=None, noise_scale=1.0):
        cond = cond.float()
        x = noise_scale * torch.randn(len(cond), self.data_dim, device=cond.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((len(cond), 1), i * dt, device=cond.device, dtype=torch.float32)
            x = x + dt * self.velocity(x, t, cond)
        return x


class ConvLayer(nn.Module):
    def __init__(self, n_var: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv1d(n_var, n_var, kernel_size, padding=padding, bias=True)
        self.pw_conv = nn.Conv1d(n_var, n_var, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, L, C]
        residual = x
        x = x.transpose(1, 2)          # [B, C, L]
        x = self.act(self.dw_conv(x))
        x = self.pw_conv(x)
        x = x.transpose(1, 2)          # [B, L, C]
        return x + residual

class ConvRefinement(nn.Module):
    def __init__(self, n_var: int, n_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([ConvLayer(n_var, kernel_size) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, n_var]
        for layer in self.layers:
            x = layer(x)
        return x


class LinearHeadWithDropout(nn.Module):
    """LinearHead variant with dropout after the hidden activation.

    Uses the same linear0/linear1 attribute names as LinearHead so that
    weights can be transferred from a non-dropout checkpoint (strict=False load).
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.linear0(x))
        x = self.dropout(x)
        x = self.linear1(x)
        return x

def expected_value_from_logits(logits, bin_centers, temperature=1.0):
    p = torch.softmax(logits / temperature, dim=1)  # [B, K]
    y_hat = (p * bin_centers.unsqueeze(0)).sum(dim=1)
    return y_hat, p

def masked_l1_l2_loss(input, target, loss_type='l1', threshold=0.01):
    """
    Compute L1 or L2 loss between input and target, 
    but ignore elements where |input - target| < threshold.

    Args:
        input (torch.Tensor): predicted tensor
        target (torch.Tensor): ground truth tensor
        loss_type (str): 'l1' or 'l2'
        threshold (float): difference threshold below which loss=0

    Returns:
        torch.Tensor: scalar loss value
    """
    diff = input - target
    mask = (diff.abs() >= threshold).float()

    if loss_type == 'l1':
        loss = (mask * diff.abs()).mean()
    elif loss_type == 'l2':
        loss = (mask * diff.pow(2)).mean()
    else:
        raise ValueError("loss_type must be 'l1' or 'l2'")

    return loss
 
class ShowoIMU(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    SUPPORTED_BASE_MODELS = ("showo", "gpt2-medium", "qwen3")

    @register_to_config
    def __init__(
            self,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            load_from_showo=True,
            totem_folder=None,
            accumulate=False,
            accumulate_orient=False,
            accumulate_pose=False,
            compression_rate: int = 4,
            time_series_static_vocab_size: int = -1,
            time_series_dynamic_vocab_size: int = -1,
            dynamic_object=False,
            text_only=False, # only supervise the text token
            motion_only=False, # only supervise the motion token
            scene_only=False, # only supervise the object/scene heads
            supervise=None, # explicit subset of ('motion','text','scene'); overrides the *_only flags
            bidirectional_motion=False,
            add_gate=False,
            gpt2_max_position_embeddings=None,
            all_continuous_recon=False,
            partial_continuous_recon=False,
            base_model='showo',
            random_init_backbone=False,
            identity_asset_category_ids=None,
            identity_embedding_dim: int = 256,
            identity_loss_weight: float = 1.0,
            object_pose_loss_weight: float = 1.0,  # weight on loss_object_pose
            text_loss_weight: float = 1.0,         # weight on loss_text
            object_track_head: str = 'mlp',
            object_track_hidden: int = 256,
            object_track_velocity_weight: float = 0.0,
            object_track_acceleration_weight: float = 0.0,
            object_track_loss_scale: float = 200.0,
            object_track_rotation_residual: str = 'add',
            object_anchor_head: str = 'regression',
            object_anchor_flow_hidden: int = 512,
            object_anchor_flow_steps: int = 20,
            object_anchor_flow_weight: float = 1.0,
            object_detach_latent: bool = False,
            object_track_root_feature: bool = False,
            object_track_frame: str = 'world',
            object_track_state_head: bool = False,
            object_track_state_weight: float = 5.0,
            object_track_state_speed_threshold: float = 0.05,
            object_track_state_angular_threshold: float = 5.0,
            object_track_state_filter: int = 5,
            object_track_state_min_run: int = 6,
            object_track_fps: float = 30.0,
            object_geometry_features=None,  # dataset_process/asset_geometry_bps.npz; None = off
            object_geometry_hidden: int = 512,
            # Parallel object-set head (plan A): multi-label category presence
            # read off the <|soobj|> hidden state. Categories are object token
            # ids minus object_set_token_bias; object_set_ignore_ids (the ground
            # plane) are never targets.
            object_set_head: bool = False,
            object_set_num_categories: int = 0,
            object_set_token_bias: int = 0,
            object_set_ignore_ids=None,
            object_set_weight: float = 1.0,
            modality_grad_clip=None,   # per-modality norm cap on the gradient each
                                       # head group sends back into the shared backbone
            modality_grad_scale=None,  # per-modality constant factor on that same
                                       # gradient (applied before the cap)
            modality_grad_stats=False, # record those norms even when nothing is clipped
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        
        self.base_model = base_model

        if base_model not in self.SUPPORTED_BASE_MODELS:
            raise ValueError(
                f"Unsupported base_model={base_model!r}. "
                f"Expected one of {self.SUPPORTED_BASE_MODELS}."
            )

        if random_init_backbone and base_model != 'showo':
            raise NotImplementedError(
                "random_init_backbone is only supported for base_model='showo'"
            )

        if base_model == 'showo':
            assert add_gate is True, "The Show-o/Phi backbone requires add_gate=True"
            if random_init_backbone:
                # phi-1.5 architecture only: build from the config and keep the
                # randomly initialized weights instead of the pretrained ones.
                from transformers.models.phi.configuration_phi import PhiConfig
                backbone_config = PhiConfig.from_pretrained(llm_model_path)
                backbone_config._attn_implementation = 'sdpa'
                self.showo = PhiForCausalLM(backbone_config, add_gate=add_gate)
            else:
                self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa', add_gate=add_gate)

        elif base_model == 'gpt2-medium':
            from transformers import GPT2LMHeadModel
            self.showo = GPT2LMHeadModel.from_pretrained(llm_model_path)
            if gpt2_max_position_embeddings is not None:
                target_positions = int(gpt2_max_position_embeddings)
                current_positions = int(self.showo.config.max_position_embeddings)
                if target_positions < current_positions:
                    raise ValueError(
                        "gpt2_max_position_embeddings cannot shrink the pretrained "
                        f"position table ({target_positions} < {current_positions})"
                    )
                if target_positions > current_positions:
                    old_wpe = self.showo.transformer.wpe
                    new_wpe = nn.Embedding(
                        target_positions,
                        old_wpe.embedding_dim,
                        device=old_wpe.weight.device,
                        dtype=old_wpe.weight.dtype,
                    )
                    nn.init.normal_(
                        new_wpe.weight,
                        mean=0.0,
                        std=float(self.showo.config.initializer_range),
                    )
                    with torch.no_grad():
                        new_wpe.weight[:current_positions].copy_(old_wpe.weight)
                    self.showo.transformer.wpe = new_wpe

                    # GPT-2 stores its causal mask as a fixed-size buffer in
                    # every attention block, so extending only wpe is not
                    # sufficient for sequences beyond the original 1024.
                    for block in self.showo.transformer.h:
                        old_bias = block.attn.bias
                        block.attn.bias = torch.tril(
                            torch.ones(
                                (target_positions, target_positions),
                                dtype=old_bias.dtype,
                                device=old_bias.device,
                            )
                        ).view(1, 1, target_positions, target_positions)

                    self.showo.config.n_positions = target_positions
                    self.showo.config.n_ctx = target_positions
                    self.showo.config.max_position_embeddings = target_positions

        elif base_model == 'qwen3':
            from transformers import AutoModelForCausalLM
            self.showo = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                attn_implementation='sdpa',
            )

        self.llm_hidden_size = self.showo.config.hidden_size
        
        # import pdb; pdb.set_trace()
        if self.base_model == 'gpt2-medium':
            self.showo.resize_token_embeddings(self.vocab_size)
        else:
            self.showo.resize_token_embeddings(self.vocab_size, mean_resizing=False)
        self.output_size = self.vocab_size

        # Which heads receive a loss. ``supervise`` is the general form; the
        # three legacy ``*_only`` booleans are the single-element cases and stay
        # valid (every pre-2026-09-13 profile uses them). A head left out of the
        # set keeps running -- eval still decodes it -- it just carries no loss,
        # and whatever it consumes becomes teacher-forced ground truth: with
        # 'motion' absent and bidirectional_motion=False, the motion positions
        # hold the quantized GT motion, which is what makes the M2T / M2S / M2TS
        # profiles read clean motion instead of the model's own estimate.
        _SUPERVISABLE = ('motion', 'text', 'scene')
        if supervise is not None:
            assert not (text_only or motion_only or scene_only), (
                "model.supervise replaces text_only / motion_only / scene_only; "
                "do not set both"
            )
            supervise = tuple(str(name) for name in supervise)
            unknown = [name for name in supervise if name not in _SUPERVISABLE]
            assert not unknown, (
                f"model.supervise has unknown entries {unknown}; "
                f"expected a subset of {list(_SUPERVISABLE)}"
            )
            assert supervise, "model.supervise must name at least one head"
        else:
            assert sum([bool(text_only), bool(motion_only), bool(scene_only)]) <= 1, (
                "text_only / motion_only / scene_only are mutually exclusive"
            )
            if text_only:
                supervise = ('text',)
            elif motion_only:
                supervise = ('motion',)
            elif scene_only:
                supervise = ('scene',)
            else:
                supervise = _SUPERVISABLE
        self.supervise = supervise
        self.keep_motion = 'motion' in supervise
        self.keep_text = 'text' in supervise
        self.keep_scene = 'scene' in supervise
        # Kept for checkpoints and callers that still read them.
        self.text_only = supervise == ('text',)
        self.motion_only = supervise == ('motion',)
        self.scene_only = supervise == ('scene',)

        # --- per-modality gradient clipping -------------------------------
        # Motion, text and object heads are separate, and the causal layout
        # ([imu, sostatus | motion, eostatus, text, obj]) keeps the motion
        # positions from ever attending to text/object tokens, so the only
        # place the three tasks interact is the shared backbone. Each one
        # reaches it through exactly one slice of `hidden_states`, which makes
        # that slice's gradient the task's entire contribution to the trunk.
        # Capping its norm bounds how hard one modality can pull the backbone
        # in a single micro-batch -- the object losses are heavy-tailed (their
        # sum exceeds the whole motion term in ~34% of steps and reaches 100+
        # on HOI batches, which are only ~5% of the corpus), and those spikes
        # are what the trunk's motion quality pays for.
        clip = dict(modality_grad_clip or {})
        unknown_clip = [k for k in clip if k not in _SUPERVISABLE]
        assert not unknown_clip, (
            f"model.modality_grad_clip has unknown entries {unknown_clip}; "
            f"expected a subset of {_SUPERVISABLE}"
        )
        self.modality_grad_clip = {k: float(v or 0.0) for k, v in clip.items()}
        scale = dict(modality_grad_scale or {})
        unknown_scale = [k for k in scale if k not in _SUPERVISABLE]
        assert not unknown_scale, (
            f"model.modality_grad_scale has unknown entries {unknown_scale}; "
            f"expected a subset of {_SUPERVISABLE}"
        )
        # A constant factor is the right knob when a modality's pull on the
        # trunk is steady rather than spiky (measured 2026-09-13: scene sits at
        # ~2.25x the motion gradient every step, and motion's own max/p50 is
        # only 1.6). Scaling here instead of scaling the loss leaves the
        # modality's own heads learning at full rate -- only its influence on
        # the shared parameters shrinks.
        self.modality_grad_scale = {k: float(v) for k, v in scale.items() if v is not None}
        self.modality_grad_stats = bool(modality_grad_stats)
        # Raw (pre-clip) norms, one entry per micro-batch, drained by run.py at
        # each log step. Kept as device tensors so recording costs no sync.
        self.modality_grad_norm = {}

        self.bidirectional_motion = bidirectional_motion
        self.dynamic_object = dynamic_object
        self.identity_loss_weight = float(identity_loss_weight)
        # Scene and text weights are applied in total_loss only; the reported
        # loss_* entries stay unweighted so logs remain comparable across runs.
        self.object_pose_loss_weight = float(object_pose_loss_weight)
        self.text_loss_weight = float(text_loss_weight)
        self.compression_rate = compression_rate
        self.accumulate = accumulate
        self.accumulate_orient = accumulate_orient
        self.accumulate_pose = accumulate_pose
        self.all_continuous_recon = all_continuous_recon
        self.partial_continuous_recon = partial_continuous_recon
        self.totem_folder = totem_folder
        self.time_series_static_vocab_size = time_series_static_vocab_size
        self.time_series_dynamic_vocab_size = time_series_dynamic_vocab_size
        assert time_series_static_vocab_size > 0, 'time_series_static_vocab_size must be positive'
        assert time_series_dynamic_vocab_size > 0, 'time_series_dynamic_vocab_size must be positive'
        assert not (all_continuous_recon and partial_continuous_recon)

        self.rot_dof = 6

        if all_continuous_recon:
            # all continuous recon: motion and pose
            self.motion_head = LinearHead(self.llm_hidden_size, compression_rate *(3 + self.rot_dof), hidden_dim=128)
            self.motion_refine = ConvRefinement(n_var=3 + self.rot_dof, n_layers=1, kernel_size=3)
            self.pose_head = LinearHead(self.llm_hidden_size, compression_rate * (21 * self.rot_dof), hidden_dim=512)
            self.pose_refine = ConvRefinement(n_var=21 * self.rot_dof, n_layers=1, kernel_size=3)
            assert not accumulate_pose, 'accumulate_pose should be False for continuous recon'

        elif partial_continuous_recon:
            # motion: discrete
            self.mean_head = LinearHead(self.llm_hidden_size, self.time_series_dynamic_vocab_size * (3 + self.rot_dof))
            self.std_head = LinearHead(self.llm_hidden_size, self.time_series_dynamic_vocab_size * (3 + self.rot_dof))
            self.value_head = LinearHead(self.llm_hidden_size, self.time_series_static_vocab_size * (3 + self.rot_dof))
            assert accumulate and accumulate_orient
            # pose: continuous
            self.pose_head = LinearHead(self.llm_hidden_size, compression_rate * (21 * self.rot_dof), hidden_dim=512)
            self.pose_refine = ConvRefinement(n_var=21 * self.rot_dof, n_layers=1, kernel_size=3)
            assert not accumulate_pose, 'accumulate_pose should be False for continuous recon'
        
        else:
            # all discrete
            self.mean_head = LinearHead(self.llm_hidden_size, self.time_series_dynamic_vocab_size * (3 + self.rot_dof))
            self.std_head = LinearHead(self.llm_hidden_size, self.time_series_dynamic_vocab_size * (3 + self.rot_dof))
            self.value_head = LinearHead(self.llm_hidden_size, self.time_series_static_vocab_size * (3 + self.rot_dof))
            self.pose_mean_head = LinearHead(self.llm_hidden_size, self.time_series_dynamic_vocab_size * (21 * self.rot_dof))
            self.pose_std_head = LinearHead(self.llm_hidden_size, self.time_series_dynamic_vocab_size * (21 * self.rot_dof))
            self.pose_value_head = LinearHead(self.llm_hidden_size, self.time_series_static_vocab_size * (21 * self.rot_dof))  # static status head
    
        # Keep the legacy 12-channel object modules so checkpoints trained with
        # rotation(6) + translation(3) + scale(3) load strictly.  Scale is only
        # a compatibility channel: callers feed a fixed neutral value and the
        # loss below supervises rotation + translation only.
        self.object_pose_dim = 3 + self.rot_dof
        self.object_checkpoint_dim = self.object_pose_dim + 3
        self.object_aggregator = IMUAggregator4(
            nvar=self.object_checkpoint_dim,
            d_model=self.llm_hidden_size,
        )
        self.object_mean_head = LinearHead(
            self.llm_hidden_size,
            self.object_checkpoint_dim,
        )
        self.object_set_weight = float(object_set_weight)
        self.object_set_token_bias = int(object_set_token_bias)
        self.object_set_ignore_ids = [int(i) for i in (object_set_ignore_ids or [])]
        self.object_set_classifier = None
        if object_set_head:
            if object_set_num_categories <= 0:
                raise ValueError("object_set_head needs object_set_num_categories > 0")
            self.object_set_classifier = nn.Sequential(
                nn.LayerNorm(self.llm_hidden_size),
                nn.Linear(self.llm_hidden_size, int(object_set_num_categories)),
            )
            # Start near the empirical prior (a few of ~100 categories present)
            # so the summed BCE does not open with a ~70-nat spike.
            nn.init.zeros_(self.object_set_classifier[1].weight)
            nn.init.constant_(self.object_set_classifier[1].bias, -4.0)
        self.identity_head = (
            ObjectIdentityHead(
                self.llm_hidden_size,
                identity_asset_category_ids,
                embedding_dim=identity_embedding_dim,
            )
            if identity_asset_category_ids
            else None
        )
        # TriDi-style shape code of each object's asset (GT at train time, the
        # identity head's retrieval at inference), added to the object hidden
        # state that the pose / anchor / track heads read -- never to the one
        # the identity head reads, which would hand it its own target. The
        # last layer is zero-initialised so a checkpoint trained without it
        # starts out producing exactly the same predictions.
        self.object_geometry_embed = None
        if object_geometry_features:
            if self.identity_head is None:
                raise ValueError("object_geometry_features needs the identity head's asset bank")
            geometry = np.load(object_geometry_features)
            features = torch.as_tensor(geometry['features'], dtype=torch.float32)
            valid = torch.as_tensor(geometry['valid'], dtype=torch.bool)
            if len(features) != self.identity_head.num_assets:
                raise ValueError(
                    f"{object_geometry_features} has {len(features)} assets, "
                    f"the identity bank {self.identity_head.num_assets}"
                )
            mean = features[valid].mean(dim=0)
            std = features[valid].std(dim=0).clamp_min(1e-4)
            features = torch.where(valid[:, None], (features - mean) / std, torch.zeros_like(features))
            self.register_buffer('object_geometry_bank', features, persistent=False)
            self.register_buffer('object_geometry_mask', valid, persistent=False)
            hidden = int(object_geometry_hidden)
            self.object_geometry_embed = nn.Sequential(
                nn.Linear(features.shape[1], hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.llm_hidden_size),
            )
            nn.init.zeros_(self.object_geometry_embed[-1].weight)
            nn.init.zeros_(self.object_geometry_embed[-1].bias)
        
        self.object_track_head = str(object_track_head)
        # Feed the per-frame human root trajectory (GT at training time,
        # decoded prediction at inference) to the conv track decoder.
        self.object_track_root_feature = bool(object_track_root_feature) and self.object_track_head == 'conv'
        # 'world': frame-0-relative track in the (frame-0) body coordinates.
        # 'body': the track is expressed per frame in the moving body frame
        # (pelvis origin, heading-aligned yaw), so a carried object is nearly
        # static and its yaw is relative to the person; composed back to world
        # with the (GT / predicted) root trajectory.
        self.object_track_frame = str(object_track_frame)
        if self.object_track_frame not in ('world', 'body'):
            raise ValueError(f"Unsupported object_track_frame={object_track_frame!r}")
        if self.object_track_frame == 'body' and dynamic_object:
            if not self.object_track_root_feature:
                raise ValueError("object_track_frame='body' requires object_track_root_feature=True (conv head)")
            if str(object_track_rotation_residual) != 'compose':
                raise ValueError("object_track_frame='body' requires object_track_rotation_residual='compose'")
        # 'add': residual 6D is added to the anchor 6D (legacy, not a valid
        # rotation composition). 'compose': the track head predicts the
        # relative rotation R_t R_0^T as 6D (identity offset on the zero
        # output) and it is left-multiplied onto the anchor rotation.
        self.object_track_rotation_residual = str(object_track_rotation_residual)
        if self.object_track_rotation_residual not in ('add', 'compose'):
            raise ValueError(
                f"Unsupported object_track_rotation_residual={object_track_rotation_residual!r}"
            )
        # Detach the backbone latent feeding the flow anchor head and the
        # track head so their (noisy) gradients cannot move the shared
        # human-motion representation.
        self.object_detach_latent = bool(object_detach_latent)
        self.object_anchor_head = str(object_anchor_head)
        self.object_anchor_flow_steps = int(object_anchor_flow_steps)
        self.object_anchor_flow_weight = float(object_anchor_flow_weight)
        if self.object_anchor_head == 'flow':
            self.object_anchor_flow_head = AnchorFlowHead(
                self.llm_hidden_size, self.object_pose_dim, hidden_dim=int(object_anchor_flow_hidden)
            )
        elif self.object_anchor_head != 'regression':
            raise ValueError(
                f"Unsupported object_anchor_head={object_anchor_head!r}; expected 'regression' or 'flow'"
            )
        self.object_track_velocity_weight = float(object_track_velocity_weight)
        self.object_track_acceleration_weight = float(object_track_acceleration_weight)
        # Shared scale of the position / velocity / acceleration track terms.
        self.object_track_loss_scale = float(object_track_loss_scale)
        # Per-frame motion state (moving / static in the world). Static runs
        # hold the world pose, moving runs use the body-frame residual; the
        # GT state is derived from the GT track by speed thresholds.
        self.object_track_state_head = bool(object_track_state_head)
        self.object_track_state_weight = float(object_track_state_weight)
        self.object_track_state_speed_threshold = float(object_track_state_speed_threshold)
        self.object_track_state_angular_threshold = float(object_track_state_angular_threshold)
        self.object_track_state_filter = int(object_track_state_filter)
        self.object_track_state_min_run = int(object_track_state_min_run)
        self.object_track_fps = float(object_track_fps)
        if self.object_track_state_head and dynamic_object:
            if self.object_track_head != 'conv' or self.object_track_frame != 'body':
                raise ValueError("object_track_state_head requires object_track_head='conv' and object_track_frame='body'")
        if dynamic_object:
            # A shared temporal motion query H[t] drives both the human-motion
            # heads and the object-track head. 'mlp': each query predicts one
            # full compression window independently (legacy). 'conv': a
            # temporal decoder over the whole query sequence (see
            # TemporalTrackDecoder), which removes per-window independence.
            if self.object_track_head == 'mlp':
                self.object_dynamic_mean_head = LinearHead(
                    2 * self.llm_hidden_size,
                    self.compression_rate * (3 + self.rot_dof),
                )
            elif self.object_track_head == 'conv':
                self.object_track_decoder = TemporalTrackDecoder(
                    2 * self.llm_hidden_size,
                    3 + self.rot_dof,
                    self.compression_rate,
                    hidden_dim=int(object_track_hidden),
                    root_dim=(3 + self.rot_dof) if object_track_root_feature else 0,
                    state_head=self.object_track_state_head,
                )
            else:
                raise ValueError(
                    f"Unsupported object_track_head={object_track_head!r}; expected 'mlp' or 'conv'"
                )

        self.status_aggregator = SignalAggregator(nvar=(3+self.rot_dof+21*self.rot_dof), d_model=self.llm_hidden_size, chunk_size=self.compression_rate, embed_dim=64)
        if bidirectional_motion:
            # As query embedding for bidirectional motion
            _scale = self.llm_hidden_size ** -0.5
            self.status_learnable_embeddings = nn.Parameter(torch.randn(1, self.llm_hidden_size) * _scale, requires_grad=True)

        # IMU aggregator
        self.imu_aggregator = IMUAggregator4(nvar=(3+3+9)*self.compression_rate, d_model=self.llm_hidden_size) # rotation use 9d

    @property
    def embed_tokens(self):
        """Return the input embedding layer for the selected language backbone."""
        if self.base_model == 'gpt2-medium':
            return self.showo.transformer.wte
        return self.showo.model.embed_tokens

    def _watch_modality_grad(self, tensor, name):
        """Record and optionally norm-clip one modality's gradient into the trunk.

        ``tensor`` must be the single slice of the backbone's ``hidden_states``
        that feeds ``name``'s heads, so its gradient is exactly what that task
        pushes back through the shared parameters. Returns ``tensor`` so it can
        wrap the slicing expression.
        """
        limit = self.modality_grad_clip.get(name, 0.0)
        factor = self.modality_grad_scale.get(name, 1.0)
        if not self.modality_grad_stats and limit <= 0 and factor == 1.0:
            return tensor
        if not torch.is_grad_enabled() or not tensor.requires_grad:
            return tensor

        def hook(grad, name=name, limit=limit, factor=factor):
            # The recorded norm is always the raw one, so a probe run and a
            # clipped/scaled run report the same quantity.
            norm = torch.linalg.vector_norm(grad.detach(), dtype=torch.float32)
            if self.modality_grad_stats:
                self.modality_grad_norm.setdefault(name, []).append(norm)
            adjust = factor
            if limit > 0:
                capped = limit / (norm * factor + 1e-6)
                if capped < 1.0:
                    adjust = factor * capped
            if adjust == 1.0:
                return None
            return grad * (adjust if isinstance(adjust, float) else adjust.to(grad.dtype))

        tensor.register_hook(hook)
        return tensor

    def backbone_forward(self, input_embeddings, attention_mask):
        """Run only the backbone and return its final hidden states.

        The IMU pipeline owns its task heads, so computing the causal LM logits
        inside the backbone would waste substantial memory (especially for the
        Qwen3 vocabulary). Fully causal callers use a compact 2D padding mask;
        prefix/block-attention callers provide a prepared 4D additive mask.
        """
        if self.base_model == 'showo':
            # early_output returns the final hidden states before lm_head.
            # Without it PhiForCausalLM projects the whole sequence onto the
            # 58k vocabulary and upcasts to fp32 -- ~2.5 GiB of transient
            # memory per micro-batch -- only for the result to be discarded;
            # the text logits are computed below on the packed text span.
            return self.showo(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                early_output=True,
            )

        if self.base_model == 'gpt2-medium':
            outputs = self.showo(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return outputs.hidden_states[-1]

        outputs = self.showo.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def backbone_forward_cached(
        self,
        input_embeddings,
        attention_mask,
        past_key_values=None,
    ):
        """Backbone-only inference forward that also returns the KV cache."""
        common_kwargs = dict(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        if self.base_model == 'showo':
            outputs = self.showo.model(**common_kwargs)
        elif self.base_model == 'gpt2-medium':
            outputs = self.showo.transformer(**common_kwargs)
        else:
            outputs = self.showo.model(**common_kwargs)
        return outputs.last_hidden_state, outputs.past_key_values

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def _identity_6d(self, like):
        return torch.tensor(IDENTITY_6D, dtype=like.dtype, device=like.device)

    def body_frames(self, root):
        """Per-frame body frame from root features ``[B, T, 9]``: origin p_t
        (pelvis) and a yaw-only heading rotation H_t built from the pelvis
        x-axis projected on the ground plane (robust to bending forward).
        Returns ``p [B, T, 3]`` and ``H [B, T, 3, 3]`` (columns = frame axes)."""
        root = root.float()
        p = root[..., 0:3]
        R = cont_6d_to_matrix(root[..., 3:9])
        ex = R[..., :, 0].clone()
        ex[..., 1] = 0.0
        ex = F.normalize(ex, dim=-1, eps=1e-6)
        ey = torch.zeros_like(ex)
        ey[..., 1] = 1.0
        ez = torch.linalg.cross(ex, ey, dim=-1)
        H = torch.stack([ex, ey, ez], dim=-1)
        return p, H

    def _to_body_frame(self, track, root):
        """World (frame-0 body coords) pose track ``[N, T, 9]`` -> per-frame body frame."""
        p, H = self.body_frames(root)
        rot = H.transpose(-1, -2) @ cont_6d_to_matrix(track[..., :6].float())
        transl = torch.einsum('ntji,ntj->nti', H, track[..., 6:].float() - p)
        return torch.cat([matrix_to_cont_6d(rot), transl], dim=-1)

    def relative_object_tracks(self, pred, target, root=None):
        """Map raw head output and GT per-frame pose ``[N, T, 9]`` to the
        frame-0-relative quantities compared by the track loss."""
        if self.object_track_rotation_residual == 'add':
            return pred - pred[:, 0:1], target - target[:, 0:1]
        if self.object_track_frame == 'body':
            if root is None:
                raise ValueError("object_track_frame='body' needs the root trajectory")
            target = self._to_body_frame(target, root).to(target.dtype)
        target_rot = cont_6d_to_matrix(target[..., :6].float())  # [N, T, 3, 3]
        relative_rot = target_rot @ target_rot[:, 0:1].transpose(-1, -2)
        target_rel = torch.cat([
            matrix_to_cont_6d(relative_rot).to(target.dtype),
            target[..., 6:] - target[:, 0:1, 6:],
        ], dim=-1)
        pred_rel = torch.cat([
            pred[..., :6] + self._identity_6d(pred),
            pred[..., 6:] - pred[:, 0:1, 6:],
        ], dim=-1)
        return pred_rel, target_rel

    def compose_object_tracks(self, anchor, residual, root=None):
        """Absolute per-frame pose ``[N, T, 9]`` (frame-0 body coords) from the
        first-frame anchor ``[N, 9]`` and the raw track-head output ``[N, T, 9]``.
        In 'body' mode the residual lives in the per-frame body frame given by
        ``root`` ``[N, T, 9]`` and is mapped back to world."""
        if self.object_track_rotation_residual == 'add':
            return anchor[:, None] + (residual - residual[:, 0:1])
        anchor = anchor.float()
        residual = residual.float()
        relative_rot = cont_6d_to_matrix(residual[..., :6] + self._identity_6d(residual))
        anchor_rot = cont_6d_to_matrix(anchor[:, :6])[:, None]  # [N, 1, 3, 3]
        anchor_transl = anchor[:, None, 6:]
        if self.object_track_frame == 'body':
            if root is None:
                raise ValueError("object_track_frame='body' needs the root trajectory")
            p, H = self.body_frames(root)
            # anchor (frame 0, world) -> body frame at frame 0
            anchor_rot = H[:, 0:1].transpose(-1, -2) @ anchor_rot
            anchor_transl = torch.einsum('nji,nj->ni', H[:, 0], anchor[:, 6:] - p[:, 0])[:, None]
        rot = relative_rot @ anchor_rot
        transl = anchor_transl + (residual[..., 6:] - residual[:, 0:1, 6:])
        if self.object_track_frame == 'body':
            rot = H @ rot
            transl = p + torch.einsum('ntij,ntj->nti', H, transl)
        return torch.cat([matrix_to_cont_6d(rot), transl], dim=-1)

    def object_motion_state(self, track):
        """GT moving flag ``[N, T]`` from a GT world pose track (thresholds + run filter)."""
        moving = motion_state_from_track(
            track, self.object_track_state_speed_threshold,
            self.object_track_state_angular_threshold, self.object_track_fps,
        )
        return filter_motion_state(moving, self.object_track_state_filter, self.object_track_state_min_run)

    def decode_motion_state(self, state_logits):
        """Predicted moving flag ``[N, T]`` from logits (threshold 0.5 + run filter)."""
        return filter_motion_state(
            state_logits.float() > 0, self.object_track_state_filter, self.object_track_state_min_run
        )

    def gated_relative_object_tracks(self, pred, target, root, moving):
        """Run-gated version of ``relative_object_tracks`` (body frame): every
        moving run is compared relative to the frame before it (teacher-forced
        GT pose); static frames are masked out. Returns ``pred_rel``,
        ``target_rel`` ``[N, T, 9]`` and the frame mask ``[N, T]`` (bool)."""
        target_body = self._to_body_frame(target, root)
        moving, ref = run_reference_index(moving)
        pred = pred.float()
        pred_rot, pred_transl = relative_to_reference(
            cont_6d_to_matrix(pred[..., :6] + self._identity_6d(pred)), pred[..., 6:], ref
        )
        target_rot, target_transl = relative_to_reference(
            cont_6d_to_matrix(target_body[..., :6]), target_body[..., 6:], ref
        )
        pred_rel = torch.cat([matrix_to_cont_6d(pred_rot), pred_transl], dim=-1)
        target_rel = torch.cat([matrix_to_cont_6d(target_rot), target_transl], dim=-1)
        return pred_rel, target_rel, moving

    def gated_compose_object_tracks(self, anchor, residual, root, moving):
        """``compose_object_tracks`` with per-frame gating: static runs hold
        the world pose, moving runs continue in the body frame."""
        p, H = self.body_frames(root)
        return gated_compose_tracks(anchor, residual, p, H, moving, self._identity_6d(residual.float()))

    def object_geometry_term(self, asset_indices, like):
        """``[N, d_model]`` shape code of each object's catalog asset, zero for
        unknown assets (``-100``) and assets without geometry (the ground).
        ``None`` when geometry conditioning is off."""
        if self.object_geometry_embed is None:
            return None
        index = asset_indices.to(device=like.device, dtype=torch.long).reshape(-1)
        known = (index >= 0) & (index < len(self.object_geometry_mask))
        index = torch.where(known, index, torch.zeros_like(index))
        known = known & self.object_geometry_mask[index]
        weight = self.object_geometry_embed[0].weight
        term = self.object_geometry_embed(self.object_geometry_bank[index].to(weight.dtype))
        return (term * known[:, None].to(term.dtype)).to(like.dtype)

    def sample_object_anchor(self, object_hidden, generator=None, noise_scale=1.0):
        """Sample the first-frame pose ``[N, 9]`` from the flow head and
        orthonormalise its 6D rotation. ``noise_scale`` > 1 widens the prior
        (eval-only diversity knob)."""
        x = self.object_anchor_flow_head.sample(
            object_hidden, steps=self.object_anchor_flow_steps, generator=generator,
            noise_scale=noise_scale,
        )
        rot6d = matrix_to_cont_6d(cont_6d_to_matrix(x[:, :6]))
        return torch.cat([rot6d, x[:, 6:]], dim=-1)

    def root_trajectory_features(self, motion, relative):
        """Absolute per-frame root translation + 6D orientation ``[B, T, 9]``.

        ``relative=False``: ``motion`` is the GT label (``labels[10]`` =
        imu_batch transl / orient, already absolute in frame-0 body coords).
        ``relative=True``: ``motion`` is the tokenizer-decoded ``x_recon`` whose
        transl / orient are per-frame deltas when ``accumulate`` /
        ``accumulate_orient`` (same recovery as eval_model applies to the human).
        """
        motion = motion.float()
        transl = motion[..., 0:3]
        orient = motion[..., 3:3 + self.rot_dof]
        if not relative:
            return torch.cat([transl, orient], dim=-1)
        if self.accumulate:
            transl = torch.cumsum(transl, dim=1)
        if self.accumulate_orient:
            B, T, _ = orient.shape
            orient = recover_absolute_rotation(
                orient.reshape(B, T, 1, self.rot_dof), rot_rep='6d'
            ).reshape(B, T, self.rot_dof)
        return torch.cat([transl, orient], dim=-1)

    def predict_object_tracks(self, object_hidden, status_hidden, root=None, return_state=False,
                              geometry=None):
        """Per-frame object pose residuals from object queries and the shared
        temporal motion queries.

        object_hidden: ``[N, d_model]`` (one row per object token);
        status_hidden: ``[N, T, d_model]`` (motion queries, T status tokens).
        Returns ``[N, T * compression_rate, 3 + rot_dof]`` (6D rot | transl).
        The caller subtracts frame 0 (training target and inference anchor).
        ``geometry`` (``object_geometry_term``) is added after the detach so
        the shape code keeps its gradient under ``object_detach_latent``.
        """
        if self.object_detach_latent:
            object_hidden = object_hidden.detach()
            status_hidden = status_hidden.detach()
        if geometry is not None:
            object_hidden = object_hidden + geometry
        n_status_token = status_hidden.shape[1]
        features = torch.cat([
            object_hidden[:, None].expand(-1, n_status_token, -1),
            status_hidden.to(object_hidden.dtype),
        ], dim=-1)
        out_dim = 3 + self.rot_dof
        state_logits = None
        if self.object_track_head == 'mlp':
            pred = self.object_dynamic_mean_head(features)
        elif self.object_track_root_feature:
            pred = self.object_track_decoder(features, root=root, return_state=return_state)
        else:
            pred = self.object_track_decoder(features, return_state=return_state)
        if return_state:
            if self.object_track_head == 'mlp':
                raise ValueError("the mlp track head has no motion-state output")
            pred, state_logits = pred
        pred = pred.reshape(len(object_hidden), n_status_token * self.compression_rate, out_dim)
        if return_state:
            return pred, state_logits
        return pred

    def forward(
            self,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            batch_size_imu=0,
            input_imu_len=0,
            labels_mask_text=None,
            labels_mask_image=None,

            # bidirectional motion
            # imu_embeddings = None,
            # query_embeddings = None,
            **kwargs,
    ):
        assert attention_mask is not None, 'attention_mask must be provided'
        hidden_states = self.backbone_forward(input_embeddings, attention_mask)

        length = hidden_states.shape[1]
        bs = input_embeddings.shape[0]
        device = input_embeddings.device

        loss_mean_batch = torch.tensor(0.0, device=device)
        loss_std_batch = torch.tensor(0.0, device=device)
        loss_static_status_batch = torch.tensor(0.0, device=device)
        loss_text_batch = torch.tensor(0.0, device=device)
        loss_object_token_batch = torch.tensor(0.0, device=device)
        loss_object_pose_batch = torch.tensor(0.0, device=device)
        loss_obj_dynamic_pose_batch = torch.tensor(0.0, device=device)
        loss_object_anchor_flow_batch = torch.tensor(0.0, device=device)
        loss_object_track_state_batch = torch.tensor(0.0, device=device)
        loss_object_identity_batch = torch.tensor(0.0, device=device)
        obj_track_state_accuracy_batch = torch.tensor(0.0, device=device)

        # Tensors (not Python ints) so accelerator.gather works even when a
        # modality is disabled and its accuracy is never assigned.
        mean_accuracy_batch = torch.tensor(0.0, device=device)
        std_accuracy_batch = torch.tensor(0.0, device=device)
        static_accuracy_batch = torch.tensor(0.0, device=device)
        text_accuracy_batch = torch.tensor(0.0, device=device)
        obj_id_accuracy_batch = torch.tensor(0.0, device=device)
        obj_pose_accuracy_batch = torch.tensor(0.0, device=device)
        obj_dynamic_pose_accuracy_batch = torch.tensor(0.0, device=device)
        object_identity_accuracy_batch = torch.tensor(0.0, device=device)
        n_text_valid = 0

        # Batched status projection (requires uniform n_status_token and input_imu_len across batch)
        n_status_token = labels[0][0].shape[1]
        # When bidirectional_motion: predict from [imu, sostatus | query, eostatus, text, obj]
        # When causal: predict from [imu, sostatus | motion, eostatus, text, obj]
        status_start_idx = (input_imu_len[0] if self.bidirectional_motion else input_imu_len[0] - 1)
        status_end_idx = status_start_idx + n_status_token

        status_hidden_states_all = self._watch_modality_grad(
            hidden_states[:, status_start_idx:status_end_idx], 'motion'
        )  # [bs, n_status_token, d_model]

        if self.all_continuous_recon:
            n_frame = n_status_token * self.compression_rate
            n_var = 3 + 22 * self.rot_dof

            # Direct regression: use float32 for better numerical stability in pose/motion regression
            status_hidden_f32 = status_hidden_states_all.float()
            x_motion = self.motion_head(status_hidden_f32).reshape(bs, n_frame, 3 + self.rot_dof)  # [bs, n_frame, 3 + rot_dof]
            x_pose = self.pose_head(status_hidden_f32).reshape(bs, n_frame, 21 * self.rot_dof)  # [bs, n_frame, 21 * rot_dof]
            x_motion = self.motion_refine(x_motion)
            x_pose = self.pose_refine(x_pose)
            x_recon = torch.cat([x_motion, x_pose], dim=-1) # [bs, n_frame, 3 + 22 * rot_dof]

            # Motion part: transl cumsum, orient recover_absolute_rotation; pose: no accumulation
            # import pdb; pdb.set_trace()
            x_transl = x_recon[:, :, :3]  # [bs, n_frame, 3]
            x_orient = x_recon[:, :, 3:3 + self.rot_dof]  # [bs, n_frame, rot_dof]
            x_pose_part = x_recon[:, :, 3 + self.rot_dof:]  # [bs, n_frame, 21*rot_dof]
            if self.accumulate_orient:
                # recover_absolute_rotation expects [batch, n_time, n_var, rot_dof]
                x_orient = x_orient.reshape(bs, n_frame, 1, self.rot_dof)
                x_orient = recover_absolute_rotation(x_orient, rot_rep='6d').reshape(bs, n_frame, self.rot_dof)
            x_recon = torch.cat([x_transl, x_orient, x_pose_part], dim=-1)

            loss_mean = torch.tensor(0.0, device=device)
            loss_std = torch.tensor(0.0, device=device)
            loss_static_status = torch.tensor(0.0, device=device)

            loss_recon_batch = torch.tensor(0.0, device=device)
            if self.keep_motion:
                gt_motion = torch.stack([labels[10][b].to(device).float() for b in range(bs)])
                # gt_motion: [bs, n_status_token, compression_rate, n_var] -> [bs, n_frame, n_var]
                gt_motion = gt_motion.reshape(bs, n_frame, n_var)
                # loss_recon_batch = F.l1_loss(x_recon.float(), gt_motion.float()) * 10
                # loss_recon_batch = F.mse_loss(x_recon.float(), gt_motion.float()) * 200
                recon_l1 = F.smooth_l1_loss(x_recon.float(), gt_motion.float(), beta=0.02, reduction="none")
                if len(labels) > 15 and labels[15] is not None and len(labels[15]) == bs:
                    # Per-sample channel weights (run.py, from the sample's ``motion_supervise``):
                    # 0 on channel groups whose video pseudo-label is untrusted.
                    w = torch.stack([labels[15][b].to(device).float() for b in range(bs)])  # [bs, n_var]
                    w = w[:, None, :].expand_as(recon_l1)
                    loss_recon_batch = (recon_l1 * w).sum() / w.sum().clamp(min=1.0)
                else:
                    loss_recon_batch = recon_l1.mean()
                loss_recon_batch = loss_recon_batch * (1000 / _RECONSTRUCTION_REFERENCE_BATCH_SIZE)

            # Dummy logits for per-sample loop (accuracy will be 0)
            x_mean_logits_all = None
            x_std_logits_all = None
            x_value_logits_all = None
        elif self.partial_continuous_recon:
            n_frame = n_status_token * self.compression_rate
            # transl + orient: discrete (CE loss), same as all_discrete motion part
            x_mean_logits_all = self.mean_head(status_hidden_states_all).reshape(
                bs, -1, (3 + self.rot_dof), self.time_series_dynamic_vocab_size
            )
            x_std_logits_all = self.std_head(status_hidden_states_all).reshape(
                bs, -1, (3 + self.rot_dof), self.time_series_dynamic_vocab_size
            )
            x_value_logits_all = self.value_head(status_hidden_states_all).reshape(
                bs, -1, (3 + self.rot_dof), self.time_series_static_vocab_size
            )
            # pose: continuous only
            x_pose = self.pose_head(status_hidden_states_all).reshape(bs, n_frame, 21 * self.rot_dof)
            x_pose = self.pose_refine(x_pose)
            # transl + orient CE loss (labels[0/1/2] first 3+rot_dof dims)
            if self.keep_motion:
                loss_mean = _ce_ignoring_all_masked(
                    x_mean_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[0][b][:, :, :3 + self.rot_dof] for b in range(bs)]).reshape(-1),
                )
                loss_std = _ce_ignoring_all_masked(
                    x_std_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[1][b][:, :, :3 + self.rot_dof] for b in range(bs)]).reshape(-1),
                )
                loss_static_status = _ce_ignoring_all_masked(
                    x_value_logits_all.reshape(-1, self.time_series_static_vocab_size),
                    torch.stack([labels[2][b][:, :, :3 + self.rot_dof] for b in range(bs)]).reshape(-1),
                )
                gt_motion = torch.stack([labels[10][b].to(device).float() for b in range(bs)])
                gt_motion = gt_motion.reshape(bs, n_frame, 3 + 22 * self.rot_dof)
                gt_pose = gt_motion[:, :, 3 + self.rot_dof:]
                pose_l1 = F.smooth_l1_loss(x_pose.float(), gt_pose.float(), beta=0.02, reduction="none")
                if len(labels) > 15 and labels[15] is not None and len(labels[15]) == bs:
                    # Per-sample channel weights (run.py, from ``motion_supervise``).
                    # Here the head covers the pose channels only, so take their
                    # slice; today it is all ones (NCSA masks traj/orient, never
                    # pose) but an unweighted mean would silently ignore the flag.
                    w = torch.stack([labels[15][b].to(device).float() for b in range(bs)])[:, 3 + self.rot_dof:]
                    w = w[:, None, :].expand_as(pose_l1)
                    loss_recon_batch = (pose_l1 * w).sum() / w.sum().clamp(min=1.0)
                else:
                    loss_recon_batch = pose_l1.mean()
                loss_recon_batch = loss_recon_batch * (500 / _RECONSTRUCTION_REFERENCE_BATCH_SIZE)
            else:
                loss_mean = torch.tensor(0.0, device=device)
                loss_std = torch.tensor(0.0, device=device)
                loss_static_status = torch.tensor(0.0, device=device)
                loss_recon_batch = torch.tensor(0.0, device=device)
        else:
            x_mean_logits_all = self.mean_head(status_hidden_states_all).reshape(
                bs, -1, (3 + self.rot_dof), self.time_series_dynamic_vocab_size
            )
            x_std_logits_all = self.std_head(status_hidden_states_all).reshape(
                bs, -1, (3 + self.rot_dof), self.time_series_dynamic_vocab_size
            )
            x_value_logits_all = self.value_head(status_hidden_states_all).reshape(
                bs, -1, (3 + self.rot_dof), self.time_series_static_vocab_size
            )

            x_pose_mean_logits_all = self.pose_mean_head(status_hidden_states_all).reshape(
                bs, -1, 21 * self.rot_dof, self.time_series_dynamic_vocab_size
            )
            x_pose_std_logits_all = self.pose_std_head(status_hidden_states_all).reshape(
                bs, -1, 21 * self.rot_dof, self.time_series_dynamic_vocab_size
            )
            x_pose_value_logits_all = self.pose_value_head(status_hidden_states_all).reshape(
                bs, -1, 21 * self.rot_dof, self.time_series_static_vocab_size
            )

            x_mean_logits_all  = torch.cat([x_mean_logits_all,  x_pose_mean_logits_all],  dim=2)
            x_std_logits_all   = torch.cat([x_std_logits_all,   x_pose_std_logits_all],   dim=2)
            x_value_logits_all = torch.cat([x_value_logits_all, x_pose_value_logits_all], dim=2)

            # --- Batched motion losses (outside the per-sample loop) ---
            if self.keep_motion:
                loss_mean = _ce_ignoring_all_masked(
                    x_mean_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[0][b] for b in range(bs)]).reshape(-1),
                )
                loss_std = _ce_ignoring_all_masked(
                    x_std_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[1][b] for b in range(bs)]).reshape(-1),
                )
                loss_static_status = _ce_ignoring_all_masked(
                    x_value_logits_all.reshape(-1, self.time_series_static_vocab_size),
                    torch.stack([labels[2][b] for b in range(bs)]).reshape(-1),
                    ignore_index=-100,
                )
            else:
                loss_mean = torch.tensor(0.0, device=hidden_states.device)
                loss_std  = torch.tensor(0.0, device=hidden_states.device)
                loss_static_status = torch.tensor(0.0, device=hidden_states.device)

            loss_recon_batch = torch.tensor(0.0, device=hidden_states.device)

        # Pack ragged autoregressive spans so the large vocabulary projection
        # and token loss each run once for the whole batch. Per-sample means are
        # recovered below to preserve the legacy sample weighting exactly.
        text_token_lengths = [
            len(gt_text_token) - int(self.bidirectional_motion)
            for gt_text_token in labels[3]
        ]
        text_end_indices = [
            status_end_idx + text_token_len
            for text_token_len in text_token_lengths
        ]
        packed_text_hidden = torch.cat([
            hidden_states[b, status_end_idx:text_end_indices[b]]
            for b in range(bs)
        ], dim=0)
        packed_text_hidden = self._watch_modality_grad(packed_text_hidden, 'text')
        packed_text_logits = self.showo.lm_head(packed_text_hidden)

        text_token_counts = [
            (
                int(labels[12][b])
                if len(labels) > 12
                else max(len(labels[3][b]) - 5 - len(labels[7][b]), 0)
            )
            for b in range(bs)
        ]
        split_targets = [
            split_caption_object_targets(
                labels[3][b], text_token_counts[b], self.bidirectional_motion
            )
            for b in range(bs)
        ]
        caption_targets_by_sample = [targets[0] for targets in split_targets]
        object_targets_by_sample = [targets[1] for targets in split_targets]
        object_id_targets_by_sample = []
        for b, caption_targets in enumerate(caption_targets_by_sample):
            object_id_targets = torch.full_like(caption_targets, -100)
            n_obj_token = len(labels[7][b])
            if n_obj_token > 0:
                object_id_targets[-1 - n_obj_token:-1] = labels[3][b][
                    -1 - n_obj_token:-1
                ]
            object_id_targets_by_sample.append(object_id_targets)
        packed_caption_targets = torch.cat(caption_targets_by_sample)
        packed_object_targets = torch.cat(object_targets_by_sample)
        packed_object_id_targets = torch.cat(object_id_targets_by_sample)
        packed_targets = torch.where(
            packed_caption_targets != -100,
            packed_caption_targets,
            packed_object_targets,
        )
        packed_token_loss = F.cross_entropy(
            packed_text_logits,
            packed_targets,
            ignore_index=-100,
            reduction='none',
        )
        text_sample_ids = torch.repeat_interleave(
            torch.arange(bs, device=device),
            torch.tensor(text_token_lengths, device=device),
        )
        caption_mask = packed_caption_targets != -100
        object_mask = packed_object_targets != -100
        caption_loss_by_sample, _ = _segmented_mean(
            packed_token_loss[caption_mask], text_sample_ids[caption_mask], bs
        )
        object_token_loss_by_sample, _ = _segmented_mean(
            packed_token_loss[object_mask], text_sample_ids[object_mask], bs
        )
        loss_object_set_batch = torch.tensor(0.0, device=device)
        object_set_f1_batch = torch.tensor(0.0, device=device)
        if self.object_set_classifier is not None and self.keep_scene:
            set_rows, set_targets = [], []
            n_categories = self.object_set_classifier[1].out_features
            for b in range(bs):
                n_obj_token = len(labels[7][b])
                if n_obj_token == 0:
                    continue
                # Position j of the text span predicts label j; the first id
                # sits at L-1-n (see object_id_targets above), so that row is
                # the <|soobj|> hidden state.
                set_rows.append(hidden_states[b, status_end_idx + text_token_lengths[b] - 1 - n_obj_token])
                ids = labels[3][b][-1 - n_obj_token:-1].to(device) - self.object_set_token_bias
                target = torch.zeros(n_categories, device=device)
                target[ids] = 1.0
                if self.object_set_ignore_ids:
                    target[self.object_set_ignore_ids] = 0.0
                set_targets.append(target)
            if set_rows:
                set_logits = self.object_set_classifier(torch.stack(set_rows)).float()
                set_targets = torch.stack(set_targets)
                valid = torch.ones(n_categories, dtype=torch.bool, device=device)
                if self.object_set_ignore_ids:
                    valid[self.object_set_ignore_ids] = False
                loss_object_set_batch = F.binary_cross_entropy_with_logits(
                    set_logits[:, valid], set_targets[:, valid], reduction='none'
                ).sum(dim=1).mean()
                with torch.no_grad():
                    pred = (set_logits[:, valid] > 0).float()
                    tp = (pred * set_targets[:, valid]).sum()
                    object_set_f1_batch = 2 * tp / (pred.sum() + set_targets[:, valid].sum()).clamp_min(1.0)
        with torch.no_grad():
            packed_text_predictions = packed_text_logits.argmax(dim=-1)
            caption_correct = (
                packed_text_predictions[caption_mask]
                == packed_caption_targets[caption_mask]
            ).float()
            caption_accuracy_by_sample, _ = _segmented_mean(
                caption_correct, text_sample_ids[caption_mask], bs
            )
            object_id_mask = packed_object_id_targets != -100
            object_id_correct = (
                packed_text_predictions[object_id_mask]
                == packed_object_id_targets[object_id_mask]
            ).float()
            object_id_accuracy_by_sample, _ = _segmented_mean(
                object_id_correct, text_sample_ids[object_id_mask], bs
            )

        # Pack ragged object spans so the pose, identity, and dynamic heads and
        # their losses can also operate on the whole batch.
        object_counts = [len(gt_object_status) for gt_object_status in labels[7]]
        object_hidden_chunks = []
        object_chunk_indices = []
        pose_loss_by_sample = hidden_states.new_zeros(bs)
        pose_accuracy_by_sample = hidden_states.new_zeros(bs)
        identity_loss_by_sample = hidden_states.new_zeros(bs)
        identity_accuracy_by_sample = hidden_states.new_zeros(bs)
        dynamic_loss_by_sample = hidden_states.new_zeros(bs)
        dynamic_accuracy_by_sample = hidden_states.new_zeros(bs)
        track_state_loss_by_sample = hidden_states.new_zeros(bs)
        track_state_accuracy_by_sample = hidden_states.new_zeros(bs)
        anchor_flow_loss_by_sample = hidden_states.new_zeros(bs)
        # Per-sample object counts behind each loss. The dynamic-track and
        # motion-state terms only see objects with a real trajectory (the
        # constant ground plane is excluded), so a batch that mixes
        # object-free sources with HOI sources must average over the samples
        # that actually carry supervision instead of over the whole batch --
        # otherwise the object-head gradient is silently scaled by the HOI
        # fraction of the batch.
        object_count_by_sample = hidden_states.new_zeros(bs)
        dynamic_count_by_sample = hidden_states.new_zeros(bs)
        dynamic_track_count_by_sample = hidden_states.new_zeros(bs)
        for b, n_obj_token in enumerate(object_counts):
            if n_obj_token == 0:
                continue
            object_start_idx = text_end_indices[b]
            object_hidden_chunks.append(
                hidden_states[b, object_start_idx:object_start_idx + n_obj_token]
            )
            object_chunk_indices.append(b)
        if object_hidden_chunks:
            packed_object_hidden = self._watch_modality_grad(
                torch.cat(object_hidden_chunks, dim=0), 'scene'
            )
            packed_geometry = None
            if self.object_geometry_embed is not None and len(labels) > 11:
                packed_geometry = self.object_geometry_term(
                    torch.cat([labels[11][b] for b in object_chunk_indices]),
                    packed_object_hidden,
                )
            packed_object_pred = self.object_mean_head(
                packed_object_hidden if packed_geometry is None
                else packed_object_hidden + packed_geometry
            )
            object_sample_ids = torch.cat([
                torch.full(
                    (object_counts[b],), b, dtype=torch.long, device=device
                )
                for b in object_chunk_indices
            ])
            packed_object_status = torch.cat([
                labels[7][b] for b in object_chunk_indices
            ]).to(device)
            anchor_valid = (
                torch.cat([labels[14][b].to(device) for b in object_chunk_indices])
                if len(labels) > 14 else torch.ones_like(object_sample_ids, dtype=torch.bool)
            )
            # Reliable ground remains supervised: inference samples its pose
            # from this head. Unknown/uncertain floors are masked in BOTH heads.
            pose_loss_by_sample, anchor_flow_loss_by_sample, object_count_by_sample = object_anchor_losses(
                packed_object_pred[..., :self.object_pose_dim], packed_object_status,
                packed_object_hidden, object_sample_ids, anchor_valid, bs,
                flow_head=self.object_anchor_flow_head if self.object_anchor_head == 'flow' else None,
                detach_latent=self.object_detach_latent,
                geometry=packed_geometry,
            )
            with torch.no_grad():
                pose_error_per_object = torch.abs(
                    packed_object_pred[..., :self.object_pose_dim]
                    - packed_object_status.to(packed_object_pred.dtype)
                ).mean(dim=-1)
                pose_accuracy_by_sample, _ = _segmented_mean(
                    pose_error_per_object[anchor_valid], object_sample_ids[anchor_valid], bs
                )

            if self.keep_scene and self.identity_head is not None and len(labels) > 11:
                identity_chunks = []
                for b in object_chunk_indices:
                    gt_identity = labels[11][b]
                    if len(gt_identity) != object_counts[b]:
                        raise ValueError(
                            "object identity labels must align with category tokens"
                        )
                    identity_chunks.append(gt_identity)
                packed_identity = torch.cat(identity_chunks).to(
                    device=device, dtype=torch.long
                )
                identity_valid = packed_identity != -100
                identity_targets = packed_identity[identity_valid]
                if torch.any(
                    (identity_targets < 0)
                    | (identity_targets >= self.identity_head.num_assets)
                ):
                    raise ValueError("target asset index is outside the global asset bank")
                if identity_targets.numel() > 0:
                    identity_categories = self.identity_head.asset_category_ids[
                        identity_targets
                    ]
                    identity_logits = self.identity_head(
                        packed_object_hidden[identity_valid], identity_categories
                    )
                    identity_loss_per_object = F.cross_entropy(
                        identity_logits, identity_targets, reduction='none'
                    )
                    identity_loss_by_sample, _ = _segmented_mean(
                        identity_loss_per_object,
                        object_sample_ids[identity_valid],
                        bs,
                    )
                    with torch.no_grad():
                        identity_correct = (
                            identity_logits.argmax(dim=-1) == identity_targets
                        ).float()
                        identity_accuracy_by_sample, _ = _segmented_mean(
                            identity_correct,
                            object_sample_ids[identity_valid],
                            bs,
                        )

            # Objects with a real trajectory. ``labels[13]`` marks the
            # constant ground plane (dataset_process/object_taxonomy.py
            # is_static_ground), which every source carries and which would
            # otherwise contribute 913k degenerate zero-motion targets once
            # MotionMillion is trained in dynamic-object mode: a free ride for
            # the track decoder and an overwhelming "static" prior for the
            # motion-state head. It is dropped from the track / state losses
            # and never enters the temporal decoder at all, so the per-step
            # cost of the object-free sources stays where it was.
            if self.dynamic_object:
                if len(labels) > 13 and labels[13]:
                    dynamic_object_mask = torch.cat([
                        labels[13][b].to(device) for b in object_chunk_indices
                    ])
                else:
                    # Older label layout: no mask, every object is dynamic.
                    dynamic_object_mask = torch.ones(
                        packed_object_hidden.shape[0], dtype=torch.bool, device=device
                    )
                dynamic_indices = dynamic_object_mask.nonzero(as_tuple=True)[0]
            else:
                dynamic_indices = object_sample_ids.new_empty(0)

            if dynamic_indices.numel() > 0:
                dynamic_sample_ids = object_sample_ids[dynamic_indices]
                packed_root = None
                if self.object_track_root_feature:
                    # Teacher forcing: GT root trajectory (the decoded
                    # prediction is used at inference, see run.eval_model).
                    root_motion = torch.stack(
                        [labels[10][b].to(device).float() for b in range(bs)]
                    )
                    root_motion = root_motion.reshape(bs, n_status_token * self.compression_rate, -1)
                    packed_root = self.root_trajectory_features(root_motion, relative=False)[dynamic_sample_ids]
                packed_dynamic_hidden = packed_object_hidden[dynamic_indices]
                packed_dynamic_geometry = (
                    None if packed_geometry is None else packed_geometry[dynamic_indices]
                )
                packed_dynamic_target = torch.cat([
                    labels[8][b] for b in object_chunk_indices
                ]).to(device)[dynamic_indices]
                if self.object_track_state_head:
                    packed_dynamic_pred, packed_state_logits = self.predict_object_tracks(
                        packed_dynamic_hidden, status_hidden_states_all[dynamic_sample_ids],
                        root=packed_root, return_state=True, geometry=packed_dynamic_geometry,
                    )
                    with torch.no_grad():
                        gt_moving = self.object_motion_state(packed_dynamic_target)
                    packed_dynamic_pred, packed_dynamic_target, frame_mask = self.gated_relative_object_tracks(
                        packed_dynamic_pred, packed_dynamic_target, packed_root, gt_moving
                    )
                    # Frame 0 is the anchor; supervise the state on frames >= 1.
                    state_target = gt_moving[:, 1:].float()
                    state_loss_per_object = F.binary_cross_entropy_with_logits(
                        packed_state_logits[:, 1:].float(), state_target, reduction='none'
                    ).mean(dim=1)
                    track_state_loss_by_sample, _ = _segmented_mean(
                        state_loss_per_object, dynamic_sample_ids, bs
                    )
                    with torch.no_grad():
                        state_correct = ((packed_state_logits[:, 1:] > 0) == (state_target > 0.5)).float().mean(dim=1)
                        track_state_accuracy_by_sample, _ = _segmented_mean(
                            state_correct, dynamic_sample_ids, bs
                        )
                else:
                    packed_dynamic_pred = self.predict_object_tracks(
                        packed_dynamic_hidden, status_hidden_states_all[dynamic_sample_ids], root=packed_root,
                        geometry=packed_dynamic_geometry,
                    )
                    packed_dynamic_pred, packed_dynamic_target = self.relative_object_tracks(
                        packed_dynamic_pred, packed_dynamic_target, root=packed_root
                    )
                    frame_mask = None

                def _masked_frame_mean(per_frame, mask):
                    # per_frame [N, T', D] -> per-object mean over the masked
                    # frames (0 for objects without any masked frame).
                    per_frame = per_frame.mean(dim=-1)
                    if mask is None:
                        return per_frame.mean(dim=1)
                    mask = mask.to(per_frame.dtype)
                    return (per_frame * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

                dynamic_loss_per_object = _masked_frame_mean(F.mse_loss(
                    packed_dynamic_pred,
                    packed_dynamic_target,
                    reduction='none',
                ), frame_mask) * self.object_track_loss_scale
                # Temporal-smoothness terms: plain per-frame MSE barely
                # penalises centimetre-scale per-frame noise, which is what
                # makes decoded tracks jitter. Velocity / acceleration are in
                # per-frame units and scaled like the position term. With the
                # state gate they only act inside a moving run.
                if self.object_track_velocity_weight > 0 and packed_dynamic_pred.shape[1] > 1:
                    pred_vel = packed_dynamic_pred[:, 1:] - packed_dynamic_pred[:, :-1]
                    target_vel = packed_dynamic_target[:, 1:] - packed_dynamic_target[:, :-1]
                    vel_mask = None if frame_mask is None else (frame_mask[:, 1:] & frame_mask[:, :-1])
                    dynamic_loss_per_object = dynamic_loss_per_object + _masked_frame_mean(F.mse_loss(
                        pred_vel, target_vel, reduction='none'
                    ), vel_mask) * self.object_track_loss_scale * self.object_track_velocity_weight
                    if self.object_track_acceleration_weight > 0 and pred_vel.shape[1] > 1:
                        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
                        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
                        acc_mask = None if vel_mask is None else (vel_mask[:, 1:] & vel_mask[:, :-1])
                        dynamic_loss_per_object = dynamic_loss_per_object + _masked_frame_mean(F.mse_loss(
                            pred_acc, target_acc, reduction='none'
                        ), acc_mask) * self.object_track_loss_scale * self.object_track_acceleration_weight
                # The track terms only exist on gated (moving) frames, so an
                # object that is static for the whole crop contributes a
                # constant zero. Counting it would dilute the count-weighted
                # mean with samples that carry no track supervision at all; the
                # motion-state head is supervised on those objects and keeps
                # the full per-sample object count.
                if frame_mask is None:
                    track_supervised = torch.ones_like(dynamic_sample_ids, dtype=torch.bool)
                else:
                    track_supervised = frame_mask.any(dim=1)
                track_sample_ids = dynamic_sample_ids[track_supervised]
                dynamic_loss_by_sample, dynamic_track_count_by_sample = _segmented_mean(
                    dynamic_loss_per_object[track_supervised], track_sample_ids, bs
                )
                dynamic_count_by_sample = torch.bincount(
                    dynamic_sample_ids, minlength=bs
                ).to(dynamic_loss_per_object.dtype)
                with torch.no_grad():
                    dynamic_error_per_object = _masked_frame_mean(torch.abs(
                        packed_dynamic_pred - packed_dynamic_target
                    ), frame_mask)
                    dynamic_accuracy_by_sample, _ = _segmented_mean(
                        dynamic_error_per_object[track_supervised], track_sample_ids, bs
                    )

        valid_text_samples = torch.tensor(
            [count > 0 for count in text_token_counts], device=device
        )
        n_text_valid = sum(count > 0 for count in text_token_counts)
        if self.keep_text and n_text_valid > 0:
            loss_text_batch = caption_loss_by_sample[valid_text_samples].mean()
            text_accuracy_batch = caption_accuracy_by_sample[
                valid_text_samples
            ].mean()
        elif self.keep_text:
            # Every sample of this micro-batch is caption-less (a third of the
            # MotionMillion clips carry no description, and the length buckets
            # group them together). The value is still zero, but it keeps the
            # graph connected: under text_only supervision this is the only
            # loss, so a detached zero makes backward fail.
            loss_text_batch = packed_text_logits.float().mean() * 0.0
        if self.keep_scene:
            loss_object_token_batch = object_token_loss_by_sample.mean()
            loss_object_pose_batch = supervised_mean(pose_loss_by_sample, object_count_by_sample)
            loss_object_identity_batch = identity_loss_by_sample.mean()
            if self.dynamic_object:
                loss_obj_dynamic_pose_batch = supervised_mean(
                    dynamic_loss_by_sample, dynamic_track_count_by_sample
                )
                if self.object_track_state_head:
                    loss_object_track_state_batch = supervised_mean(
                        track_state_loss_by_sample, dynamic_count_by_sample
                    )
                    obj_track_state_accuracy_batch = supervised_mean(
                        track_state_accuracy_by_sample, dynamic_count_by_sample
                    )
            if self.object_anchor_head == 'flow':
                loss_object_anchor_flow_batch = supervised_mean(
                    anchor_flow_loss_by_sample, object_count_by_sample
                )
        obj_id_accuracy_batch = object_id_accuracy_by_sample.mean()
        obj_pose_accuracy_batch = pose_accuracy_by_sample.mean()
        object_identity_accuracy_batch = identity_accuracy_by_sample.mean()
        if self.dynamic_object:
            obj_dynamic_pose_accuracy_batch = supervised_mean(
                dynamic_accuracy_by_sample, dynamic_track_count_by_sample
            )

        if not self.all_continuous_recon:
            loss_mean_batch = loss_mean / 3.0
            loss_std_batch = loss_std / 3.0
            loss_static_status_batch = loss_static_status / 3.0

        # Only the inexpensive per-sample motion metrics remain ragged here;
        # all trainable text/object heads and losses above are batchified.
        for b in range(bs):
            if self.all_continuous_recon:
                x_mean_logits = x_std_logits = x_value_logits = None
            else:
                x_mean_logits  = x_mean_logits_all[b:b+1]
                x_std_logits   = x_std_logits_all[b:b+1]
                x_value_logits = x_value_logits_all[b:b+1]

            # --- accuracy (no_grad) ---
            with torch.no_grad():
                if self.all_continuous_recon:
                    mean_accuracy   = torch.tensor(0.0, device=hidden_states.device)
                    std_accuracy    = torch.tensor(0.0, device=hidden_states.device)
                    static_accuracy = torch.tensor(0.0, device=hidden_states.device)
                elif self.partial_continuous_recon:
                    mean_accuracy   = masked_accuracy(x_mean_logits,  labels[0][b][:, :, :3 + self.rot_dof])
                    std_accuracy     = masked_accuracy(x_std_logits,   labels[1][b][:, :, :3 + self.rot_dof])
                    static_accuracy = masked_accuracy(x_value_logits, labels[2][b][:, :, :3 + self.rot_dof])
                else:
                    mean_accuracy    = masked_accuracy(x_mean_logits,  labels[0][b])
                    std_accuracy     = masked_accuracy(x_std_logits,   labels[1][b])
                    static_accuracy  = masked_accuracy(x_value_logits, labels[2][b])

            mean_accuracy_batch           += mean_accuracy
            std_accuracy_batch            += std_accuracy
            static_accuracy_batch         += static_accuracy

        # --- normalize remaining motion metrics by batch size ---
        mean_accuracy_batch           /= bs
        std_accuracy_batch            /= bs
        static_accuracy_batch         /= bs

        total_loss = (loss_mean_batch + loss_std_batch + loss_static_status_batch
                    + self.text_loss_weight * loss_text_batch
                    + loss_object_token_batch
                    + self.object_pose_loss_weight * loss_object_pose_batch
                    + self.identity_loss_weight * loss_object_identity_batch)
        if self.dynamic_object:
            total_loss = total_loss + loss_obj_dynamic_pose_batch
            if self.object_track_state_head:
                total_loss = total_loss + self.object_track_state_weight * loss_object_track_state_batch
        if self.object_anchor_head == 'flow':
            total_loss = total_loss + self.object_anchor_flow_weight * loss_object_anchor_flow_batch
        if self.object_set_classifier is not None:
            total_loss = total_loss + self.object_set_weight * loss_object_set_batch
        total_loss = total_loss + loss_recon_batch
        loss_recon_out = loss_recon_batch 

        out = {
            'loss_mean':               loss_mean_batch,
            'loss_std':                loss_std_batch,
            'loss_static':             loss_static_status_batch,
            'loss_recon':              loss_recon_out,
            'loss_text':               loss_text_batch,
            'loss_object_token':       loss_object_token_batch,
            'loss_object_pose':        loss_object_pose_batch,
            'loss_obj_dynamic_pose':   loss_obj_dynamic_pose_batch,
            'loss_object_anchor_flow': loss_object_anchor_flow_batch,
            'loss_object_track_state': loss_object_track_state_batch,
            'obj_track_state_accuracy': obj_track_state_accuracy_batch,
            'loss_object_identity':    loss_object_identity_batch,
            'loss_object_set':         loss_object_set_batch,
            'object_set_f1':           object_set_f1_batch,
            'mean_accuracy':           mean_accuracy_batch,
            'std_accuracy':            std_accuracy_batch,
            'static_accuracy':         static_accuracy_batch,
            'text_accuracy':           text_accuracy_batch,
            'obj_id_accuracy':         obj_id_accuracy_batch,
            'obj_pose_accuracy':       obj_pose_accuracy_batch,
            'obj_dynamic_pose_accuracy': obj_dynamic_pose_accuracy_batch,
            'object_identity_accuracy': object_identity_accuracy_batch,
            'total_loss':              total_loss,
            'length':                  length,
        }
        # The recon outputs are computed whenever the recon mode is active;
        # keep_motion only gates their losses. eval_model still decodes motion
        # under text_only, so return them regardless of keep_motion.
        if self.all_continuous_recon:
            out['x_recon'] = x_recon
        if self.partial_continuous_recon:
            out['x_mean_logits_all'] = x_mean_logits_all
            out['x_std_logits_all'] = x_std_logits_all
            out['x_value_logits_all'] = x_value_logits_all
            out['x_pose'] = x_pose
        # Bidirectional motion decoding also consumes the status-query hidden
        # states in ``run.eval_model``; dynamic-object prediction is not the
        # only caller.
        if self.dynamic_object or self.bidirectional_motion:
            out['motion_query_hidden_states'] = status_hidden_states_all
        return out
