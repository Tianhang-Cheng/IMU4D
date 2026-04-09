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
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
# from .sampling import cosine_schedule, mask_by_random_topk
from .phi_imu import PhiForCausalLM
from .embed import IMUAggregator4,  SignalAggregator
from utils.rotation2 import recover_absolute_rotation


@torch.no_grad()
def masked_accuracy(logits, labels):
        mask = labels != -100
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)  # avoid nan
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == labels) & mask
        return correct.float().sum() / mask.sum()

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
            bidirectional_motion=False,
            add_gate=False,
            all_continuous_recon=False,
            partial_continuous_recon=False,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        
        base_model = kwargs.get('base_model', 'showo')
        self.base_model = base_model

        assert add_gate is True

        if base_model == 'showo':
            self.llm_hidden_size = 2048
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa', add_gate=add_gate)

        elif base_model == 'gpt2-medium':
            from transformers import GPT2LMHeadModel
            self.showo = GPT2LMHeadModel.from_pretrained(llm_model_path)
            self.llm_hidden_size = self.showo.config.n_embd  # 1024 for gpt2-medium
        
        # import pdb; pdb.set_trace()
        if self.base_model == 'gpt2-medium':
            self.showo.resize_token_embeddings(self.vocab_size)
        else:
            self.showo.resize_token_embeddings(self.vocab_size, mean_resizing=False)
        self.output_size = self.vocab_size

        self.keep_text = not motion_only
        self.keep_motion = not text_only
        self.keep_scene = not (text_only or motion_only)

        self.bidirectional_motion = bidirectional_motion
        self.dynamic_object = dynamic_object
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
    
        self.object_aggregator = IMUAggregator4(nvar=12, d_model=self.llm_hidden_size) # obj_rot(6) + obj_transl(3) + bbox (3), linear projection
        self.object_mean_head = LinearHead(self.llm_hidden_size, 3 + self.rot_dof + 3) # obj_rot(6) + obj_transl(3) + bbox (3)
        
        if dynamic_object:
            self.object_dynamic_mean_head = LinearHead(self.llm_hidden_size + (22*self.rot_dof+3), (3 + self.rot_dof + 3))

        self.status_aggregator = SignalAggregator(nvar=(3+self.rot_dof+21*self.rot_dof), d_model=self.llm_hidden_size, chunk_size=self.compression_rate, embed_dim=64)
        if bidirectional_motion:
            # As query embedding for bidirectional motion
            _scale = self.llm_hidden_size ** -0.5
            self.status_learnable_embeddings = nn.Parameter(torch.randn(1, self.llm_hidden_size) * _scale, requires_grad=True)

        # IMU aggregator
        self.imu_aggregator = IMUAggregator4(nvar=(3+3+9)*self.compression_rate, d_model=self.llm_hidden_size) # rotation use 9d

    @property
    def embed_tokens(self):
        """Embedding layer: Phi uses showo.model.embed_tokens, GPT2 uses showo.transformer.wte."""
        if self.base_model == 'gpt2-medium':
            return self.showo.transformer.wte
        return self.showo.model.embed_tokens

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

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
        if self.base_model == 'gpt2-medium':
            outputs = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask,
                                 output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        elif self.base_model == 'showo':
            hidden_states = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)

        length = hidden_states.shape[1]
        bs = input_embeddings.shape[0]
        device = input_embeddings.device

        loss_mean_batch = torch.tensor(0.0, device=device)
        loss_std_batch = torch.tensor(0.0, device=device)
        loss_static_status_batch = torch.tensor(0.0, device=device)
        loss_text_batch = torch.tensor(0.0, device=device)
        loss_object_pose_batch = torch.tensor(0.0, device=device)
        loss_obj_dynamic_pose_batch = torch.tensor(0.0, device=device)

        mean_accuracy_batch = 0
        std_accuracy_batch = 0
        static_accuracy_batch = 0
        text_accuracy_batch = 0
        obj_id_accuracy_batch = 0
        obj_pose_accuracy_batch = 0
        obj_dynamic_pose_accuracy_batch = 0

        # Batched status projection (requires uniform n_status_token and input_imu_len across batch)
        n_status_token = labels[0][0].shape[1]
        # When bidirectional_motion: predict from [imu, sostatus | query, eostatus, text, obj]
        # When causal: predict from [imu, sostatus | motion, eostatus, text, obj]
        status_start_idx = (input_imu_len[0] if self.bidirectional_motion else input_imu_len[0] - 1)
        status_end_idx = status_start_idx + n_status_token

        status_hidden_states_all = hidden_states[:, status_start_idx:status_end_idx]  # [bs, n_status_token, d_model]

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
                loss_recon_batch = F.smooth_l1_loss(x_recon.float(), gt_motion.float(), beta=0.02) * 1000

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
                loss_mean = F.cross_entropy(
                    x_mean_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[0][b][:, :, :3 + self.rot_dof] for b in range(bs)]).reshape(-1),
                    ignore_index=-100,
                )
                loss_std = F.cross_entropy(
                    x_std_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[1][b][:, :, :3 + self.rot_dof] for b in range(bs)]).reshape(-1),
                    ignore_index=-100,
                )
                loss_static_status = F.cross_entropy(
                    x_value_logits_all.reshape(-1, self.time_series_static_vocab_size),
                    torch.stack([labels[2][b][:, :, :3 + self.rot_dof] for b in range(bs)]).reshape(-1),
                    ignore_index=-100,
                )
                gt_motion = torch.stack([labels[10][b].to(device).float() for b in range(bs)])
                gt_motion = gt_motion.reshape(bs, n_frame, 3 + 22 * self.rot_dof)
                gt_pose = gt_motion[:, :, 3 + self.rot_dof:]
                loss_recon_batch = F.smooth_l1_loss(x_pose.float(), gt_pose.float(), beta=0.02) * 500
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
                loss_mean = F.cross_entropy(
                    x_mean_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[0][b] for b in range(bs)]).reshape(-1),
                    ignore_index=-100,
                )
                loss_std = F.cross_entropy(
                    x_std_logits_all.reshape(-1, self.time_series_dynamic_vocab_size),
                    torch.stack([labels[1][b] for b in range(bs)]).reshape(-1),
                    ignore_index=-100,
                )
                loss_static_status = F.cross_entropy(
                    x_value_logits_all.reshape(-1, self.time_series_static_vocab_size),
                    torch.stack([labels[2][b] for b in range(bs)]).reshape(-1),
                    ignore_index=-100,
                )
            else:
                loss_mean = torch.tensor(0.0, device=hidden_states.device)
                loss_std  = torch.tensor(0.0, device=hidden_states.device)
                loss_static_status = torch.tensor(0.0, device=hidden_states.device)

            loss_recon_batch = torch.tensor(0.0, device=hidden_states.device)

        for b in range(bs):
            if self.all_continuous_recon:
                x_mean_logits = x_std_logits = x_value_logits = None
            else:
                x_mean_logits  = x_mean_logits_all[b:b+1]
                x_std_logits   = x_std_logits_all[b:b+1]
                x_value_logits = x_value_logits_all[b:b+1]

            # --- text loss ---
            gt_text_token = labels[3][b] # [seq_len]
            if self.bidirectional_motion:
                text_token_len = len(gt_text_token) - 1 # bidirectional和ar之间的衔接会导致1个token的gap无法supervise，所以需要减去1
            else:
                text_token_len = len(gt_text_token)
            text_start_idx = status_end_idx
            text_end_idx   = text_start_idx + text_token_len
            text_hidden_states = hidden_states[b, text_start_idx:text_end_idx]
            text_logits = self.showo.lm_head(text_hidden_states)

            if self.keep_text:
                loss_text = F.cross_entropy(
                    text_logits.contiguous().view(-1, self.output_size),
                    gt_text_token.reshape(-1) if not self.bidirectional_motion else gt_text_token[1:].reshape(-1),
                    ignore_index=-100,
                )
            else:
                loss_text = torch.tensor(0.0, device=hidden_states.device)

            # --- object pose loss ---
            gt_object_status = labels[7][b]
            n_obj_token = 0
            loss_object_pose = torch.tensor(0.0, device=hidden_states.device)
            loss_obj_dynamic_pose = torch.tensor(0.0, device=hidden_states.device)

            if len(gt_object_status) > 0:
                n_obj_token = gt_object_status.shape[0]

                object_start_idx = text_end_idx
                object_end_idx   = object_start_idx + n_obj_token
                object_hidden_states = hidden_states[b, object_start_idx:object_end_idx]

                x_obj_mean_pred = self.object_mean_head(object_hidden_states)
                gt_object_status = gt_object_status.to(x_obj_mean_pred.dtype)

                if self.keep_scene:
                    loss_object_pose = F.smooth_l1_loss(
                        x_obj_mean_pred.float(), gt_object_status.float(), beta=0.5
                    ) * 20

                if self.dynamic_object:
                    object_hidden_states_expand = object_hidden_states[:, None].expand(-1, n_status_token * 4, -1)
                    motion_value = labels[9][b]
                    motion_value_expand = motion_value[None].expand(n_obj_token, -1, -1)
                    dynamic_diff_embeddings = torch.cat([object_hidden_states_expand, motion_value_expand], dim=2)

                    x_obj_dynamic_mean_pred = self.object_dynamic_mean_head(dynamic_diff_embeddings)
                    x_obj_dynamic_mean_pred = x_obj_dynamic_mean_pred.reshape(n_obj_token, n_status_token * 4, 12)
                    gt_obj_dynamic_status = labels[8][b]

                    x_obj_dynamic_mean_pred = x_obj_dynamic_mean_pred - x_obj_dynamic_mean_pred[:, 0:1]
                    gt_obj_dynamic_status   = gt_obj_dynamic_status   - gt_obj_dynamic_status[:, 0:1]

                    if self.keep_scene:
                        loss_obj_dynamic_pose = F.mse_loss(
                            x_obj_dynamic_mean_pred, gt_obj_dynamic_status
                        ) * 200

            # --- accumulate losses ---
            if not self.all_continuous_recon:
                loss_mean_batch          += loss_mean
                loss_std_batch           += loss_std
                loss_static_status_batch += loss_static_status
            loss_text_batch          += loss_text
            loss_object_pose_batch   += loss_object_pose
            if self.dynamic_object:
                loss_obj_dynamic_pose_batch += loss_obj_dynamic_pose

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
                if not self.bidirectional_motion:
                    text_accuracy    = masked_accuracy(text_logits[:-3-n_obj_token], gt_text_token[:-3-n_obj_token])
                else:
                    text_accuracy    = masked_accuracy(text_logits[:-3-n_obj_token], gt_text_token[1:-3-n_obj_token])
                obj_id_accuracy  = masked_accuracy(text_logits[-1-n_obj_token:-1], gt_text_token[-1-n_obj_token:-1])
                obj_pose_accuracy = (
                    torch.abs(x_obj_mean_pred - gt_object_status).mean()
                    if n_obj_token > 0
                    else torch.tensor(0.0, device=hidden_states.device)
                )
                obj_dynamic_pose_accuracy = (
                    torch.abs(x_obj_dynamic_mean_pred - gt_obj_dynamic_status).mean()
                    if self.dynamic_object
                    else torch.tensor(0.0, device=hidden_states.device)
                )

            mean_accuracy_batch           += mean_accuracy
            std_accuracy_batch            += std_accuracy
            static_accuracy_batch         += static_accuracy
            text_accuracy_batch           += text_accuracy
            obj_id_accuracy_batch         += obj_id_accuracy
            obj_pose_accuracy_batch       += obj_pose_accuracy
            if self.dynamic_object:
                obj_dynamic_pose_accuracy_batch += obj_dynamic_pose_accuracy

        # --- normalize by batch size ---
        if not self.all_continuous_recon:
            loss_mean_batch          = loss_mean_batch          / bs / 3.0
            loss_std_batch           = loss_std_batch           / bs / 3.0
            loss_static_status_batch = loss_static_status_batch / bs / 3.0
        loss_text_batch          /= bs
        loss_object_pose_batch   /= bs

        mean_accuracy_batch           /= bs
        std_accuracy_batch            /= bs
        static_accuracy_batch         /= bs
        text_accuracy_batch           /= bs
        obj_id_accuracy_batch         /= bs
        obj_pose_accuracy_batch       /= bs
        if self.dynamic_object:
            obj_dynamic_pose_accuracy_batch /= bs

        total_loss = (loss_mean_batch + loss_std_batch + loss_static_status_batch
                    + loss_text_batch + loss_object_pose_batch)
        if self.dynamic_object:
            loss_obj_dynamic_pose_batch /= bs
            total_loss = total_loss + loss_obj_dynamic_pose_batch
        if loss_recon_batch.item() > 0:
            loss_recon_batch = loss_recon_batch / bs
            total_loss = total_loss + loss_recon_batch
        loss_recon_out = loss_recon_batch 

        out = {
            'loss_mean':               loss_mean_batch,
            'loss_std':                loss_std_batch,
            'loss_static':             loss_static_status_batch,
            'loss_recon':              loss_recon_out,
            'loss_text':               loss_text_batch,
            'loss_object_pose':        loss_object_pose_batch,
            'loss_obj_dynamic_pose':   loss_obj_dynamic_pose_batch,
            'mean_accuracy':           mean_accuracy_batch,
            'std_accuracy':            std_accuracy_batch,
            'static_accuracy':         static_accuracy_batch,
            'text_accuracy':           text_accuracy_batch,
            'obj_id_accuracy':         obj_id_accuracy_batch,
            'obj_pose_accuracy':       obj_pose_accuracy_batch,
            'obj_dynamic_pose_accuracy': obj_dynamic_pose_accuracy_batch,
            'total_loss':              total_loss,
            'length':                  length,
        }
        if self.all_continuous_recon and self.keep_motion:
            out['x_recon'] = x_recon
        if self.partial_continuous_recon and self.keep_motion:
            out['x_mean_logits_all'] = x_mean_logits_all
            out['x_std_logits_all'] = x_std_logits_all
            out['x_value_logits_all'] = x_value_logits_all
            out['x_pose'] = x_pose
        return out