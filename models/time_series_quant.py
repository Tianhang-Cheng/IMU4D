import json
import numpy as np
import os
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotation2 import calculate_relative_rotation, convert_rotation
from .embed import PositionalEncoding, RoPEEncoding

"""
We copy the model definition here to resolve the file path issue
"""

def world_to_pelvis_local(R_pelvis, T_world_diff):
    """
    Convert absolute rotations/translations to pelvis-local coordinates.

    Args:
        R_pelvis (torch.Tensor): [B, N, 3, 3] pelvis rotation matrices.
        T_world_diff    (torch.Tensor): [B, N, 3] absolute joint translations.

    Returns:
        T_local_diff (torch.Tensor): [B, N, 3] local translations w.r.t pelvis.
    """

    R_pelvis_inv = R_pelvis.transpose(-1, -2)  # inverse since rotation

    T_local_diff = torch.matmul(R_pelvis_inv, T_world_diff.unsqueeze(-1)).squeeze(-1)
    return T_local_diff


def pelvis_local_to_world(R_pelvis, T_pelvis, R_local=None, T_local=None):
    """
    Convert pelvis-local rotations/translations back to absolute coordinates.

    Args:
        R_pelvis (torch.Tensor): [B, N, 3, 3] pelvis rotation matrices.
        T_pelvis (torch.Tensor): [B, N, 3] pelvis translations.
        R_local  (torch.Tensor): [B, N, 3, 3] local joint rotations.
        T_local  (torch.Tensor): [B, N, 3] local joint translations.

    Returns:
        R_abs (torch.Tensor): [B, N, 3, 3] absolute rotations.
        T_abs (torch.Tensor): [B, N, 3] absolute translations.
    """
    if R_local is not None:
        R_abs = torch.matmul(R_pelvis, R_local)
        return R_abs

    if T_local is not None:
        T_abs = torch.matmul(R_pelvis, T_local.unsqueeze(-1)).squeeze(-1) + T_pelvis
        return T_abs


def seed_everything(seed=42, deterministic=True):
    """
    Set all seeds to make results reproducible. Copied from Uni4D.
    
    Args:
        seed: Integer seed for reproducibility.
        deterministic: If True, ensures PyTorch operations are deterministic.
                    May impact performance due to deterministic algorithms.
    
    Note: Setting deterministic=True may significantly impact performance.
        Only use it when absolute reproducibility is required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # PyTorch backend
    if deterministic:
        print("Deterministic operations enabled.")
        # Configure backend for deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for deterministic operations
        os.environ['CUBLAS_WORKSPACE_CONFIG']= ":4096:8"
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Optional: Force PyTorch operations to be deterministic
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        print("Deterministic operations disabled.")
        # Better performance, but not fully deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def moving_average_filter(tensor, window_size, dim=1):
    """
    Applies a moving average filter to a tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        window_size (int): The size of the moving average window.
        dim (int): The dimension along which to apply the filter. Default is 1.

    Returns:
        torch.Tensor: The filtered tensor with the same shape as the input.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size should be an odd number for symmetric padding to maintain shape.")

    # Create a 1D convolution kernel for the moving average
    # Kernel shape for F.conv1d: (out_channels, in_channels/groups, kernel_size)
    # We have 1 input channel and 1 output channel for our 'virtual' conv1d operation
    kernel = torch.ones(1, 1, window_size, dtype=tensor.dtype, device=tensor.device) / window_size

    original_shape = tensor.shape
    batch_size, length, n_variables = original_shape

    if dim == 1: # Moving average along the length dimension
        # Reshape to (batch * n_variables, 1, length) for conv1d
        # (N, C_in, L_in) where N = batch * n_variables, C_in = 1, L_in = length
        reshaped_tensor = tensor.permute(0, 2, 1).reshape(-1, 1, length)
        padding = window_size // 2
        filtered_reshaped = F.conv1d(reshaped_tensor, kernel, padding=padding)
        # filtered_reshaped shape will be (batch * n_variables, 1, length)
        
        # Reshape back to original dimensions: (batch, n_variables, length) then (batch, length, n_variables)
        filtered_tensor = filtered_reshaped.reshape(batch_size, n_variables, length).permute(0, 2, 1)

    elif dim == 0: # Moving average along the batch dimension
        # Reshape to (n_variables * length, 1, batch_size)
        reshaped_tensor = tensor.permute(2, 1, 0).reshape(-1, 1, batch_size)
        padding = window_size // 2
        filtered_reshaped = F.conv1d(reshaped_tensor, kernel, padding=padding)
        
        # Reshape back to original dimensions
        filtered_tensor = filtered_reshaped.reshape(n_variables, length, batch_size).permute(2, 1, 0)

    elif dim == 2: # Moving average along the n_variable dimension
        # Reshape to (batch * length, 1, n_variables)
        reshaped_tensor = tensor.reshape(-1, 1, n_variables)
        padding = window_size // 2
        filtered_reshaped = F.conv1d(reshaped_tensor, kernel, padding=padding)
        
        # Reshape back to original dimensions
        filtered_tensor = filtered_reshaped.reshape(batch_size, length, n_variables)
    else:
        raise ValueError("Invalid dimension for filtering. Must be 0, 1, or 2.")

    # Assert to ensure the shape is preserved
    if filtered_tensor.shape != original_shape:
        raise RuntimeError(f"Shape changed after filtering! Original: {original_shape}, Filtered: {filtered_tensor.shape}")

    return filtered_tensor


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor):
        super(Encoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=3, padding=1)
            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 8:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 12:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = F.relu(x)

            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 16:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])

            x = self._conv_1(x)
            x = F.relu(x)

            x = self._conv_2(x)
            x = F.relu(x)

            x = self._conv_A(x)
            x = F.relu(x)

            x = self._conv_B(x)
            x = F.relu(x)

            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, compression_factor):
        super(Decoder, self).__init__()
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            # To get the correct shape back the kernel size has to be 5 not 4
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=5,
                                                    stride=3, padding=1)

            self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_4 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)

            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)

            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_B = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                    out_channels=1,
                                                    kernel_size=4,
                                                    stride=2, padding=1)

    def forward(self, inputs, compression_factor):
        if compression_factor == 4:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 8:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)

        elif compression_factor == 12:
            x = self._conv_1(inputs)
            x = self._residual_stack(x)

            x = self._conv_trans_2(x)
            x = F.relu(x)

            x = self._conv_trans_3(x)
            x = F.relu(x)

            x = self._conv_trans_4(x)

            return torch.squeeze(x)

        elif compression_factor == 16:
            x = self._conv_1(inputs)

            x = self._residual_stack(x)

            x = self._conv_trans_A(x)
            x = F.relu(x)

            x = self._conv_trans_B(x)
            x = F.relu(x)

            x = self._conv_trans_1(x)
            x = F.relu(x)

            x = self._conv_trans_2(x)

            return torch.squeeze(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        weight_type = self._embedding.weight.dtype
        encodings = encodings.to(weight_type)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings


class vqvae(nn.Module):
    def __init__(self, vqvae_config):
        super().__init__()
        num_hiddens = vqvae_config['block_hidden_size']
        num_residual_layers = vqvae_config['num_residual_layers']
        num_residual_hiddens = vqvae_config['res_hidden_size']
        embedding_dim = vqvae_config['embedding_dim']
        num_embeddings = vqvae_config['num_embeddings']
        commitment_cost = vqvae_config['commitment_cost']
        self.compression_factor = vqvae_config['compression_factor']

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.encoder = Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, self.compression_factor)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, self.compression_factor)

    def shared_eval(self, batch, mode):
        """
        batch: [bs*nvars, ntime]
        """
        if mode == 'train':
            raise NotImplementedError('Training not implemented in this script')
        if mode == 'val' or mode == 'test':
            with torch.no_grad():
                z = self.encoder(batch, self.compression_factor)# z: [bs*nvars, embedding_dim, n_chunks]
                vq_loss, quantized, perplexity, _, encoding_indices, _ = self.vq(z)

                data_recon = self.decoder(quantized, self.compression_factor) # [bs*nvars, ntime]
                loss = torch.tensor(0.0).to(batch.device)
                recon_error = torch.tensor(0.0).to(batch.device)
        return loss, vq_loss, recon_error, data_recon, encoding_indices


imu_meaning = {0: 'acc_x', 1: 'acc_y', 2: 'acc_z',
               3: 'gyro_x', 4: 'gyro_y', 5: 'gyro_z',
               6: 'mag_x', 7: 'mag_y', 8: 'mag_z',
               
               9: 'rot_x', 10: 'rot_y', 11: 'rot_z',
               12: 'x', 13: 'y', 14: 'z',

               15: 'rot_x', 16: 'rot_y', 17: 'rot_z',
               18: 'x', 19: 'y', 20: 'z',
               }

data_types = {'imu': 9, 'traj': 3}

class UniformQuantizer:
    """
    Uniform quantizer that works with PyTorch tensors and supports batch operations.
    """
    
    def __init__(self, min_val, max_val, num_bins=1024, device=None, apply_sqrt=False):
        """
        Initialize the quantizer.
        
        Args:
            min_val: Minimum value of the range
            max_val: Maximum value of the range
            num_bins: Number of bins (default 1024)
            device: PyTorch device (auto-detected if None)
        """
        if apply_sqrt:
            # 使用符号平方根处理跨零情况
            min_val = self._signed_sqrt(torch.tensor(float(min_val)))
            max_val = self._signed_sqrt(torch.tensor(float(max_val)))
        else:
            min_val = torch.tensor(float(min_val))
            max_val = torch.tensor(float(max_val))

        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.device = device
        self.apply_sqrt = apply_sqrt
        
        # Calculate bin width
        self.bin_width = (max_val - min_val) / num_bins
        
        # Create bin centers tensor
        bin_centers = torch.linspace(min_val + self.bin_width/2, max_val - self.bin_width/2, num_bins)
        if device is not None:
            bin_centers = bin_centers.to(device)
        self.bin_centers = bin_centers
        # self.bin_centers_embeddings = self.get_bin_center_embeddings(d=1)  # Default embedding dimension

        bin_centers_norm = ((bin_centers-self.min_val) / (self.max_val - self.min_val))  # Normalize to [0, 1]

        self.bin_centers_embeddings = bin_centers_norm

        # dim = 32
        # self.bin_centers_embeddings = nn.Parameter(torch.ones(num_bins, dim))
        # self.bin_centers_embeddings.data = bin_centers_norm.unsqueeze(1) * self.bin_centers_embeddings.data

        # import pdb; pdb.set_trace()
        # _=1


    def _signed_sqrt(self, x):
            """核心改进：向量化的符号平方根"""
            return torch.sign(x) * torch.sqrt(torch.abs(x))

    def _signed_pow2(self, x):
        """核心改进：向量化的符号平方反变换"""
        return torch.sign(x) * (x ** 2)


    @torch.no_grad()
    def quantize(self, values):
        """
        Quantize input tensor values to bins.
        
        Args:
            values: torch.Tensor of any shape
        
        Returns:
            bin_indices: torch.Tensor of bin indices (same shape as input)
            quantized_values: torch.Tensor of quantized values (same shape as input)
        """
        # Ensure computations are performed in float32
        values = values.to(torch.float32)

        x = self._signed_sqrt(values) if self.apply_sqrt else values
        
        # 2. 映射到索引
        # 使用 clamp 确保不会溢出索引范围
        indices = ((x - self.min_val) / self.bin_width).long()
        indices = torch.clamp(indices, 0, self.num_bins - 1)
        
        # 3. 反量化得到值
        quantized_values = self.decode(indices)
        
        return indices, quantized_values

    @torch.no_grad()
    def decode(self, bin_indices):
        """
        Decode bin indices back to original values.
        
        Args:
            bin_indices: torch.Tensor of bin indices (same shape as input)
        
        Returns:
            decoded_values: torch.Tensor of decoded values (same shape as input)
        """
        # Handle device placement
        # if self.device is None:
        #     self.device = bin_indices.device
        #     self.bin_centers = self.bin_centers.to(self.device)

        # 1. 查找中心点
        x = self.bin_centers[bin_indices]
        
        # 2. 逆变换
        if self.apply_sqrt:
            x = self._signed_pow2(x)
        
        return x

class DiscreteQuantizer(nn.Module):
    def __init__(self, 
                 accumulate: bool, accumulate_orient: bool, accumulate_pose: bool, local_coordinate: bool,
                 compression_rate=4, n_bins=8192, d_model=2048,
                 large_model=False, global_model=False, use_6d=False,
                 use_rope_embed: bool = False,
                 totem_folder='/home/tianhang/code/imu-human-mllm/third_party/TOTEM'):
        super(DiscreteQuantizer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compression_rate = compression_rate
        self.n_bins = n_bins
        self.totem_folder = totem_folder
        self.accumulate = accumulate
        self.accumulate_orient = accumulate_orient
        self.accumulate_pose = accumulate_pose
        self.local_coordinate = local_coordinate
        self.large_model = large_model
        self.global_model = global_model
        self.use_6d = use_6d
        self.use_rope_embed = use_rope_embed
        self.rot_var_num = 6 if use_6d else 3
        self.quant = self.create_quantizer()

        if use_rope_embed:
            self.time_embedder = RoPEEncoding(d_model=d_model, max_len=1000)
        else:
            self.time_embedder = PositionalEncoding(d_model=d_model, max_len=1000)

        assert not large_model
        assert accumulate
        assert accumulate_orient
        assert not accumulate_pose

    def create_quantizer(self):

        # Get config file
        assert self.compression_rate in [4, 8, 12, 16]
        if self.large_model:
            config_file = os.path.join(self.totem_folder, f'configs/train_{self.compression_rate}_large.json')
        elif self.global_model:
            config_file = os.path.join(self.totem_folder, f'configs/train_{self.compression_rate}_global.json')
        else:
            config_file = os.path.join(self.totem_folder, f'configs/train_{self.compression_rate}.json')

        # Load JSON config file
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Create summary dictionary
        summary = {}
        general_seed = 42
        summary['general_seed'] = general_seed
        summary['data initialization location'] = self.device
        summary['device'] = self.device  # add the cpu/gpu to the summary
        seed_everything(general_seed, deterministic=False)

        # Setup model
        vqvae_config = config['vqvae_config']
        model = vqvae(vqvae_config)

        if self.large_model:
            weight = torch.load(os.path.join(self.totem_folder, f'pretrained_weight/{self.compression_rate}_large_final.pth'), map_location='cpu', weights_only=True)
        elif self.global_model:
            weight = torch.load(os.path.join(self.totem_folder, f'pretrained_weight/{self.compression_rate}_global_final.pth'), map_location='cpu', weights_only=True)
        else:
            weight = torch.load(os.path.join(self.totem_folder, f'pretrained_weight/{self.compression_rate}_final.pth'), map_location='cpu', weights_only=True)
        
        model.load_state_dict(weight)
        model = model.to(self.device)
        self.model = model
        del weight

        # The mean and std range for time series data, including imu, human traj and human pose
        # The normalized window size is 4
        mean_range = (-3.0, 3.0)
        std_range = (0, 0.5)
        # mean_range = (-38.4, 37.6)
        # std_range = (0, 75.06)
        self.mean_quantizer = UniformQuantizer(mean_range[0], mean_range[1], device=self.device, num_bins=self.n_bins, apply_sqrt=True)
        self.std_quantizer = UniformQuantizer(std_range[0], std_range[1], device=self.device, num_bins=self.n_bins, apply_sqrt=True)

    def forward(self, batch_x, normalization_window_size=4, viz=False, **kwargs):
        """
        batch_x: [bs, ntime, nvars]
        normalization_window_size: int, the size of the window to use to normalize the data. can be different from the compression rate.
        return_continous: bool, if True, return the continuous values of the data, and discrete indices of mean and std
        return_id: bool, if True, return the indices of the quantized value, and discrete indices of mean and std
        """

        if isinstance(batch_x, np.ndarray):
            x = copy.deepcopy(batch_x)
        elif isinstance(batch_x, torch.Tensor):
            x = batch_x.clone()

        if len(x.shape) == 2:
            x = x[None]
            flag = True
        else:
            flag = False

        assert len(x.shape) == 3, f'Invalid data shape: {x.shape}'
        assert 'data_type' in kwargs or kwargs.get('no_preprocess', False), f'Invalid data type: {kwargs.get("data_type", "unknown")}'

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        bs, ntime, nvars = x.shape

        assert normalization_window_size > 0, f'Invalid normalization window size: {normalization_window_size}'
        assert normalization_window_size % self.compression_rate == 0, f'window size {normalization_window_size} should be divisible by compression rate {self.compression_rate}'
        
        weight_type = self.get_weight_type()

        pad_len = 0
        if ntime % self.compression_rate != 0:
            # zero padding
            pad_len = self.compression_rate - ntime % self.compression_rate
            pad_value = np.zeros((bs, pad_len, nvars)) # + batch_x[:, -1:]
            x = np.concatenate((x, pad_value), axis=1)
            ntime = x.shape[1]

        pad_len2 = 0
        if ntime % normalization_window_size != 0:
            # zero padding
            pad_len2 = normalization_window_size - ntime % normalization_window_size
            pad_value = np.zeros((bs, pad_len2, nvars)) # + batch_x[:, -1:]
            x = np.concatenate((x, pad_value), axis=1)
            ntime = x.shape[1]

        n_windows = ntime // normalization_window_size
        batch_x_window = torch.from_numpy(x).to(weight_type).to(self.device) # [bs, ntime, nvars]

        # apply moving average filter to smooth the input
        # if kwargs.get('smooth_input', False):            
        #     batch_x_window = moving_average_filter(batch_x_window, window_size=5, dim=1)
        
        if kwargs['data_type'] == 'imu':
            # no preprocessing
            batch_x_window = batch_x_window.reshape(bs, n_windows, normalization_window_size, nvars)

        elif kwargs['data_type'] == 'traj':
            if self.accumulate:
                # diff the traj
                # batch_x_window = batch_x_window.reshape(bs, n_windows, normalization_window_size, nvars)
                # batch_x_window[:, 1:, :, 0:3] = batch_x_window[:, 1:, :, 0:3] - batch_x_window[:, 0:-1, :, 0:3].mean(dim=2, keepdim=True) # FIXME: old offset is not correct
                batch_x_window[:, 1:] = batch_x_window[:, 1:] - batch_x_window[:, 0:-1] # [bs, ntime, nvars], world diff
                if False:
                    raise ValueError('not tested, seems not stable during training')
                    # project to local coordinate
                    ref_R = kwargs['ref_R'].to(self.device).clone() # [ntime, 3, 3]
                    ref_R = torch.cat([torch.eye(3).to(ref_R.dtype)[None].to(self.device), ref_R[1:]], dim=0)[None] # the first frame is identity, [1, ntime, 3, 3]
                    batch_x_window = world_to_pelvis_local(R_pelvis=ref_R, T_world_diff=batch_x_window) # [bs, ntime, 3], local diff
            batch_x_window = batch_x_window.reshape(bs, n_windows, normalization_window_size, nvars)
        
        elif kwargs['data_type'] == 'orient':
            if self.accumulate_orient:
                # diff the orientation
                batch_x_window = batch_x_window.reshape(bs, ntime, nvars//self.rot_var_num, -1)
                batch_x_window = calculate_relative_rotation(batch_x_window, src_rep='6d' if self.use_6d else 'aa', tgt_rep='mat') # [bs, ntime, n_joints, 3, 3]
                batch_x_window = batch_x_window.permute(0, 2, 1, 3, 4).reshape(-1, ntime, 3, 3) # [bs*n_joints, ntime, 3, 3]
                batch_x_window = convert_rotation(batch_x_window, src_rep='mat', tgt_rep='6d') # [bs*n_joints, ntime, 6]
                batch_x_window = batch_x_window.reshape(bs, -1, ntime, self.rot_var_num).permute(0, 2, 1, 3).reshape(bs, ntime, -1) # [bs, ntime, nvars]

            if not self.use_6d:
                batch_x_window = batch_x_window / np.pi # normalize to [-1, 1]
            batch_x_window = batch_x_window.reshape(bs, n_windows, normalization_window_size, nvars)
        
        elif kwargs['data_type'] == 'pose':
            if self.accumulate_pose:
                batch_x_window = batch_x_window.reshape(bs, ntime, nvars//self.rot_var_num, -1)
                batch_x_window = calculate_relative_rotation(batch_x_window, src_rep='6d' if self.use_6d else 'aa') # [bs, ntime, n_joints, 3, 3]
                batch_x_window = batch_x_window.permute(0, 2, 1, 3, 4).reshape(-1, ntime, 3, 3) # [bs*n_joints, ntime, 3, 3]
                batch_x_window = convert_rotation(batch_x_window, src_rep='mat', tgt_rep='6d') # [bs*n_joints, ntime, 6]
                batch_x_window = batch_x_window.reshape(bs, -1, ntime, self.rot_var_num).permute(0, 2, 1, 3).reshape(bs, ntime, -1) # [bs, ntime, nvars]

            if not self.use_6d:
                batch_x_window = batch_x_window / np.pi # normalize to [-1, 1]
            batch_x_window = batch_x_window.reshape(bs, n_windows, normalization_window_size, nvars)
        
        else:
            raise ValueError(f'Invalid data type: {kwargs["data_type"]}. Supported types are: imu, traj, orient.')
        
        if self.global_model:
            # placeholder for global model, no normalization
            gt_mean = torch.zeros((bs, n_windows, 1, nvars), device=self.device) # [bs, n_windows, 1, nvars]
            gt_std = torch.ones((bs, n_windows, 1, nvars), device=self.device) # [bs, n_windows, 1, nvars]
        else:
            gt_mean = batch_x_window.mean(dim=2, keepdim=True)
            gt_std = batch_x_window.std(dim=2, keepdim=True, unbiased=False) # [bs, n_windows, 1, nvars]
            batch_x_window = (batch_x_window - gt_mean) / (gt_std.clamp(min=1e-5)) # [bs, n_windows, normalization_window_size, nvars]

        # if kwargs.get('debug', False):
        #     gt_raw = {'batch_x_window': batch_x_window.clone().detach().cpu().numpy(),
        #               'batch_x_window_raw': batch_x_window_raw.clone().detach().cpu().numpy(),
        #               'gt_mean': gt_mean.clone().detach().cpu().numpy(),
        #               'gt_std': gt_std.clone().detach().cpu().numpy()}
        #     # import pdb; pdb.set_trace()
        #     np.savez('/scratch/benk/tcheng1/code/imu-human-mllm/third_party/Show-o/temp/gt_raw.npz', **gt_raw)

        batch_x_cuda = batch_x_window.to(weight_type).to(self.device)

        batch_x_cuda = batch_x_cuda.permute(0, 3, 1, 2) # [bs, nvars, n_windows, normalization_window_size]

        batch_x_cuda = batch_x_cuda.reshape(-1, normalization_window_size) # [bs*nvars*n_windows, normalization_window_size]
            
        if gt_mean is not None:
            mean_indices, quantized_mean = self.mean_quantizer.quantize(gt_mean)
            std_indices, quantized_std = self.std_quantizer.quantize(gt_std)

        # if kwargs.get('debug', False):
        #     import pdb; pdb.set_trace()
        #     self.mean_quantizer.quantize(torch.tensor([1.0]))
        #     self.mean_quantizer.quantize(torch.tensor([0.01]))
        #     print((quantized_std.detach().cpu() - gt_std).abs().max())
        #     print((quantized_mean.detach().cpu() - gt_mean).abs().max())
        #     (quantized_mean - gt_mean)[0,0]
        #     _=1

        loss, vq_loss, recon_error, x_recon, encoding_indices = self.model.shared_eval(batch_x_cuda, mode='val')
    
        x_recon = x_recon.reshape(bs, nvars, ntime).permute(0, 2, 1) # [bs, ntime, nvars]
        x_recon = x_recon.reshape(bs, -1, normalization_window_size, nvars)
        pred_val = x_recon * quantized_std + quantized_mean
        pred_val = pred_val.reshape(bs, ntime, nvars)
        x_recon_denorm = pred_val

        # recover to debug
        # if kwargs['data_type'] == 'orient':
        #     _x = pred_val.clone() * np.pi
        #     _x = _x.reshape(bs, ntime, 1, 3)
        #     _x = recover_absolute_rotation(_x)
        #     _x = _x.detach().cpu().numpy()

        #     import matplotlib.pyplot as plt
        #     fig = plt.figure(figsize=(12, 6))
        #     plt.plot(_x[0, :, 0, 0], label='pred x', color='r')
        #     plt.plot(_x[0, :, 0, 1], label='pred y', color='g')
        #     plt.plot(_x[0, :, 0, 2], label='pred z', color='b')
        #     plt.plot(batch_x[0, :, 0], label='gt x', color='r', linestyle='--')
        #     plt.plot(batch_x[0, :, 1], label='gt y', color='g', linestyle='--')
        #     plt.plot(batch_x[0, :, 2], label='gt z', color='b', linestyle='--')
        #     plt.legend()
        #     plt.show()

        #     _=1
        
        # if kwargs['data_type'] == 'traj':

        #     _x = pred_val.clone()
        #     _x = _x.reshape(bs, -1, normalization_window_size, nvars)
        #     _x[:,1:] = _x[:, 1:] +  torch.cumsum(_x[:, :-1, :], dim=1).mean(dim=2, keepdim=True)
        #     _x = _x.detach().cpu().numpy()
        #     _x = _x.reshape(bs, ntime, 1, 3)

        #     import matplotlib.pyplot as plt
        #     fig = plt.figure(figsize=(12, 6))
        #     plt.plot(_x[0, :, 0, 0], label='pred x', color='r')
        #     plt.plot(_x[0, :, 0, 1], label='pred y', color='g')
        #     plt.plot(_x[0, :, 0, 2], label='pred z', color='b')
        #     plt.plot(batch_x[0, :, 0], label='gt x', color='r', linestyle='--')
        #     plt.plot(batch_x[0, :, 1], label='gt y', color='g', linestyle='--')
        #     plt.plot(batch_x[0, :, 2], label='gt z', color='b', linestyle='--')
        #     plt.legend()
        #     plt.show()

        #     _=1

        if pad_len + pad_len2 > 0:
            x_recon_denorm = x_recon_denorm[:, :ntime-(pad_len+pad_len2), :]
            x = x[:, :ntime-(pad_len+pad_len2), :]
        
        if viz:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nvars, 1, figsize=(nvars*9, 12))
            fig.suptitle('Original vs Predicted for Each Variable', fontsize=16)
            for j, var_idx in enumerate(range(nvars)):
                ax = axes[j]

                a = x[0, :, var_idx]
                b = x_recon_denorm[0, :, var_idx].detach().cpu().numpy()

                ax.plot(a, label='original')
                ax.plot(b, label='predicted')
                # ax.set_title(f'Variable {var_idx}')
                ax.set_title(f'{imu_meaning[var_idx]}')

                ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle
            plt.show()

        if flag:
            x_recon_denorm = x_recon_denorm[0]

        # if return_continous:
        #     # return the continuous values
        #     # batch_x_window = batch_x_window.reshape(bs, n_windows, normalization_window_size, nvars)
        #     mean_indices = np.repeat(mean_indices.reshape(bs, n_windows, nvars), axis=1, repeats=normalization_window_size // self.compression_rate)
        #     std_indices = np.repeat(std_indices.reshape(bs, n_windows, nvars), axis=1, repeats=normalization_window_size // self.compression_rate)

        #     return x_recon_denorm, batch_x_window, np.concatenate((mean_indices, std_indices), axis=-1) # [bs, ntime//compression_rate, nvars+6]

        static_indices = encoding_indices.reshape(bs, nvars, n_windows).permute(0, 2, 1) # [bs, n_windows, nvars]
        # a window corresponds to (window_size // compression_rate) tokens
        # so we need to repeat the indices for each token in the window
        # mean_indices = np.repeat(mean_indices.reshape(bs, ntime//window_size, 1, nvars), axis=2, repeats=window_size // compression_rate)
        # mean_indices = mean_indices.reshape(bs, ntime//compression_rate, nvars)
        # std_indices = np.repeat(std_indices.reshape(bs, ntime//window_size, 1, nvars), axis=2, repeats=window_size // compression_rate)
        # std_indices = std_indices.reshape(bs, ntime//compression_rate, nvars)

        # indices = np.stack((encoding_indices, mean_indices, std_indices), axis=1) # [bs, 3, ntime//compression_rate, nvars]

        # [var, mean, std] # FIXME: use torch.repeat instead of numpy.repeat
        # mean_indices = np.repeat(mean_indices.reshape(bs, n_windows, nvars), axis=1, repeats=normalization_window_size // self.compression_rate)
        # std_indices = np.repeat(std_indices.reshape(bs, n_windows, nvars), axis=1, repeats=normalization_window_size // self.compression_rate)
        # mean_indices = mean_indices.reshape(bs, n_windows, nvars).repeat(dim=1, repeats=normalization_window_size // self.compression_rate)
        # std_indices = std_indices.reshape(bs, n_windows, nvars).repeat(dim=1, repeats= normalization_window_size // self.compression_rate)
        repeats = normalization_window_size // self.compression_rate
        mean_indices = mean_indices.reshape(bs, n_windows, nvars).repeat(1, repeats, 1)
        std_indices = std_indices.reshape(bs, n_windows, nvars).repeat(1, repeats, 1)

        # indices_dict = {
        #     'mean_indices': mean_indices,
        #     'std_indices': std_indices,
        #     'encoding_indices': encoding_indices
        # }

        # if kwargs.get('perturb_every_token', False):
        #     # perturb every token by shift with probability p
        #     p = kwargs['perturb_every_token_p']
        #     shift = kwargs['perturb_every_token_shift']
        #     mean_indices = mean_indices + (torch.rand(bs, n_windows, nvars) < p).int().to(self.device) * shift
        #     mean_indices = torch.clip(mean_indices, 0, self.n_bins - 1)
        #     std_indices = std_indices + (torch.rand(bs, n_windows, nvars) < p).int().to(self.device) * shift
        #     std_indices = torch.clip(std_indices, 0, self.n_bins - 1)
        #     static_indices = static_indices + (torch.rand(bs, n_windows, nvars) < p).int().to(self.device) * shift
        #     static_indices = torch.clip(static_indices, 0, self.model.vq._num_embeddings - 1)

        #     quantized_mean = self.mean_quantizer.decode(mean_indices).reshape(bs, n_windows, 1, nvars)
        #     quantized_std = self.std_quantizer.decode(std_indices).reshape(bs, n_windows, 1, nvars)

        #     static_embeddings = self.static_embedder(static_indices.reshape(-1))
        #     quantized_static = self.model.decoder(static_embeddings[..., None], self.compression_rate)
        #     quantized_static = quantized_static.reshape(bs, n_windows, nvars, self.compression_rate)
        #     quantized_static = quantized_static.permute(0, 1, 3, 2) # [bs, n_windows, compression_rate, nvars]

        #     x_recon_denorm = quantized_static * quantized_std + quantized_mean
        #     x_recon_denorm = x_recon_denorm.reshape(bs, ntime, nvars).detach().cpu()
        #     if flag:
        #         x_recon_denorm = x_recon_denorm[0]

        if kwargs.get('perturb_every_token', False):
            # perturb every token by shift with probability p
            p = kwargs['perturb_every_token_p']

            mask = torch.rand(bs, n_windows, nvars).to(self.device) < p
            random_indices = torch.randint(0, self.n_bins, (bs, n_windows, nvars)).to(self.device)
            mean_indices = torch.where(mask, random_indices, mean_indices)
            
            mask = torch.rand(bs, n_windows, nvars).to(self.device) < p
            random_indices = torch.randint(0, self.n_bins, (bs, n_windows, nvars)).to(self.device)
            std_indices = torch.where(mask, random_indices, std_indices)

            mask = torch.rand(bs, n_windows, nvars).to(self.device) < p
            random_indices = torch.randint(0, self.model.vq._num_embeddings, (bs, n_windows, nvars)).to(self.device)
            static_indices = torch.where(mask, random_indices, static_indices)

            quantized_mean = self.mean_quantizer.decode(mean_indices).reshape(bs, n_windows, 1, nvars)
            quantized_std = self.std_quantizer.decode(std_indices).reshape(bs, n_windows, 1, nvars)

            static_embeddings = self.static_embedder(static_indices.reshape(-1))
            quantized_static = self.model.decoder(static_embeddings[..., None], self.compression_rate)
            quantized_static = quantized_static.reshape(bs, n_windows, nvars, self.compression_rate)
            quantized_static = quantized_static.permute(0, 1, 3, 2) # [bs, n_windows, compression_rate, nvars]

            x_recon_denorm = quantized_static * quantized_std + quantized_mean
            x_recon_denorm = x_recon_denorm.reshape(bs, ntime, nvars).detach().cpu()
            if flag:
                x_recon_denorm = x_recon_denorm[0]

        if kwargs.get('return_seperate_indices', False):
            mean_indices = mean_indices.to(self.device)
            std_indices = std_indices.to(self.device)
            static_indices = static_indices.to(self.device)
            # return (x_recon_denorm, quantized_mean, quantized_std, batch_x_cuda), static_indices, mean_indices, std_indices
            return x_recon_denorm, static_indices, mean_indices, std_indices, gt_mean, gt_std
        if kwargs.get('only_quantize_static', False):
            gt_mean = gt_mean[:, :, 0].unsqueeze(-1).to(self.device)
            gt_std = gt_std[:, :, 0].unsqueeze(-1).to(self.device)
            static_indices = static_indices.to(self.device)
            # return (x_recon_denorm, quantized_mean, quantized_std, batch_x_cuda), static_indices, mean_indices, std_indices
            return x_recon_denorm, static_indices, gt_mean, gt_std            

        dynamic_indices = np.concatenate((mean_indices, std_indices), axis=-1) # [bs, ntime//compression_rate, nvars*2]
        dynamic_indices = torch.from_numpy(dynamic_indices).to(self.device)
        return x_recon_denorm, static_indices, dynamic_indices

    def mean_embedder(self, idx):
        return self.mean_quantizer.bin_centers_embeddings[idx]
    
    def std_embedder(self, idx):
        return self.std_quantizer.bin_centers_embeddings[idx]

    def static_embedder(self, idx: torch.Tensor):
        # nn.Embedding
        return self.model.vq._embedding(idx)

    def get_dynamic_vocab_size(self):
        return self.n_bins
    
    def get_static_vocab_size(self):
        return self.model.vq._num_embeddings

    def get_weight_type(self):
        return self.model.encoder._conv_1.weight.dtype