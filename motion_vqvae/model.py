import argparse
import json
import numpy as np
import os
import random
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, input_dict, optimizer=None, mode='val'):

        batch_gt = input_dict['batch'] # [bs, n_var, n_time]
        batch_norm = input_dict['batch_norm'] # [bs, n_var, n_time]
        mean = input_dict['mean'] # [bs, n_var, n_time]
        std = input_dict['std'] # [bs, n_var, n_time]

        if mode == 'train':
            assert optimizer is not None, "Optimizer must be provided for training mode"
            optimizer.zero_grad() 

            z = self.encoder(batch_norm, self.compression_factor) # [bs*n_var, n_time] -> [bs*n_var, dim, n_window]
            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
            data_recon = self.decoder(quantized, self.compression_factor)
            
            # import pdb; pdb.set_trace()
            recon_error = F.mse_loss(data_recon * std + mean, batch_gt)
            loss = recon_error + vq_loss
            
            loss.backward()
            optimizer.step()

        if mode == 'val' or mode == 'test':
            with torch.no_grad():
                z = self.encoder(batch_norm, self.compression_factor)
                vq_loss, quantized, perplexity, _, encoding_indices, _ = self.vq(z)
                data_recon = self.decoder(quantized, self.compression_factor)
                data_recon = data_recon * std + mean
                recon_error = torch.tensor(0.0).to(batch_gt.device)
                loss = torch.tensor(0.0).to(batch_gt.device)

        return loss, vq_loss, recon_error, data_recon, encoding_indices, perplexity


imu_meaning = {0: 'acc_x', 1: 'acc_y', 2: 'acc_z',
               3: 'gyro_x', 4: 'gyro_y', 5: 'gyro_z',
               6: 'mag_x', 7: 'mag_y', 8: 'mag_z',
               
               9: 'rot_x', 10: 'rot_y', 11: 'rot_z',
               12: 'x', 13: 'y', 14: 'z',

               15: 'rot_x', 16: 'rot_y', 17: 'rot_z',
               18: 'x', 19: 'y', 20: 'z',
               }

data_types = {'imu': 9, 'traj': 3}