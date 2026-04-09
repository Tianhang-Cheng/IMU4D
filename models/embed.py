
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def count_parameters(model: torch.nn.Module):
    x = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {x}")

class IMUAggregator(nn.Module):
    '''
    Temporal aggregation of IMU data using pre-embedded static and dynamic features.
    '''
    def __init__(self, nvar, d_model, chunk_size, embed_dim):
        super(IMUAggregator, self).__init__()

        self.chunk_size = chunk_size
        self.nvar = nvar
        self.embed_dim = embed_dim

        input_dim = embed_dim * 3    # value + statistic mean + statistic std
        self.imu_projection_1 = nn.Conv1d(in_channels=nvar, out_channels=d_model, kernel_size=input_dim)
        self.imu_projection_2 = nn.Linear(d_model, d_model)  # imu projection

    def forward(self, x_static, x_dynamic):
        """
        x_static: Tensor, shape = [n_chunk, nvar, embed_dim]
        x_dynamic: Tensor, shape = [n_chunk, nvar, embed_dim * 2] (mean & std concatenated)
        
        Returns: Tensor, shape = [n_chunk, d_model]
        """
        x = torch.cat((x_static, x_dynamic), dim=-1)  # Shape: [n_chunk, nvar, embed_dim * 3]
        
        x = self.imu_projection_1(x).squeeze(2) # [n_chunck, 1, d_model]
        x = nn.GELU()(x)
        x = self.imu_projection_2(x)  # [n_chunck, d_model]

        return x

class SingleSignalAggregator(nn.Module):
    def __init__(self):
        super().__init__()

        self.dynamic_embedder = nn.Linear(2, 64)  # Embed mean and std concatenated
        self.signal_projection_1 = nn.Linear(128, 128)  # Project to a common dimension
        self.signal_projection_2 = nn.Linear(128, 128)  # Additional projection for better representation
    
    def forward(self, x_static, x_dynamic):
        """
        x_static: Tensor, shape = [n_chunk, nvar, embed_dim]
        x_dynamic: Tensor, shape = [n_chunk, nvar, 2] (mean & std concatenated)
        """
        x_dynamic = self.dynamic_embedder(x_dynamic)
        x = torch.cat((x_static, x_dynamic), dim=-1)  # Shape: [n_chunk, nvar, 128]
        x = self.signal_projection_1(x)  # Project to [n_chunk, nvar, 128]
        x = nn.GELU()(x)
        x = self.signal_projection_2(x)  # Additional projection to [n_chunk, nvar, 128]
        x = nn.GELU()(x)
        return x


class MultiSignalAggregator(nn.Module):
    '''
    Temporal aggregation of IMU data using pre-embedded static and dynamic features.
    '''
    def __init__(self, nvar, d_model, chunk_size, embed_dim, **kwargs):
        super(MultiSignalAggregator, self).__init__()

        self.chunk_size = chunk_size
        self.nvar = nvar
        self.embed_dim = embed_dim

        input_dim = embed_dim * 2    # value + statistic mean + statistic std
        self.imu_projection_1 = nn.Conv1d(in_channels=nvar, out_channels=d_model, kernel_size=input_dim) # dim*2048*9
        self.imu_projection_2 = nn.Linear(d_model, d_model)  # imu projection

        assert 'single_aggregator' in kwargs, "single single aggregator must be provided"
        self.single_aggregator = kwargs['single_aggregator']

    def forward(self, x_static, x_dynamic):
        """
        x_static: Tensor, shape = [n_chunk, nvar, embed_dim]
        x_dynamic: Tensor, shape = [n_chunk, nvar, 2] (mean & std concatenated)
        
        Returns: Tensor, shape = [n_chunk, d_model]
        """
        x = self.single_aggregator(x_static, x_dynamic)  # Apply signal aggregator to get [n_chunk, nvar, 128]
        
        # import pdb; pdb.set_trace()
        x = self.imu_projection_1(x).squeeze(2) # [n_chunck, 1, d_model]
        x = nn.GELU()(x)
        x = self.imu_projection_2(x)  # [n_chunck, d_model]
        return x

class IMUAggregator2(nn.Module):
    '''
    Temporal aggregation of IMU data using pre-embedded static and dynamic features.
    '''
    def __init__(self, nvar, d_model, chunk_size, embed_dim, **kwargs):
        super(IMUAggregator2, self).__init__()

        self.chunk_size = chunk_size
        self.nvar = nvar
        self.embed_dim = embed_dim

        input_dim = embed_dim + 2    # value + statistic mean + statistic std
        self.imu_projection_1 = nn.Conv1d(in_channels=nvar, out_channels=d_model, kernel_size=input_dim) # 64*2048*9
        self.imu_projection_2 = nn.Linear(d_model, d_model)  # imu projection
        self.imu_projection_3 = nn.Linear(d_model, d_model)  # additional projection for better representation

        # Initialize weights
        nn.init.xavier_uniform_(self.imu_projection_1.weight)
        nn.init.xavier_uniform_(self.imu_projection_2.weight)
        nn.init.xavier_uniform_(self.imu_projection_3.weight)

    def forward(self, x_static, x_dynamic):
        """
        x_static: Tensor, shape = [n_chunk, nvar, embed_dim]
        x_dynamic: Tensor, shape = [n_chunk, nvar, embed_dim * 2] (mean & std concatenated)
        
        Returns: Tensor, shape = [n_chunk, d_model]
        """
        x = torch.cat((x_static, x_dynamic), dim=-1)  # Shape: [n_chunk, nvar, embed_dim * 3]
        
        # import pdb; pdb.set_trace()
        x = self.imu_projection_1(x).squeeze(2) # [n_chunck, 1, d_model]
        x = nn.GELU()(x)
        x = self.imu_projection_2(x)  # [n_chunck, d_model]
        x = nn.GELU()(x)
        x = self.imu_projection_3(x)

        return x


class IMUAggregator3(nn.Module):
    '''
    Temporal aggregation of IMU data using pre-embedded static and dynamic features.
    '''
    def __init__(self, nvar, d_model,  **kwargs):
        super(IMUAggregator3, self).__init__()

        self.nvar = nvar

        self.imu_projection_0 = nn.Linear(nvar, d_model)
        self.imu_projection_2 = nn.Linear(d_model, d_model)  # imu projection
        self.imu_projection_3 = nn.Linear(d_model, d_model)  # additional projection for better representation

        # Initialize weights
        nn.init.xavier_uniform_(self.imu_projection_0.weight)
        nn.init.xavier_uniform_(self.imu_projection_2.weight)
        nn.init.xavier_uniform_(self.imu_projection_3.weight)

    def forward(self, x):
        """
        x: Tensor, shape = [n_chunk, nvar]
        
        Returns: Tensor, shape = [n_chunk, d_model]
        """
        # import pdb; pdb.set_trace()
        x = self.imu_projection_0(x)
        x = nn.GELU()(x)
        x = self.imu_projection_2(x)
        x = nn.GELU()(x)
        x = self.imu_projection_3(x)

        return x
    


class IMUAggregator4(nn.Module):
    '''
    Temporal aggregation of IMU data using pre-embedded static and dynamic features.
    '''
    def __init__(self, nvar, d_model, **kwargs):
        super(IMUAggregator4, self).__init__()

        self.nvar = nvar

        self.imu_projection_1 = nn.Linear(self.nvar, d_model)
        self.imu_projection_2 = nn.Linear(d_model, d_model)  # imu projection
        self.imu_projection_3 = nn.Linear(d_model, d_model)  # additional projection for better representation

        # Initialize weights
        nn.init.xavier_uniform_(self.imu_projection_1.weight)
        nn.init.xavier_uniform_(self.imu_projection_2.weight)
        nn.init.xavier_uniform_(self.imu_projection_3.weight)

    def forward(self, x, allow_projection=False):
        """
        x: Tensor, shape = [n_chunk, n]
        time: Tensor, shape = [n_chunk, 1]

        Returns: Tensor, shape = [n_chunk, d_model]
        """
        # import pdb; pdb.set_trace()
        if allow_projection:
            # dtype transform to model dtype
            x = x.to(self.imu_projection_1.weight.dtype)
        
        x = self.imu_projection_1(x)
        x = nn.GELU()(x)
        x = self.imu_projection_2(x)
        x = nn.GELU()(x)
        x = self.imu_projection_3(x)

        return x

class SignalAggregator(nn.Module):
    '''
    Temporal aggregation of IMU data using pre-embedded static and dynamic features.
    '''
    def __init__(self, nvar, d_model, chunk_size, embed_dim, **kwargs):
        super(SignalAggregator, self).__init__()

        self.chunk_size = chunk_size
        self.nvar = nvar
        self.embed_dim = embed_dim

        input_dim = embed_dim + 2 # mean value + statistic mean + statistic std
        self.imu_projection_1 = nn.Conv1d(in_channels=nvar, out_channels=d_model, kernel_size=input_dim) 
        self.imu_projection_2 = nn.Linear(d_model, d_model)  # imu projection
        self.imu_projection_3 = nn.Linear(d_model, d_model)  # additional projection for better representation

        # Initialize weights
        nn.init.xavier_uniform_(self.imu_projection_1.weight)
        nn.init.xavier_uniform_(self.imu_projection_2.weight)
        nn.init.xavier_uniform_(self.imu_projection_3.weight)

    def forward(self, x_static, x_dynamic):
        """
        x_static: Tensor, shape = [n_chunk, nvar, embed_dim]
        x_dynamic: Tensor, shape = [n_chunk, nvar, 2] (mean & std concatenated)

        Returns: Tensor, shape = [n_chunk, d_model]
        """
        # x = torch.cat((x_static, x_dynamic), dim=-1)  # Shape: [n_chunk, nvar, embed_dim + 3]
        x = torch.cat((x_static, x_dynamic), dim=-1)  # Shape: [n_chunk, nvar, embed_dim + 3]

        # import pdb; pdb.set_trace()
        x = self.imu_projection_1(x).squeeze(2) # [n_chunck, 1, d_model]
        x = nn.GELU()(x)
        x = self.imu_projection_2(x)  # [n_chunck, d_model]
        x = nn.GELU()(x)
        x = self.imu_projection_3(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.base_fps = 30

    def forward(self, x, fps=None, return_embed_only=False):
        assert fps is not None, "fps must be provided for scaling positional encodings"

        seq_len = x.size(0)
        fps_scale = self.base_fps / fps
        
        # Create scaled position indices (float)
        scaled_positions = torch.arange(0, seq_len, dtype=torch.float, device=x.device) * fps_scale
        
        # Get integer parts and fractional parts
        pos_floor = scaled_positions.long().clamp(0, self.pe.size(0) - 2)
        pos_ceil = (pos_floor + 1).clamp(0, self.pe.size(0) - 1)
        alpha = (scaled_positions - pos_floor.float()).unsqueeze(1)
        
        # Linear interpolation
        pe_scaled = (1 - alpha) * self.pe[pos_floor] + alpha * self.pe[pos_ceil]
        
        if return_embed_only:
            return pe_scaled
        else:
            return x + pe_scaled
    
    def get_pe_at_index(self, t, fps=None):
        """
        Get interpolated positional encoding at a specific temporal index.
        Useful for non-integer or fine-grained temporal positions.
        
        Args:
            t: float or Tensor, temporal index or indices (can be non-integer)
            fps: float, frames per second for scaling
            
        Returns:
            Tensor of shape [d_model] if t is scalar, or [len(t), d_model] if t is Tensor
        """
        assert fps is not None, "fps must be provided for scaling positional encodings"
        fps_scale = self.base_fps / fps
        
        if isinstance(t, (int, float)):
            scaled_t = t * fps_scale
            t_floor = int(min(scaled_t, self.pe.size(0) - 2))
            t_ceil = min(t_floor + 1, self.pe.size(0) - 1)
            alpha = scaled_t - t_floor
            
            return (1 - alpha) * self.pe[t_floor] + alpha * self.pe[t_ceil]
        else:
            # Handle tensor input
            scaled_t = t * fps_scale
            t_floor = scaled_t.long().clamp(0, self.pe.size(0) - 2)
            t_ceil = (t_floor + 1).clamp(0, self.pe.size(0) - 1)
            alpha = (scaled_t - t_floor.float()).unsqueeze(-1)
            
            return (1 - alpha) * self.pe[t_floor] + alpha * self.pe[t_ceil]


class RoPEEncoding(nn.Module):
    """
    1D Rotary Position Embedding (RoPE) as a drop-in replacement for
    sinusoidal positional encoding. Supports FPS-scaled and interpolated
    temporal positions.

    RoPE rotates query/key pairs in 2D subspaces rather than adding a fixed
    offset, but this class exposes the same interface as PositionalEncoding
    so it can be used additively OR via apply_to_qk() for the proper RoPE usage.
    """

    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.base_fps = 30

        # Precompute inverse frequencies: shape [d_model // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)  # [d/2]

        # Precompute cos/sin cache for integer positions: shape [max_len, d_model]
        self._build_cache(max_len)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cache(self, max_len: int):
        positions = torch.arange(max_len, dtype=torch.float)          # [L]
        freqs = torch.outer(positions, self.inv_freq)                  # [L, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)                        # [L, d]
        self.register_buffer("cos_cache", emb.cos())                   # [L, d]
        self.register_buffer("sin_cache", emb.sin())                   # [L, d]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension into the first half."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    # ------------------------------------------------------------------
    # Low-level: get cos/sin at arbitrary (possibly non-integer) positions
    # ------------------------------------------------------------------

    def _interp(self, scaled_positions: torch.Tensor):
        """
        Linearly interpolate cos/sin caches at fractional positions.

        Args:
            scaled_positions: float Tensor of any shape

        Returns:
            cos_emb, sin_emb — same leading shape, last dim d_model
        """
        flat = scaled_positions.reshape(-1)                            # [N]
        p_floor = flat.long().clamp(0, self.max_len - 2)
        p_ceil  = (p_floor + 1).clamp(0, self.max_len - 1)
        alpha   = (flat - p_floor.float()).unsqueeze(-1)               # [N, 1]

        cos_emb = (1 - alpha) * self.cos_cache[p_floor] + alpha * self.cos_cache[p_ceil]
        sin_emb = (1 - alpha) * self.sin_cache[p_floor] + alpha * self.sin_cache[p_ceil]

        shape = scaled_positions.shape + (self.d_model,)
        return cos_emb.reshape(shape), sin_emb.reshape(shape)

    def _scaled_positions(self, seq_len: int, fps: float, device: torch.device):
        fps_scale = self.base_fps / fps
        return torch.arange(seq_len, dtype=torch.float, device=device) * fps_scale

    # ------------------------------------------------------------------
    # Core RoPE application
    # ------------------------------------------------------------------

    def apply_rope(self, x: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor):
        """
        Apply RoPE rotation to x.

        Args:
            x:       [..., seq_len, d_model]  (queries or keys)
            cos_emb: [seq_len, d_model]
            sin_emb: [seq_len, d_model]

        Returns:
            Rotated tensor of same shape as x.
        """
        return x * cos_emb + self._rotate_half(x) * sin_emb

    # ------------------------------------------------------------------
    # Public interface (mirrors PositionalEncoding)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, fps: float = None, return_embed_only: bool = False):
        """
        Args:
            x:   [seq_len, d_model]
            fps: frames per second for temporal scaling

        Returns:
            RoPE-rotated x (or the cos/sin embeddings when return_embed_only=True).

        Note:
            For transformer Q/K attention use apply_to_qk() instead, which is
            the canonical RoPE usage. This forward() applies the rotation
            directly to x so the class is a drop-in replacement.
        """
        assert fps is not None, "fps must be provided"
        scaled_pos = self._scaled_positions(x.size(0), fps, x.device)  # [L]
        cos_emb, sin_emb = self._interp(scaled_pos)                     # [L, d]

        if return_embed_only:
            # Return stacked (cos, sin) so callers can reconstruct the rotation
            return cos_emb, sin_emb

        return self.apply_rope(x, cos_emb, sin_emb)

    def apply_to_qk(self, q: torch.Tensor, k: torch.Tensor, fps: float):
        """
        Canonical RoPE usage: rotate queries and keys before dot-product attention.

        Args:
            q, k: [seq_len, d_model]  or  [batch, seq_len, d_model]
            fps:  frames per second

        Returns:
            q_rot, k_rot — same shapes as inputs
        """
        assert fps is not None
        seq_len = q.shape[-2]
        device  = q.device
        scaled_pos = self._scaled_positions(seq_len, fps, device)
        cos_emb, sin_emb = self._interp(scaled_pos)                     # [L, d]

        return self.apply_rope(q, cos_emb, sin_emb), self.apply_rope(k, cos_emb, sin_emb)

    def get_pe_at_index(self, t, fps: float = None, return_additive: bool = True):
        """
        Get interpolated RoPE embedding at a specific temporal index.

        For additive use (compatible with PositionalEncoding interface), returns
        cos_emb + sin_emb as a single [d_model] tensor. For rotation use, set
        return_additive=False to get (cos_emb, sin_emb).

        Args:
            t:   int/float scalar  →  returns [d] or (cos [d], sin [d])
                 Tensor            →  returns [N, d] or (cos [N, d], sin [N, d])
            fps: frames per second
            return_additive: if True (default), return cos+sin as single tensor for +=

        Returns:
            Tensor [d_model] or [N, d_model] when return_additive=True;
            (cos_emb, sin_emb) when return_additive=False.
        """
        assert fps is not None, "fps must be provided"
        fps_scale = self.base_fps / fps
        device = self.cos_cache.device if hasattr(self, 'cos_cache') else next(self.buffers()).device

        if isinstance(t, (int, float)):
            scaled_t = torch.tensor(t * fps_scale, device=device)
        else:
            scaled_t = (t.float() * fps_scale).to(device)

        cos_emb, sin_emb = self._interp(scaled_t)
        if return_additive:
            return cos_emb + sin_emb
        return cos_emb, sin_emb