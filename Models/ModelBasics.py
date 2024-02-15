import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from tqdm import tqdm
import time
import math
from einops import rearrange


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int, s: int, p: int = 0, d: int = 1, g: int = 1):
        """Convolution 1D with BatchNorm1d and LeakyReLU activation. Expect input shape (B, C, L).

        :param in_c: in channels
        :param out_c: out channels
        :param k: kernel size
        :param s: stride
        :param p: padding, defaults to 0
        :param d: dilation, defaults to 1
        :param g: group, defaults to 1
        """
        super(ConvNormAct, self).__init__(
            nn.Conv1d(in_c, out_c, k, s, p, d, g, bias=False),
            nn.GroupNorm(32, out_c),
            Swish()
        )


class CrossAttention(nn.Module):
    def __init__(self, in_c: int, context_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.H = num_heads

        self.q_proj = nn.Sequential(
            nn.GroupNorm(32, in_c),
            Swish(),
            nn.Conv1d(in_c, in_c * self.H, 1, 1, 0),
        )

        self.kv_proj = nn.Conv1d(context_dim, in_c * 2 * self.H, 3, 1, 1)

        self.scale = 1 / math.sqrt(in_c)

        self.post_stage = nn.Sequential(
            nn.GroupNorm(32, in_c * self.H),
            nn.Dropout(dropout),
            nn.Conv1d(in_c * self.H, in_c, 1, 1, 0),
        )

        nn.init.zeros_(self.post_stage[-1].weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C_x, L_x)
        # context: (B, C_c, L_c)

        q = rearrange(self.q_proj(x), 'b (h c) l -> (b h) c l', h=self.H)  # (B*H, C_x, L_x)

        k, v = self.kv_proj(context).chunk(2, dim=1)
        k = rearrange(k, 'b (h c) l -> (b h) l c', h=self.H)  # (B*H, L_c, C_x)
        v = rearrange(v, 'b (h c) l -> (b h) c l', h=self.H)  # (B*H, C_x, L_c)

        # (L_c, C_x) @ (C_x, L_x) -> (L_c, L_x)
        attn = torch.softmax(torch.bmm(k, q) * self.scale, dim=1)  # (B*H, L_c, L_x)

        return x + self.post_stage(rearrange(torch.bmm(v, attn), '(b h) c l -> b (h c) l', h=self.H))  # (B, C, L)


class TimestepEncoder(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, max_time: int, hidden_dim: int = 256, embed_dim: int = 128) -> None:
        super().__init__()

        # --- Diffusion step Encoding ---
        position = torch.arange(max_time, dtype=torch.float32, device=self.device).unsqueeze(1)  # (max_time, 1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=self.device) * -(
                math.log(1.0e4) / hidden_dim))  # (feature_dim / 2)
        self.pos_enc = torch.zeros((max_time, hidden_dim), dtype=torch.float32,
                                   device=self.device)  # (max_time, feature_dim)
        self.pos_enc[:, 0::2] = torch.sin(position * div_term)
        self.pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.proj = nn.Linear(hidden_dim, embed_dim)  # (B, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        time_embed = self.pos_enc[t, :]  # (B, hidden_dim)
        return self.proj(time_embed)  # (B, embed_dim, 1)


class PosLengthEncoder(nn.Module):
    """ Add positional embeddings and length embeddings to the input """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, in_c: int, out_c: int, max_length: int, embed_dim: int, k: int = 3, s: int = 1,
                 p: int = 1) -> None:
        super().__init__()

        position = torch.arange(max_length + 1, dtype=torch.float32, device=self.device).unsqueeze(1)  # (max_time, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32, device=self.device) *
                             -(math.log(1.0e4) / embed_dim))  # (feature_dim / 2)
        self.pos_enc = torch.zeros((max_length + 1, embed_dim), dtype=torch.float32,
                                   device=self.device)  # (max_time, feature_dim)
        self.pos_enc[:, 0::2] = torch.sin(position * div_term)
        self.pos_enc[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = self.pos_enc.transpose(0, 1).unsqueeze(0)  # (1, feature_dim, max_time)

        self.out_proj = nn.Conv1d(in_c + embed_dim + embed_dim, out_c, k, s, p)

    def forward(self, x):
        # x: (B, C, L)
        B, _, L = x.shape
        len_embed = self.pos_enc[:, :, L].unsqueeze(-1).repeat(B, 1, L)  # (B, E, L)
        pos_embed = self.pos_enc[:, :, :L].repeat(B, 1, 1)  # (B, E, L)
        return self.out_proj(torch.cat([x, len_embed, pos_embed], dim=1))  # (B, C + 2E, L)


def inferSpeedTest1K(model, *dummy_inputs):
    model.eval()
    start = time.time()
    with torch.no_grad():
        if len(dummy_inputs) == 1:
            for i in tqdm(range(1000)):
                model(dummy_inputs[0])
        else:
            for i in tqdm(range(1000)):
                model(*dummy_inputs)
    end = time.time()
    print(f"Each inference takes {end - start:.6f} ms")