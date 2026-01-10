import torch.nn as nn
from einops import rearrange

F = nn.functional
class MHSA2D(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "Число каналов должно быть кратно числу голов"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V за один проход (1×1 conv)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # Получаем Q, K, V
        qkv = self.qkv(x)  # (B, 3C, H, W)
        qkv = rearrange(qkv, 'b (three h d) h0 w0 -> three b h (h0 w0) d', three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # каждая: (B, num_heads, N, head_dim)

        # Attention: (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Применяем внимание к значениям
        out = attn @ v  # (B, heads, N, head_dim)
        out = rearrange(out, 'b h n d -> b (h d) n')  # (B, C, N)
        out = out.view(B, C, H, W)  # (B, C, H, W)

        # Финальная проекция
        out = self.proj(out)
        out = self.proj_drop(out)
        return out