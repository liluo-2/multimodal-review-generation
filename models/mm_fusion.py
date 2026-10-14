# model_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusionWithGate(nn.Module):
    """Single-head cross attention: Q = learnable vector, KV = concat(text, image)"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, text_emb, img_emb):
        # ✅ 统一到本层权重的 dtype/device（最小保险）
        kv = torch.cat([text_emb, img_emb], dim=1)
        kv = kv.to(dtype=self.key_proj.weight.dtype, device=self.key_proj.weight.device)

        K = self.key_proj(kv)
        V = self.value_proj(kv)
        Q = self.query.expand(K.size(0), -1, -1).to(dtype=K.dtype, device=K.device)

        attn = torch.matmul(Q, K.transpose(-1, -2)) / (K.size(-1) ** 0.5)
        w = F.softmax(attn, dim=-1)
        fused = torch.matmul(w, V)  # (B,1,D)
        return self.out_proj(fused)

class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross attention for fusing text and image embeddings.
    Q = learnable query tokens, shared across data
    K,V = concat(text_seq, img_seq)
    Output: (B, num_queries, hidden_dim)
    """
    def __init__(self, hidden_dim=256, num_queries=32, num_heads=8, dropout=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable Q tokens
        self.query = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.normal_(self.query, std=0.02)

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Transformer-style normalization & FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x, B):
        """(B, L, D) → (B, H, L, head_dim)"""
        return x.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, text_emb, img_emb):
        """
        text_emb: (B, Lt, D) or None
        img_emb:  (B, Li, D) or None
        """
        # ---- 统一到 (B, L, D) ----
        if text_emb is None and img_emb is None:
            raise ValueError("Both text_emb and img_emb cannot be None.")

        if text_emb is not None and text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0)
        if img_emb is not None and img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(0)

        B = text_emb.size(0) if text_emb is not None else img_emb.size(0)

        if text_emb is None:
            text_emb = torch.zeros(B, 1, self.hidden_dim, device=img_emb.device)
        if img_emb is None:
            img_emb = torch.zeros(B, 1, self.hidden_dim, device=text_emb.device)

        # ---- Concat KV ----
        kv = torch.cat([text_emb, img_emb], dim=1)  # (B, Lt+Li, D)

        # ---- Projection ----
        Q = self.q_proj(self.query).expand(B, -1, -1)      # (B, Nq, D)
        K = self.k_proj(kv)                                # (B, L, D)
        V = self.v_proj(kv)                                # (B, L, D)

        # ---- Multi-head attention ----
        Q = self._reshape(Q, B)                            # (B, H, Nq, Hd)
        K = self._reshape(K, B)                            # (B, H, L, Hd)
        V = self._reshape(V, B)                            # (B, H, L, Hd)

        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)                    # (B, H, Nq, Hd)

        # ---- Combine heads ----
        context = context.transpose(1, 2).reshape(B, self.num_queries, self.hidden_dim)
        context = self.out_proj(context)
        context = self.dropout(context)

        # ---- Residual + Norm ----
        out = self.norm1(context + self.query.expand(B, -1, -1))

        # ---- FFN ----
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)

        return out    # (B, num_queries, hidden_dim)

class MultiHeadGMU(nn.Module):
    """4-head GMU gating for (text, cross, image) fusion -> (B,4,D)
    If cross_emb is None, only uses text_emb and img_emb (2-input mode)
    """
    def __init__(self, dim, num_heads=4, use_softmax=False):
        super().__init__()
        self.num_heads = num_heads
        self.use_softmax = use_softmax
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "Wv": nn.Linear(dim, dim),
                "Wt": nn.Linear(dim, dim),
                "Wc": nn.Linear(dim, dim),
                "Wz": nn.Linear(3 * dim, dim),
                "Wz_2": nn.Linear(2 * dim, dim)  # For 2-input mode
            }) for _ in range(num_heads)
        ])

    def forward(self, text_emb, cross_emb=None, img_emb=None):
        # Support both 3-input (text, cross, img) and 2-input (text, img) modes
        if cross_emb is None:
            # 2-input mode: only text and img
            if img_emb is None:
                raise ValueError("At least one of cross_emb or img_emb must be provided")
            outs = []
            for h in self.heads:
                hv = torch.tanh(h["Wv"](img_emb))
                ht = torch.tanh(h["Wt"](text_emb))
                
                if not self.use_softmax:
                    z = torch.sigmoid(h["Wz_2"](torch.cat([hv, ht], dim=-1)))
                    fused = z * ht + (1 - z) * hv
                else:
                    wz = h["Wz_2"](torch.cat([hv, ht], dim=-1))  # (B,1,D)
                    # softmax-style：用每个维度的两路权重
                    alpha = torch.softmax(
                        torch.stack([hv, ht], dim=2), dim=2
                    )  # (B,1,2,D)
                    wv, wt = alpha[:, :, 0, :], alpha[:, :, 1, :]
                    fused = wv * hv + wt * ht
                
                outs.append(fused)
            return torch.cat(outs, dim=1)  # (B,4,D)
        else:
            # 3-input mode: text, cross, img
            if img_emb is None:
                raise ValueError("img_emb must be provided when cross_emb is provided")
            outs = []
            for h in self.heads:
                hv = torch.tanh(h["Wv"](img_emb))
                ht = torch.tanh(h["Wt"](text_emb))
                hc = torch.tanh(h["Wc"](cross_emb))

                if not self.use_softmax:
                    z = torch.sigmoid(h["Wz"](torch.cat([hv, ht, hc], dim=-1)))
                    fused = z * hc + (1 - z) * (0.5 * (hv + ht))
                else:
                    wz = h["Wz"](torch.cat([hv, ht, hc], dim=-1))  # (B,1,D)
                    # softmax-style：用每个维度的三路权重
                    alpha = torch.softmax(
                        torch.stack([hv, ht, hc], dim=2), dim=2
                    )  # (B,1,3,D)
                    wv, wt, wc = alpha[:, :, 0, :], alpha[:, :, 1, :], alpha[:, :, 2, :]
                    fused = wv * hv + wt * ht + wc * hc

                outs.append(fused)

            return torch.cat(outs, dim=1)  # (B,4,D)