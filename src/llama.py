import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from typing import Tuple, Optional, List


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


def precompute_freqs_cis(
    dim: int, seq_len: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the frequency tensors for rotary embeddings"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to the input tensor"""
    batch_size, seq_len, n_heads, head_dim = x.shape
    dim_half = head_dim // 2
    x_reshape = x.reshape(batch_size, seq_len, n_heads, 2, dim_half)
    x1, x2 = x_reshape[..., 0, :], x_reshape[..., 1, :]
    cos_expanded = cos[:seq_len, None, :].expand(seq_len, n_heads, dim_half)
    sin_expanded = sin[:seq_len, None, :].expand(seq_len, n_heads, dim_half)
    out1 = x1 * cos_expanded - x2 * sin_expanded
    out2 = x1 * sin_expanded + x2 * cos_expanded
    out = torch.stack([out1, out2], dim=-2)
    return out.reshape(batch_size, seq_len, n_heads, head_dim).to(dtype=x.dtype)


class FlashAttention(nn.Module):
    """Attention block using Flash Attention for efficient computation"""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert (
            self.head_dim % 8 == 0
        ), "Head dimension must be divisible by 8 for Flash Attention"
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            # Use cached keys and values if available
            k_cache, v_cache = kv_cache
            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=1)
                v = torch.cat([v_cache, v], dim=1)

            # Return updated cache
            new_cache = (k, v)

        output = flash_attn_func(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True
        )
        output = output.reshape(batch_size, seq_len, -1)

        if kv_cache is not None:
            return self.wo(output), new_cache
        return self.wo(output)


class MLP(nn.Module):
    """MLP block with SwiGLU activation"""

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaBlock(nn.Module):
    """Transformer block for Llama model"""

    def __init__(self, dim: int, n_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = FlashAttention(dim, n_heads, dropout)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if kv_cache is not None:
            h = self.attention_norm(x)
            attn_out, new_kv_cache = self.attention(h, cos, sin, kv_cache=kv_cache)
            h = x + attn_out
            out = h + self.mlp(self.mlp_norm(h))
            return out, new_kv_cache
        else:
            h = x + self.attention(self.attention_norm(x), cos, sin)
            out = h + self.mlp(self.mlp_norm(h))
            return out


class LlamaModel(nn.Module):
    """Complete Llama model"""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        max_seq_len: int,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        if mlp_dim is None:
            mlp_dim = 4 * dim
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.cos, self.sin = precompute_freqs_cis(self.dim // self.n_heads, max_seq_len)
        self.blocks = nn.ModuleList(
            [LlamaBlock(dim, n_heads, mlp_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        kv_cache: Optional[List[torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        assert (
            seq_len <= self.max_seq_len
        ), f"Input sequence length ({seq_len}) exceeds maximum ({self.max_seq_len})"
        h = self.token_embeddings(tokens)
        cos = self.cos
        sin = self.sin

        new_kv_cache = None
        if kv_cache is not None:
            new_kv_cache = []

        for i, block in enumerate(self.blocks):
            if kv_cache is not None:
                layer_kv_cache = kv_cache[i] if i < len(kv_cache) else None
                h, layer_new_kv_cache = block(h, cos, sin, layer_kv_cache)
                new_kv_cache.append(layer_new_kv_cache)
            else:
                h = block(h, cos, sin)

        h = self.norm(h)
        logits = self.output(h)

        def compute_loss():
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        def return_logits():
            if kv_cache is not None:
                return logits, new_kv_cache
            else:
                return logits

        return torch.cond(targets is not None, compute_loss, return_logits)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively with KV caching"""
        kv_cache = None
        for _ in range(max_new_tokens):
            if kv_cache is None:
                idx_cond = idx[:, -self.max_seq_len :]
                batch_size = idx_cond.size(0)
                logits, kv_cache = self(
                    idx_cond,
                    kv_cache=[
                        (
                            torch.zeros(
                                batch_size,
                                0,
                                self.n_heads,
                                self.dim // self.n_heads,
                                device=idx.device,
                                dtype=torch.bfloat16,
                            ),
                            torch.zeros(
                                batch_size,
                                0,
                                self.n_heads,
                                self.dim // self.n_heads,
                                device=idx.device,
                                dtype=torch.bfloat16,
                            ),
                        )
                    ]
                    * len(self.blocks),
                )
                logits = logits[:, -1, :] / temperature
            else:
                new_token = idx[:, -1:]
                logits, kv_cache = self(new_token, kv_cache=kv_cache)
                logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
