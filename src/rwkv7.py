import math
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from .rwkv7_wind import load_cuda_rwkv7_g
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed


class RWKV7TMix(nn.Module):
    def __init__(
        self,
        head_size_a: int,
        dim_att: int,
        n_embd: int,
        n_layer: int,
        head_size_divisor: int,
        layer_id: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.cuda_rwkv7_g = load_cuda_rwkv7_g(head_size_a)
        self.n_head = dim_att // head_size_a
        assert dim_att % self.n_head == 0
        H = self.n_head
        N = head_size_a
        C = n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
            )
            self.x_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
            )
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (
                    0.85 + 1.0 * ratio_0_to_1**0.5
                )
            self.w0 = nn.Parameter(
                decay_speed.reshape(1, 1, C) + 0.5
            )  # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round((1.3 * (C**0.5)) / 32) * 32))  # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round((0.6 * (C**0.8)) / 32) * 32))  # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(
                H, C, eps=(1e-5) * (head_size_divisor**2)
            )  # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()
            layer_scale = (1 + layer_id) / n_layer
            self.ln_x.weight = nn.Parameter(
                (self.ln_x.weight * 0.0) + (layer_scale**0.7)
            )

    def forward(self, x: torch.Tensor, v_first: torch.Tensor):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual
        a = torch.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = self.cuda_rwkv7_g(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class RWKV7CMix(nn.Module):
    def __init__(self, n_layer: int, n_embd: int, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(n_embd**0.5), 0.5/(n_embd**0.5))
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = F.relu(self.key(k)) ** 2

        return self.value(k)


class Block(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_embd: int,
        head_size_a: int,
        dim_att: int,
        head_size_divisor: int,
        layer_id: int,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV7TMix(
            head_size_a, dim_att, n_embd, n_layer, head_size_divisor, layer_id
        )

        self.ffn = RWKV7CMix(n_layer, n_embd, layer_id)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss: torch.Tensor, y: torch.Tensor):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV7(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_embd: int,
        head_size_a: int,
        dim_att: Optional[int],
        dim_ffn: Optional[int],
        head_size_divisor: int,
        vocab_size: int,
        head_qk: Optional[int],
        ctx_len: int,
        dropout: Optional[float],
    ):
        super().__init__()
        if dim_att is None:
            dim_att = n_embd
        if dim_ffn is None:
            dim_ffn = int((n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size
        assert n_embd % 32 == 0
        assert dim_att % 32 == 0
        assert dim_ffn % 32 == 0

        self.emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList(
            [
                Block(n_layer, n_embd, head_size_a, dim_att, head_size_divisor, i)
                for i in range(n_layer)
            ]
        )

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.head_qk = head_qk
        if head_qk:
            self.head_q = nn.Linear(n_embd, head_qk, bias=False)
            nn.init.zeros_(self.head_q.weight)
            self.head_k = nn.Linear(n_embd, head_qk, bias=False)
            nn.init.uniform_(self.head_q.weight, a=-0.1, b=0.1)
            self.register_buffer("copy_mask", torch.tril(torch.ones(ctx_len, ctx_len)))
        if dropout:
            self.drop0 = nn.Dropout(p=dropout)
        else:
            self.drop0 = None
        self.ctx_len = ctx_len
        nn.init.uniform_(self.emb.weight, a=-1e-4, b=1e-4)
        if vocab_size > n_embd:
            scale = 0.5 * math.sqrt(vocab_size / n_embd)
        else:
            scale = 0.5
        nn.init.orthogonal_(self.head.weight, gain=scale)

    def create_optimizers(
        self,
        weight_decay: float,
        ft: bool,
        cpu_adam: bool,
        lr_init: float,
        betas: Tuple[float, float],
        adam_eps: float,
        layerwise_lr=True,
    ):

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():

            if ("att.w0" in n) and layerwise_lr:
                if ft:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (weight_decay > 0)
                and (".weight" in n)
            ):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.named_parameters()}

        if layerwise_lr:
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr_scale": 1.0,
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr_scale": 5.0 if ft else 2.0,
                },  # test: 2e-3 / lr_init},
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "lr_scale": 5.0 if ft else 3.0,
                },  # test: 3e-3 / lr_init},
            ]
        else:
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr_scale": 1.0,
                }
            ]

        if weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": weight_decay,
                    "lr_scale": 1.0,
                }
            ]
        cls = DeepSpeedCPUAdam if cpu_adam else FusedAdam
        dep = {"adamw_mode": True} if cpu_adam else {"adam_w_mode": True}
        return [
            cls(
                group["params"],
                lr=lr_init * group["lr_scale"],
                betas=betas,
                eps=adam_eps,
                bias_correction=True,
                weight_decay=group["weight_decay"],
                amsgrad=False,
                **dep,
            )
            for group in optim_groups
            if len(group["params"]) > 0
        ]

    def forward(self, idx: torch.Tensor, checkpointing: bool):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        if self.drop0:
            x = self.drop0(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            if checkpointing:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)

        if self.head_qk:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / self.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            c = c @ F.one_hot(idx, num_classes=self.vocab_size).to(dtype=c.dtype)
            x = self.head(x) + c
        else:
            x = self.head(x)

        return x

    def training_step(
        self, idx: torch.Tensor, targets: torch.Tensor, checkpointing: bool
    ):
        logits = self(idx, checkpointing)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)
