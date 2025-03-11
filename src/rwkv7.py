import math
from typing import Optional, Tuple, List
import torch
from torch import nn
from torch.nn import functional as F
from .rwkv7_wind import load_cuda_rwkv7_g
from .wkv7s_infer import load_cuda_rwkv7_op, cuda_rwkv7_op_dtype
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
        self.cuda_rwkv7 = load_cuda_rwkv7_op(head_size_a)
        self.n_head = dim_att // head_size_a
        self.head_size = head_size_a
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
            -F.softplus(-(self.w0 + F.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * F.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual
        a = F.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"
        g = F.sigmoid(xg @ self.g1) @ self.g2

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

    def inference_single(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor,
        v_first: torch.Tensor,
        state: torch.Tensor,
    ):
        H, N = self.n_head, self.head_size
        xx = x_prev - x
        xr, xw, xk, xv, xa, xg = (
            x + xx * self.x_r.squeeze(),
            x + xx * self.x_w.squeeze(),
            x + xx * self.x_k.squeeze(),
            x + xx * self.x_v.squeeze(),
            x + xx * self.x_a.squeeze(),
            x + xx * self.x_g.squeeze(),
        )
        r = xr @ self.receptance.weight.t()
        w = F.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        a = F.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = F.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(H, N), dim=-1, p=2.0).view(H * N)
        k = k * (1 + (a - 1) * self.k_a)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * F.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        w = torch.exp(-0.606531 * F.sigmoid((self.w0 + w)))  # 0.606531 = exp(-0.5)

        vk = v.view(H, N, 1) @ k.view(H, 1, N)
        ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
        state = state * w.view(H, 1, N) + state @ ab + vk
        xx = state.to(dtype=x.dtype) @ r.view(H, N, 1)

        xx = self.ln_x(xx.view(1, H * N)).view(H * N)
        xx = xx + (
            (r * k * self.r_k.flatten()).view(H, N).sum(dim=-1, keepdim=True)
            * v.view(H, N)
        ).view(H * N)
        return (xx * g) @ self.output.weight.t(), x, state, v_first

    def inference_multiple(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor,
        v_first: torch.Tensor,
        state: torch.Tensor,
    ):
        H, N = self.n_head, self.head_size
        T = x.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        xr, xw, xk, xv, xa, xg = (
            x + xx * self.x_r.squeeze(),
            x + xx * self.x_w.squeeze(),
            x + xx * self.x_k.squeeze(),
            x + xx * self.x_v.squeeze(),
            x + xx * self.x_a.squeeze(),
            x + xx * self.x_g.squeeze(),
        )

        r = xr @ self.receptance.weight.t()
        w = F.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        a = F.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = F.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(T, H, N), dim=-1, p=2.0).view(T, H * N)
        k = k * (1 + (a - 1) * self.k_a)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * F.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        ######## cuda-free method
        # w = torch.exp(-0.606531 * F.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        # for t in range(T):
        #     r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
        #     vk = v_.view(H,N,1) @ k_.view(H,1,N)
        #     ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
        #     state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
        #     xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)
        w = -F.softplus(-(self.w0 + w)) - 0.5
        xx = self.cuda_rwkv7(state.float(), r, w, k, v, -kk, kk * a)

        xx = self.ln_x(xx.view(T, H * N)).view(T, H * N)
        xx = xx + (
            (r * k * self.r_k.flatten()).view(T, H, N).sum(dim=-1, keepdim=True)
            * v.view(T, H, N)
        ).view(T, H * N)
        return (xx * g) @ self.output.weight.t(), x[-1, :], state, v_first


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

    def inference_single(self, x: torch.Tensor, x_prev: torch.Tensor):
        xx = x_prev - x
        k = x + xx * self.x_k.squeeze()
        k = F.relu(k @ self.key.weight.t()) ** 2
        return k @ self.value.weight.t(), x

    def inference_multiple(self, x: torch.Tensor, x_prev: torch.Tensor):
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        k = x + xx * self.x_k.squeeze()
        k = F.relu(k @ self.key.weight.t()) ** 2
        return k @ self.value.weight.t(), x[-1, :]


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
        self.head_size = head_size_a
        self.n_embd = n_embd
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

    @torch.inference_mode()
    def inference(
        self,
        x: torch.Tensor,
        max_new_tokens=16,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        eos=-1,
    ):
        state = [None for _ in range(len(self.blocks) * 3)]
        for i in range(len(self.blocks)):  # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
            state[i * 3 + 0] = torch.zeros(
                self.n_embd,
                dtype=cuda_rwkv7_op_dtype,
                requires_grad=False,
                device=x.device,
            )
            state[i * 3 + 1] = torch.zeros(
                (self.n_embd // self.head_size, self.head_size, self.head_size),
                dtype=cuda_rwkv7_op_dtype,
                requires_grad=False,
                device=x.device,
            )
            state[i * 3 + 2] = torch.zeros(
                self.n_embd,
                dtype=cuda_rwkv7_op_dtype,
                requires_grad=False,
                device=x.device,
            )
        tokens = []
        for _ in range(max_new_tokens):
            logits, state = self.inference_step(x, state)
            x = self.sample(logits, temperature, top_p, top_k)
            tokens.append(x.item())
            if x.item() == eos:
                break
        return tokens

    def sample(
        self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int
    ):
        probs = F.softmax(logits.squeeze().float(), dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)

        if top_k > 0:
            probs[sorted_ids[top_k:]] = 0

        if top_p < 1:
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.searchsorted(cumulative_probs, top_p)
            cutoff = sorted_probs[cutoff_index]
            probs[probs < cutoff] = 0

            if top_p > 0:
                idx = torch.where(probs == cutoff)[0]
                if len(idx) > 0:
                    probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                    # assert abs(torch.sum(probs).item() - top_p) < 1e-6
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        return torch.multinomial(probs, num_samples=1)

    def inference_step(self, x: torch.Tensor, state: List[torch.Tensor]):
        if x.numel() > 1:
            return self.inference_step_multiple(x, state)
        else:
            return self.inference_step_single(x.squeeze(), state)

    def inference_step_single(self, x: torch.Tensor, state: List[torch.Tensor]):
        x = self.emb(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            if block.layer_id == 0:
                x = block.ln0(x)
            xx = block.ln1(x)

            (
                xx,
                state[block.layer_id * 3 + 0],
                state[block.layer_id * 3 + 1],
                v_first,
            ) = block.att.inference_single(
                xx,
                state[block.layer_id * 3 + 0],
                v_first,
                state[block.layer_id * 3 + 1],
            )
            x = x + xx

            xx = block.ln2(x)

            xx, state[block.layer_id * 3 + 2] = block.ffn.inference_single(
                xx,
                state[block.layer_id * 3 + 2],
            )
            x = x + xx

        x = self.ln_out(x)
        x = x @ self.head.weight.t()
        return x, state

    def inference_step_multiple(self, x: torch.Tensor, state: List[torch.Tensor]):
        x = self.emb(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            if block.layer_id == 0:
                x = block.ln0(x)
            xx = block.ln1(x)

            (
                xx,
                state[block.layer_id * 3 + 0],
                state[block.layer_id * 3 + 1],
                v_first,
            ) = block.att.inference_multiple(
                xx,
                state[block.layer_id * 3 + 0],
                v_first,
                state[block.layer_id * 3 + 1],
            )
            x = x + xx

            xx = block.ln2(x)

            xx, state[block.layer_id * 3 + 2] = block.ffn.inference_multiple(
                xx,
                state[block.layer_id * 3 + 2],
            )
            x = x + xx

        x = self.ln_out(x)
        x = x @ self.head.weight.t()
        return x[-1], state
