import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from .rwkv7_wind import load_cuda_rwkv7_g


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
            # self.output.weight.data.zero_()

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
        # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        # self.value.weight.data.zero_()

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


class RWKV(nn.Module):
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
        self.n_embd = n_embd
        self.head_qk = head_qk
        if head_qk > 0:
            self.head_q = nn.Linear(n_embd, head_qk, bias=False)
            self.head_k = nn.Linear(n_embd, head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(ctx_len, ctx_len)))
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
        else:
            self.drop0 = None
        self.ctx_len = ctx_len

    def configure_optimizers(self):

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():

            # if not p.requires_grad:
            #     continue
            if args.train_type == "states":
                if "time_sta" not in n:
                    continue

            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif ("time_sta" in n) and (args.weight_decay > 0):
                lr_decay.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n) or ("att.w0" in n)) and (
                args.layerwise_lr > 0
            ):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (args.weight_decay > 0)
                and (".weight" in n)
            ):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        if self.trainer.is_global_zero:
            print("decay", lr_decay, "\n")
            print("1x", lr_1x, "\n")
            print("2x", lr_2x, "\n")
            print("3x", lr_3x, "\n")

        param_dict = {n: p for n, p in self.named_parameters()}

        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {
                        "params": [param_dict[n] for n in lr_1x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 1.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_2x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 2e-3 / args.lr_init},
                    {
                        "params": [param_dict[n] for n in lr_3x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {
                        "params": [param_dict[n] for n in lr_1x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 1.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_2x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 2.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_3x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 3.0,
                    },
                ]
        else:
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "my_lr_scale": 1.0,
                }
            ]

        if args.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=args.lr_init,
                    betas=args.betas,
                    eps=args.adam_eps,
                    bias_correction=True,
                    adamw_mode=True,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=args.lr_init,
                betas=args.betas,
                eps=args.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=args.lr_init,
                    betas=args.betas,
                    eps=args.adam_eps,
                    bias_correction=True,
                    adamw_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=args.lr_init,
                betas=args.betas,
                eps=args.adam_eps,
                bias_correction=True,
                adam_w_mode=False,
                weight_decay=0,
                amsgrad=False,
            )
        # return ZeroOneAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx: torch.Tensor, deepspeed_checkpoint=False):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        if self.drop0:
            x = self.drop0(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            if deepspeed_checkpoint:
                import deepspeed

                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)

        if self.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / self.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            c = c @ F.one_hot(idx, num_classes=self.vocab_size).to(dtype=c.dtype)
            x = self.head(x) + c
        else:
            x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        if args.my_qa_mask != 1:
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            idx, targets, mask = batch
            mask = mask.view(-1)
            sum_mask = torch.sum(mask).item()
            # if sum_mask == 0:
            #     return torch.tensor([0.0], requires_grad=True)

            logits = self(idx)
            if sum_mask == mask.shape[0]:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                # print('rank', self.global_rank, 'loss', loss.item())
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
                )
                # loss_raw = loss
                loss = torch.sum(loss * mask) / sum_mask

                # torch.set_printoptions(threshold=10000)
                # if True: #self.global_rank == 1:
                #     tmp = ''
                #     sss = 0
                #     ccc = 0
                #     for i in range(mask.shape[0]):
                #         if mask[i] > 0:
                #             tmp += str(idx.view(-1)[i].item()) + ','
                #             sss += loss_raw.view(-1)[i].float().item()
                #             ccc += 1
                #     print('rank', self.global_rank, 'loss', loss.item(), 'lavg', sss / ccc)#, 'tmp', tmp, 'input', idx)

        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0] != "2":
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def reset_parameters(self):
        nn.init.uniform_(self.emb.weight, a=-1e-4, b=1e-4)
        if self.vocab_size > self.n_embd:
            scale = 0.5 * math.sqrt(self.vocab_size / self.n_embd)
        else:
            scale = 0.5
        nn.init.orthogonal_(self.head.weight, gain=scale)

    def generate_init_weight(self):
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print(n, "Init")
            elif n in ["emb.weight", "head.weight"]:
                pass
            else:
                scale = 0.0
                assert n.endswith(".weight")  # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in n:
                        scale = 0
                if "head_k." in n:
                    scale = 0.1
                if "head_q." in n:
                    scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
        return m
