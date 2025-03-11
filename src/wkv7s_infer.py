from torch.utils.cpp_extension import load
import torch
from pathlib import Path
from typing import List

csrc = Path(__file__).parent.parent / "csrc"

cuda_rwkv7_op_dtype = torch.bfloat16


def load_cuda_rwkv7_op(head_size=64):
    load(
        name="wkv7s",
        sources=[
            csrc.joinpath("wkv7s_infer_op.cpp"),
            csrc.joinpath("wkv7s_infer_op.cu"),
        ],
        is_python_module=False,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={head_size}",
        ],
    )
    op = torch.ops.wkv7s_infer

    class WKV7Infer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                T, C = r.size()
                H = C // head_size
                assert head_size == C // H
                assert all(
                    x.dtype == cuda_rwkv7_op_dtype and x.is_contiguous()
                    for x in [r, w, k, v, a, b]
                )
                y = torch.empty(
                    (T, C),
                    device=k.device,
                    dtype=cuda_rwkv7_op_dtype,
                    requires_grad=False,
                    memory_format=torch.contiguous_format,
                )
                op.forward(1, T, C, H, state, r, w, k, v, a, b, y)
                return y

    def rwkv7_op(
        state: List[torch.Tensor],
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return WKV7Infer.apply(state, r, w, k, v, a, b)

    return rwkv7_op
