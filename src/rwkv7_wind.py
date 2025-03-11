import torch
from torch.utils.cpp_extension import load
from pathlib import Path

csrc = Path(__file__).parent.parent / "csrc"
CHUNK_LEN = 16
cuda_rwkv7_g_dtype = torch.bfloat16


def load_cuda_rwkv7_g(head_size_a=64):
    load(
        name="rwkv7_wind_backstepping",
        sources=[csrc.joinpath("wkv7_op.cpp"), csrc.joinpath("wkv7_op.cu")],
        is_python_module=False,
        extra_cuda_cflags=[
            "-res-usage",
            f"-D_C_={head_size_a}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ],
    )
    op = torch.ops.rwkv7_wind_backstepping

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            w: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            z: torch.Tensor,
            b: torch.Tensor,
        ):
            B, T, H, C = w.shape
            assert T % CHUNK_LEN == 0
            assert all(
                i.dtype == cuda_rwkv7_g_dtype and i.is_contiguous()
                for i in [w, q, k, v, z, b]
            )
            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            op.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y

        @staticmethod
        def backward(ctx, dy: torch.Tensor):
            assert all(
                i.dtype == cuda_rwkv7_g_dtype and i.is_contiguous() for i in [dy]
            )
            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            op.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
            return dw, dq, dk, dv, dz, db

    def cuda_rwkv7_g(
        q: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B, T, HC = q.shape
        q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
        return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)

    return cuda_rwkv7_g
