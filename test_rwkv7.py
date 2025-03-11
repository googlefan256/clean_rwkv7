from src.rwkv7 import RWKV7
import torch
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    max_seq_len = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RWKV7(
        6,
        512,
        64,
        None,
        None,
        8,
        vocab_size,
        None,
        max_seq_len,
        None,
    ).to(device, dtype=torch.bfloat16, non_blocking=True)
    model_train = torch.compile(
        model, fullgraph=True, options={"triton.cudagraphs": True}
    )
    optimizers = model.create_optimizers(0.001, False, False, 6e-4, (0.9, 0.99), 1e-8)
    steps = 1000
    schedulers = [
        get_cosine_schedule_with_warmup(optim, steps // 10, steps)
        for optim in optimizers
    ]
    model.train()
    input_ids = tokenizer.encode(
        "Why is the sun hot? Well its a simple problem sur. Thank you.!",
        return_tensors="pt",
    ).to(device)
    pbar = tqdm(range(steps))
    for _ in pbar:
        for optim in optimizers:
            optim.zero_grad()
        loss = model_train.training_step(
            input_ids[:, :-1].clone(), input_ids[:, 1:].clone(), False
        )
        pbar.set_description(f"Train: {loss.item():.4f}")
        loss.backward()
        for optim, schedule in zip(optimizers, schedulers):
            optim.step()
            schedule.step()
    model.eval()
    out = model.inference(
        tokenizer.encode("Why", return_tensors="pt").to(device).squeeze(),
        max_new_tokens=20,
    )
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
