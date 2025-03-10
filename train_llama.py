import torch
import torch.optim as optim
from tqdm import tqdm
import time
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
from trl.trainer.utils import ConstantLengthDataset
from torch.utils.data import DataLoader
from trl.trainer.utils import ConstantLengthDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from src.llama import LlamaModel


def train_model():
    print("Testing training loop with LlamaModel on 'Hello world!'...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", data_files="sample/10BT/000_00000.parquet"
    )["train"]
    vocab_size = len(tokenizer)
    dim = 1024
    n_layers = 16
    n_heads = 16
    max_seq_len = 1024
    learning_rate = 1e-3
    steps = 1000
    warmup_steps = 10
    batch_size = 2
    ds: ConstantLengthDataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=ds,
        dataset_text_field="text",
        seq_length=max_seq_len + 1,
        num_of_sequences=64,
        infinite=False,
        shuffle=False,
    )
    train_loader = iter(
        DataLoader(
            ds, batch_size=batch_size, collate_fn=default_data_collator, shuffle=False
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Vocabulary size: {vocab_size}")

    model = LlamaModel(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    ).to(device, dtype=torch.bfloat16, non_blocking=True)
    print(f"{sum([p.numel() for p in model.parameters()]) / 1000000:2f}M")
    model.sin = model.sin.to(device, non_blocking=True)
    model.cos = model.cos.to(device, non_blocking=True)
    model_train = torch.compile(
        model, fullgraph=True, options={"triton.cudagraphs": True}
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    dummy_input_ids = torch.randint(
        0, 10, (batch_size, max_seq_len + 1), device=device, dtype=torch.long
    )
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        loss = model_train(
            dummy_input_ids[:, :-1].clone(), targets=dummy_input_ids[:, 1:].clone()
        )
        loss.backward()
        optimizer.step()
    print("Warmup done")
    now = time.time()
    pbar = tqdm(range(1000), desc="Train")
    for step in pbar:
        batch = next(train_loader)
        optimizer.zero_grad()
        loss = model(
            batch["input_ids"][:, :-1].to(device, non_blocking=True),
            targets=batch["input_ids"][:, 1:].to(device, non_blocking=True),
        )
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Train: {loss.item():.4f}")
    print(f"{steps}steps done in {time.time() - now}s")
    print("Generating text from trained model:")
    model.eval()
    with torch.inference_mode():
        start_text = "Hello"
        start_tokens = tokenizer.encode(start_text, return_tensors="pt").to(device)
        generated_ids = model.generate(
            start_tokens, max_new_tokens=15, temperature=0.7, top_k=40
        )
        generated_text = tokenizer.decode(generated_ids[0])
        print(f"Generated: {generated_text}")


if __name__ == "__main__":
    print("LlamaModel Test Script")
    train_model()
