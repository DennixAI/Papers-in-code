import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from config import GemmaZeroConfig
from model import GemmaZeroModel

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GemmaZeroConfig()
    model = GemmaZeroModel(config).to(device)

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = GradScaler()

    batch_size = 4
    seq_len = 1024
    
    # Dummy data
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    labels = inputs.clone()

    model.train()
    print("Starting training...")

    for step in range(100):
        optimizer.zero_grad()
        
        with autocast(dtype=torch.float16):
            logits = model(inputs)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 10 == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"Step {step} | Loss: {loss.item():.4f} | VRAM: {mem:.2f} GB")

if __name__ == "__main__":
    train()