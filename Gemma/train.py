%%writefile train.py
import os
import subprocess
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import IterableDataset, DataLoader

# --- Dependency Check ---
try:
    import bitsandbytes as bnb
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from huggingface_hub import HfApi, create_repo, hf_hub_download
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call(["pip", "install", "-q", "datasets", "transformers", "accelerate", "bitsandbytes", "huggingface_hub"])
    import bitsandbytes as bnb
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from huggingface_hub import HfApi, create_repo, hf_hub_download

# --- Local Imports ---
from config import GemmaZeroConfig
from model import GemmaZeroModel

# --- Configuration ---
HF_TOKEN = "" # <--- PASTE YOUR WRITE TOKEN HERE
REPO_NAME = "FusionCorp/gemma-zero"

# --- Dataset Class ---
class TinyStoriesDataset(IterableDataset):
    def __init__(self, seq_len=1024):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def __iter__(self):
        iterator = iter(self.dataset)
        for item in iterator:
            if len(item['text']) < 50: continue
            tokens = self.tokenizer(
                item['text'], max_length=self.seq_len, truncation=True, 
                padding="max_length", return_tensors="pt"
            )
            yield tokens.input_ids.squeeze(0)

# --- Main Training Function ---
def train():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    if HF_TOKEN:
        try:
            api = HfApi(token=HF_TOKEN)
            create_repo(repo_id=REPO_NAME, repo_type="model", token=HF_TOKEN, exist_ok=True)
            print(f"✅ Connected to HF Repo: {REPO_NAME}")
        except Exception as e:
            print(f"⚠️ HF Connection failed: {e}")

    # 2. Model & Config
    config = GemmaZeroConfig()
    model = GemmaZeroModel(config).to(device)
    model.gradient_checkpointing_enable()
    
    print(f"🚀 Model initialized: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M Parameters")

    # 3. Optimization
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=6e-4, weight_decay=0.01)
    scaler = GradScaler()
    
    # 4. Hyperparameters
    BATCH_SIZE = 8
    ACCUM_STEPS = 4
    SEQ_LEN = config.max_position_embeddings
    TOTAL_STEPS = 5000
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, TOTAL_STEPS)

    # 5. Resume Logic
    FULL_CHECKPOINT_NAME = "full_checkpoint.pth"
    LIGHT_WEIGHTS_NAME = "pytorch_model.bin"
    start_step = 0

    # Auto-Download Checkpoint
    if not os.path.exists(FULL_CHECKPOINT_NAME) and HF_TOKEN:
        try:
            print("🔍 Checking remote checkpoint...")
            path = hf_hub_download(repo_id=REPO_NAME, filename="latest_full_checkpoint.pth", token=HF_TOKEN)
            import shutil
            shutil.copy(path, FULL_CHECKPOINT_NAME)
        except:
            print("ℹ️ No remote checkpoint found. Starting fresh.")

    # Load Checkpoint
    if os.path.exists(FULL_CHECKPOINT_NAME):
        print("🔄 Loading checkpoint...")
        checkpoint = torch.load(FULL_CHECKPOINT_NAME, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"✅ Resumed from Step {start_step}")
    
    # 6. Data Loop
    dataloader = DataLoader(TinyStoriesDataset(SEQ_LEN), batch_size=BATCH_SIZE, num_workers=0)
    data_iter = iter(dataloader)
    pad_token_id = 50256
    model.train()

    print("🏁 Starting Training Loop...")
    for step in range(start_step, TOTAL_STEPS):
        try: inputs = next(data_iter).to(device)
        except StopIteration: data_iter = iter(dataloader); inputs = next(data_iter).to(device)
        
        labels = inputs.clone()
        labels[labels == pad_token_id] = -100

        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            loss = loss / ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            if (step + 1) % 50 == 0:
                 print(f"Step {step+1} | Loss: {loss.item() * ACCUM_STEPS:.4f}")

        # 7. Save & Upload
        if (step + 1) % 500 == 0:
            print(f"💾 Saving Step {step+1}...")
            # Save Light
            torch.save(model.state_dict(), LIGHT_WEIGHTS_NAME)
            # Save Full
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'step': step,
            }, FULL_CHECKPOINT_NAME)

            if HF_TOKEN:
                try:
                    api.upload_file(path_or_fileobj=LIGHT_WEIGHTS_NAME, path_in_repo=f"checkpoint-{step+1}/pytorch_model.bin", repo_id=REPO_NAME)
                    api.upload_file(path_or_fileobj=FULL_CHECKPOINT_NAME, path_in_repo="latest_full_checkpoint.pth", repo_id=REPO_NAME)
                    print("☁️ Upload Success")
                except Exception as e:
                    print(f"❌ Upload Failed: {e}")

if __name__ == "__main__":
    train()