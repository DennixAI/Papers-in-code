%%writefile inference.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os

# Import your local model definitions
from config import GemmaZeroConfig
from model import GemmaZeroModel

def load_model(weights_path="pytorch_model.bin", device="cuda"):
    """
    Loads the architecture and weights.
    """
    print(f"Loading model config and weights from {weights_path}...")
    
    # 1. Initialize the empty architecture
    config = GemmaZeroConfig()
    model = GemmaZeroModel(config)
    
    # 2. Load the weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find {weights_path}. Make sure to train first!")
    
    # Map to CPU first to avoid OOM, then move to GPU later
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Handle cases where the state_dict might be nested (e.g., if you loaded the 'full' checkpoint)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    model.load_state_dict(state_dict)
    
    # 3. Prepare for inference
    model.to(device)
    model.eval() # CRITICAL: Disables dropout and training-specific logic
    print(" Model loaded successfully!")
    return model

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_k=50, device="cuda"):
    """
    Generates text using the model.
    temperature: Higher (1.0) = Creative/Crazy, Lower (0.1) = Robot/Repetitive
    top_k: Limits choices to the top K most likely words (prevents gibberish)
    """
    # Encode prompt
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokens.input_ids

    print(f"Generated Story:\n{'-'*30}\n{prompt}", end="", flush=True)

    # Generation Loop
    for _ in range(max_new_tokens):
        with torch.no_grad(): # Disable gradient calculation for speed
            # Forward pass
            logits = model(input_ids)
            
            # Get logits for the last token only
            last_token_logits = logits[:, -1, :]
            
            # Apply Temperature (Scale logits)
            last_token_logits = last_token_logits / temperature
            
            # Apply Top-K Filtering
            if top_k > 0:
                v, _ = torch.topk(last_token_logits, top_k)
                last_token_logits[last_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities
            probs = F.softmax(last_token_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode and print the new token
            new_word = tokenizer.decode(next_token[0])
            print(new_word, end="", flush=True)
            
            # Stop if the model generates an End-of-Sequence token (optional)
            if next_token.item() == tokenizer.eos_token_id:
                break

    print(f"\n{'-'*30}")

if __name__ == "__main__":
    # Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHTS_FILE = "pytorch_model.bin" # Or "full_checkpoint.pth"
    
    # Load Tokenizer (Same as training)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load Model
    model = load_model(WEIGHTS_FILE, DEVICE)
    
    # Test Prompts
    prompts = [
        "Once upon a time, there was a little dog named",
        "The girl wanted to buy a balloon but",
        "In the dark forest, a big bear"
    ]
    
    for p in prompts:
        generate(model, tokenizer, p, max_new_tokens=150, temperature=0.7, device=DEVICE)