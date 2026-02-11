
from dataclasses import dataclass

@dataclass
class GemmaZeroConfig:
    # Architecture
    vocab_size: int = 50257       # GPT-2 Tokenizer (Efficient for small models)
    hidden_size: int = 768        # Embedding dimension
    intermediate_size: int = 3072 # MLP expansion (4x hidden)
    num_hidden_layers: int = 6    # Depth
    
    # Attention details (Gemma Style)
    num_attention_heads: int = 6  # 6 heads * 128 dim = 768 total
    num_key_value_heads: int = 4  # Grouped Query Attention (GQA)
    head_dim: int = 128           # Large head dimension (Gemma signature)
    
    # Context & Stability
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attn_logit_softcapping: float = 50.0  # Config only (manual impl required if not using FlashAttn)
    final_logit_softcapping: float = 30.0 # Active