from dataclasses import dataclass

@dataclass
class GemmaZeroConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0