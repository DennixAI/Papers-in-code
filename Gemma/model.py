%%writefile model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import our custom modules
from config import GemmaZeroConfig
from modules import GemmaRMSNorm, GemmaRotaryEmbedding, apply_rotary_pos_emb

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaZeroConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

        # GEMMA FEATURE: QK-Norm
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # 1. QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 2. RoPE
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. GQA Expansion
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # 4. Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)

class GemmaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False), 
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False), 
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False) 
        )
        self.mlp_gate = self.mlp[0]; self.mlp_up = self.mlp[1]; self.mlp_down = self.mlp[2]

    def forward(self, x, mask=None):
        r = x; x = self.input_layernorm(x); x = self.self_attn(x, attention_mask=mask); x = r + x
        r = x; x = self.post_attention_layernorm(x)
        gate, val = self.mlp_gate(x), self.mlp_up(x)
        # GEMMA FEATURE: GeGLU
        x = self.mlp_down(F.gelu(gate) * val)
        return r + x

class GemmaZeroModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GemmaBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # GEMMA FEATURE: Embed Scale
        self.embed_scale = math.sqrt(config.hidden_size)
        self.gradient_checkpointing = False
        
        # Initialize weights properly
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self): 
        self.gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids) * self.embed_scale
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, None, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)
        logits = torch.matmul(x, self.embed_tokens.weight.t())
        
        # GEMMA FEATURE: Final Logit Soft Capping
        if self.config.final_logit_softcapping:
             logits = torch.tanh(logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
              
        return logits