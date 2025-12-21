import torch
import torch.nn as nn
import math
from config import GemmaZeroConfig
from modules import GemmaRMSNorm, GemmaAttention, GemmaMLP

class GemmaBlock(nn.Module):
    def __init__(self, config: GemmaZeroConfig):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class GemmaZeroModel(nn.Module):
    def __init__(self, config: GemmaZeroConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GemmaBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_scale = math.sqrt(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids) * self.embed_scale
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        x = self.norm(x)
        logits = torch.matmul(x, self.embed_tokens.weight.t())
        
        if self.config.final_logit_softcapping is not None:
             logits = torch.tanh(logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
             
        return logits