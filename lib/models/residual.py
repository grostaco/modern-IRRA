import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)

        self.dense1 = nn.Linear(d_model, d_model * 4)
        self.dense2 = nn.Linear(d_model * 4, d_model)

        self.ln2 = nn.LayerNorm(d_model)

    
    def forward(self, inputs_embeds: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        h = self.ln1(inputs_embeds)
        h = inputs_embeds + self.attn(h, h, h, attn_mask=attn_mask)[0]

        z = self.ln2(h)
        z = self.dense1(z)
        z = F.relu(z)
        z = self.dense2(z)

        h = h + z

        return h 

class ResidualEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        
        self.d_model = d_model 
        self.num_layers = num_layers 
        
        self.residual_blocks = nn.Sequential(*[
            ResidualAttentionBlock(d_model, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.residual_blocks(x)
    