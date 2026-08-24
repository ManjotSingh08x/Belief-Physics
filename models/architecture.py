import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, is_causal=True):
        seq_len = x.size(1)
        attn_mask = None
        if is_causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=x.device
            )
            
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask, is_causal=is_causal
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class ContinuousTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, input_dim)
        
    def forward(self, x):
        b, t, _ = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        
        x = self.input_proj(x) + self.pos_embed(pos)
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        return self.output_proj(x)


class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
