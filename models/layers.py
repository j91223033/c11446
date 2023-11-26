import torch.nn as nn
import torch

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x.clone()
        x = self.w_1(x) 
        x = self.act(x)
        x = self.dropout1(x)
        x = self.w_2(x) 
        x = self.dropout2(x) 
        return self.ln(x + residual)

class CrossFFN(nn.Module):
    def __init__(self, n_head=8, d_model=768, d_hidden=2048, dropout=0.1, use_ffn=True):
        super(CrossFFN, self).__init__()
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head)
        self.dropout_cross = nn.Dropout(dropout)
        self.ln_cross = nn.LayerNorm(d_model)
        # Feed-forward
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = FeedForwardLayer(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
    
    def forward(self, x, y):
        x = x.transpose(0,1)
        y = y.transpose(0,1)
        # Compute cross-attention
        residual = x.clone()
        x, _ = self.cross_attn(x, y, y)
        x = self.dropout_cross(x)
        x = self.ln_cross(x + residual)
        # Compute feed-forward
        if self.use_ffn:
            x = self.ffn(x)    

        return x.transpose(0,1)

