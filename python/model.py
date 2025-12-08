"""
LiT (Limit Order Book Transformer) Model
Input: (Batch, Seq_Len, 40, 2) - 20 levels of Bid/Ask Price/Vol
Architecture: Structured Patches -> Multi-Head Self-Attention -> LSTM Head
Output: 3-class probability (Up, Down, Stationary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StructuredPatchEmbedding(nn.Module):
    """Convert order book levels into structured patches"""
    
    def __init__(self, patch_size: Tuple[int, int] = (2, 4), embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: (H/2, 4) patches from (40, 2) input
        # 40 levels / 2 = 20 patches vertically, each patch is (2, 4)
        self.patch_embed = nn.Linear(patch_size[0] * patch_size[1], embed_dim)
        self.position_embed = nn.Parameter(torch.randn(1, 20, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Seq_Len, 40, 2) - order book levels
        Returns: (B, Seq_Len, 20, embed_dim) - embedded patches
        """
        B, Seq_Len, H, W = x.shape
        
        # Reshape to create patches: (H/2, 4) = (2, 4)
        # Each patch covers 2 levels and 4 features (bid_price, bid_vol, ask_price, ask_vol)
        x = x.reshape(B, Seq_Len, H // self.patch_size[0], self.patch_size[0], W * self.patch_size[1] // W)
        
        # Flatten patch dimensions
        patches = x.reshape(B, Seq_Len, H // self.patch_size[0], -1)
        
        # Embed patches
        embedded = self.patch_embed(patches)
        
        # Add positional embedding
        embedded = embedded + self.position_embed.unsqueeze(0).unsqueeze(0)
        
        return embedded


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention for order book patterns"""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Seq_Len, Num_Patches, embed_dim)
        Returns: (B, Seq_Len, Num_Patches, embed_dim)
        """
        B, Seq_Len, N, E = x.shape
        
        # Reshape for multi-head attention
        x = x.reshape(B * Seq_Len, N, E)
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B * Seq_Len, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*Seq, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B * Seq_Len, N, E)
        out = self.proj(out)
        out = self.dropout(out)
        
        # Reshape back
        out = out.reshape(B, Seq_Len, N, E)
        
        return out


class LiTTransformer(nn.Module):
    """
    Limit Order Book Transformer
    Predicts micro-price movement probability
    """
    
    def __init__(
        self,
        seq_len: int = 100,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        lstm_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Structured patch embedding
        self.patch_embed = StructuredPatchEmbedding(patch_size=(2, 4), embed_dim=embed_dim)
        
        # Transformer blocks
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # LSTM head for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim * 20,  # 20 patches * embed_dim
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Classification head: Up, Down, Stationary
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),  # 3 classes
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Seq_Len, 40, 2) - order book levels
        Returns: (B, 3) - probability distribution [P_up, P_down, P_stationary]
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, Seq_Len, 20, embed_dim)
        
        # Apply transformer blocks
        for attn, norm in zip(self.attention_layers, self.norms):
            residual = x
            x = attn(x)
            x = norm(x + residual)
        
        # Flatten patches for LSTM
        B, Seq_Len, N, E = x.shape
        x = x.reshape(B, Seq_Len, N * E)  # (B, Seq_Len, 20*embed_dim)
        
        # LSTM forward
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (B, lstm_hidden)
        
        # Classification
        logits = self.classifier(last_hidden)  # (B, 3)
        
        return logits
    
    def predict_direction(self, x: torch.Tensor) -> Tuple[int, float]:
        """
        Predict direction and confidence
        Returns: (direction: -1/0/1, confidence: float)
        """
        with torch.inference_mode():
            probs = self.forward(x)
            direction = torch.argmax(probs, dim=-1).item() - 1  # -1, 0, 1
            confidence = probs.max().item()
            return direction, confidence


class PricePredictor(nn.Module):
    """Wrapper that outputs price prediction instead of classification"""
    
    def __init__(self, lit_model: LiTTransformer, current_price: float = 0.0):
        super().__init__()
        self.lit_model = lit_model
        self.current_price = current_price
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B, 1) - predicted price change
        """
        probs = self.lit_model(x)  # (B, 3)
        
        # Convert probabilities to price change
        # Assumes: probs[0] = up, probs[1] = down, probs[2] = stationary
        price_change = (probs[:, 0] - probs[:, 1]) * self.current_price * 0.001  # 0.1% move
        
        return price_change.unsqueeze(-1)


def create_lit_model(
    seq_len: int = 100,
    embed_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 4,
    lstm_hidden: int = 256,
    dropout: float = 0.1,
) -> LiTTransformer:
    """Factory function to create LiT model"""
    return LiTTransformer(
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Test model
    model = create_lit_model(seq_len=100)
    
    # Test input: (B=2, Seq_Len=100, 40 levels, 2 features)
    test_input = torch.randn(2, 100, 40, 2)
    
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities): {output}")
    
    direction, confidence = model.predict_direction(test_input[0:1])
    print(f"Predicted direction: {direction}, Confidence: {confidence:.4f}")

