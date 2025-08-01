import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import timm
from timm.models.layers import DropPath
from utils import batch_to_patches


def positional_encoding(batch_size, num_words, embedding_dim, device):
    position = torch.arange(num_words).unsqueeze(1)             # (words, 1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))  # (embeds/2)
    
    pe = torch.zeros(num_words, embedding_dim)      # (words, embeds)
    pe[:, 0::2] = torch.sin(position * div_term)    # even indices: sin
    pe[:, 1::2] = torch.cos(position * div_term)    # odd indices: cos

    pe = pe.unsqueeze(0).repeat(batch_size, 1, 1).to(device)    # (batch_size, words, embeds)
    return pe

def apply_mask(batch):
    height, width = batch.size(-2), batch.size(-1)
    indices = torch.triu_indices(height, width, offset=1)
    batch[..., indices[0], indices[1]] = 0

# Self Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, heads, mask=False):
        super().__init__()
        assert embedding_dim % heads == 0
        self.heads = heads
        self.head_dim = embedding_dim // heads  # embeds / heads
        self.mask = mask
        
        # QKV layers
        self.Q_layers = nn.ModuleList([nn.Linear(embedding_dim, self.head_dim) for _ in range(heads)])
        self.K_layers = nn.ModuleList([nn.Linear(embedding_dim, self.head_dim) for _ in range(heads)])
        self.V_layers = nn.ModuleList([nn.Linear(embedding_dim, self.head_dim) for _ in range(heads)])
        self.attention_output_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, batch):
        """Expected input shape=(B,N+x,E) for ViT & (B,N,E) for NLP"""
        concat_batch = []              
        for i in range(self.heads):
            # a. QKV calculation         # (batch_size, words, embeds/heads)
            Q = self.Q_layers[i](batch)
            K = self.K_layers[i](batch)
            V = self.V_layers[i](batch)
            # b. (Q * Kt) and scaling by 1/(root(dk) = embeds / heads)
            QKt = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)      # (batch_size, words, words)

            # masking
            if self.mask:
                apply_mask(QKt)
            
            QKt = softmax(QKt, dim=-1)
            # c. scaled and softmaxed QKt * V to get New embeddings
            QKtV = torch.bmm(QKt, V)       # (batch_size, words, embeds/heads)
            concat_batch.append(QKtV)   
        
        # d. concat output of heads
        concat_batch = torch.cat(concat_batch, dim=-1)      # (batch_size, words, embeds)
        # e. self attention output
        return self.attention_output_layer(concat_batch)       # (batch_size, words+x, embeds)

class MultiHeadAttentionFast(nn.Module):
    def __init__(self, embed_dim, heads, mask=False):
        super().__init__()
        assert embed_dim % heads == 0
        self.heads = heads
        self.head_dim = embed_dim // heads  # embeds / heads
        self.mask = mask
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_output_layer = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, batch):
        """Expected input shape=(B,N+x,E) for ViT & (B,N,E) for NLP"""
        B, N, E = batch.shape
        batch = self.qkv(batch).reshape(B, N, 3, self.heads, self.head_dim) # multiply batch with q,k,v at once.
        q, k, v = batch.unbind(dim=2)   # each (B, T, H, D)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, N,N)
        if self.mask:
            scores = scores.masked_fill(torch.tril(torch.ones(N, N, device=scores.device)) == 0, float('-inf'))

        scores = scores.softmax(dim=-1) @ v  # (B, H, T, D)
        scores = scores.transpose(1, 2).reshape(B, N, E)
        return self.attention_output_layer(scores)       # (B, N+x, E)

# NLP Encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, heads, mlp_ratio=4, dropout=0, mask=False):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, heads, mask=mask)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_ratio * embedding_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(mlp_ratio * embedding_dim, embedding_dim)
            )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, batch):
        # First part: attention + residual + Norm
        batch = self.dropout1(self.multi_head_attention(batch)) + batch    # (batch_size, words, embeds)
        batch = self.layer_norm1(batch)  # same size

        # second part: ANN + residual + Norm
        batch = self.dropout2(self.mlp(batch)) + batch   # same size
        return self.layer_norm2(batch)

# ----------------------------------------- Computer Vision ---------------------------------------------    

# Patch embedder
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, channels=3, mask_prob=0):
        """ 
        Expects a batch of images as pytorch tensors as input and does the following: 
            - Converts images in a batch to patches. 
            - Creates embeddings for the patches.
        Args:
            embed_dim (int): Dimension of patch embeddings.
            patch_size (int): Size of each patch (assumes square).
            channels (int): Number of input image channels.
        Returns:
            Tensor of shape (B, N, embed_dim), where N is number of patches.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(channels*patch_size*patch_size, embed_dim)   # linear projection of patches
        
    def forward(self, batch):
        """Expected input shape=(B,C,H,W)"""
        B, C, H, W = batch.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, f"Image height and width {H} must be divisible by patch size {self.patch_size}"

        batch = batch_to_patches(batch, self.patch_size)    # (B, N, C * p^2) convert to patches
        return self.embedding(batch)    # (B, N, E)

# Add special tokens and positional encoding
class AddSpecialTokensAndPositionEncoding(nn.Module):
    def __init__(self, embed_dim, img_size, patch_size, distillation=False, corruption=False):
        """ 
        Expects a batch of embedded patches from the PatchEmbedding module and makes it ready for the transformer encoder: 
            - Puts a learnable [CLS] token to patch embeddings for each image.
            - If distillation=True, puts a learnable [distillation] token to patch embeddings for each image.
            - If corruption=True, puts a learnable [corruption] token to patch embeddings for each image.
            - Adds learnable positional encodings to patch embeddings for each image.
        Args:
            embed_dim (int): Dimension of patch embeddings.
            img_size (int): Height/width of input image (assumes square).
            patch_size (int): Size of each patch (assumes square).
            distillation (bool): If true, add distillation tokens for each image.
        Returns:
            Tensor of shape (B, N+x, embed_dim), where N is number of patches and x is number of 
            special tokens put for each image.
        """
        super().__init__()
        num_patches = (img_size // patch_size)**2
        x = 1
        self.distillation, self.corruption = distillation, corruption
        if self.distillation:
            self.distill_token = nn.Parameter(torch.randn(1,1,embed_dim))
            x += 1
        if self.corruption:
            self.corrupt_token = nn.Parameter(torch.randn(1,1,embed_dim))
            x += 1

        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches+x, embed_dim))
    
    def forward(self, batch):
        """Expected input shape=(B,N,E)"""
        B = batch.shape[0]
        tokens = [self.cls_token.expand(B, -1, -1)]   # copy cls tokens for all images
        if self.distillation:
            tokens += [self.distill_token.expand(B, -1, -1)]   # same for distill and corrupt
        if self.corruption:
            tokens += [self.corrupt_token.expand(B, -1, -1)]
        
        batch = torch.cat(tokens + [batch], dim=1)
        return batch + self.pos_encoding  # (B, N+x, E)

# ViT Encoder Block
class VisionTransformerEncoder(nn.Module):
    def __init__(
            self, embedding_dim, heads, mlp_ratio=4, 
            dropout=0, drop_path=0, mask=False
            ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.attn = MultiHeadAttentionFast(embedding_dim, heads, mask=mask)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_ratio * embedding_dim), 
            nn.GELU(), 
            nn.Linear(mlp_ratio * embedding_dim, embedding_dim)
            )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, batch):
        """Input shape=(B,N+x,E)"""
        # First part: Norm + attention + residual
        batch = batch + self.drop_path1(self.dropout1(self.attn(self.layer_norm1(batch))))   # (B,N+x,E)

        # second part: Norm + MLP + residual
        batch = batch + self.drop_path2(self.dropout2(self.mlp(self.layer_norm2(batch))))
        return batch    # (B,N+x,E)
    
class CorruptDeitOutputHead(nn.Module):
    def __init__(self, head_strategy, num_classes, num_img_types):
        """
        Args:
            head_strategy (int): 
                - If 1: return cls, dist, corr at train & (cls+dist)/2 at test
                - If 2: return cls, dist, corr at train & & (cls+dist+corr@W)/3 at test & corr@W at validation
                - If 3: return cls + corr@W, dist, corr at train & (cls+dist+corr@W)/3 at test
        """
        super().__init__()
        self.head_strategy = head_strategy
        if head_strategy == 2:
            self.ffn = nn.Linear(num_img_types, num_classes, bias=False)
        elif head_strategy == 3:
            self.ffn = nn.Sequential(
                nn.Linear(num_img_types, num_img_types*4, bias=False),
                nn.GELU(),
                nn.Linear(num_img_types*4, num_classes)
                )

    def forward(self, cls_tokens, distill_tokens, corrupt_tokens):
        """ Input shapes: cls & distill: (B,N), corrupt: (B,T)
        """
        return cls_tokens, distill_tokens, corrupt_tokens
        if self.training:               # training time
            if self.head_strategy == 1:
                return cls_tokens, distill_tokens, corrupt_tokens
            else:
                return (cls_tokens + self.ffn(corrupt_tokens))/2, distill_tokens, corrupt_tokens
            
        else:           # inference time
            if self.head_strategy == 1:             # calculate all this in test function, in order to analyse metrics per head
                return (cls_tokens + distill_tokens) / 2
            else:
                return (cls_tokens + distill_tokens + self.ffn(corrupt_tokens)) / 3
            

            