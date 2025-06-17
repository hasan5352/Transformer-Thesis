import torch.nn as nn
from modules import TransformerEncoder, VisionTransformerEncoder, PatchEmbedding 
from modules import AddSpecialTokensAndPositionEncoding, positional_encoding

class SentimentTransformer(nn.Module):
    def __init__(
            self, embedding_dim, vocab_size, 
            num_classes, heads=2, num_encoders=2, 
            mlp_ratio=4, mask=False
            ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoders = nn.Sequential(*[
            TransformerEncoder(embedding_dim, heads, mlp_ratio=mlp_ratio, mask=mask) for _ in range(num_encoders)
            ])
        self.linear_output = nn.Linear(embedding_dim, num_classes)

    def forward(self, batch):
        """
        batch.shape = (batch_size, words)
        """
        batch = self.embedding(batch)           # (batch_size, words, embeds)
        batch = batch + positional_encoding(len(batch), len(batch[0]), len(batch[0,0]), batch.device)    # same size
        batch = self.transformer_encoders(batch)

        # global avg pooling
        batch = batch.mean(dim=1)   # (batch_size, embeds)
        return self.linear_output(batch)        # (batch_size, num_classes)
    
class VisionTransformer(nn.Module):
    def __init__(
            self, embed_dim, img_size, patch_size, 
            num_classes, channels=3, attention_heads=2, 
            num_encoders=2, mlp_ratio=4, dropout=0
            ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(embed_dim, img_size,patch_size, channels=channels)
        self.special_tokens_pos_encoding = AddSpecialTokensAndPositionEncoding(embed_dim, img_size, patch_size)
        self.vit_encoders = nn.Sequential(*[
            VisionTransformerEncoder(embed_dim, attention_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_encoders)
            ])
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.Tanh(),
            nn.Linear(embed_dim, num_classes)
            )

    def forward(self, batch):
        """Expected input shape = (B,C,H,W)"""
        batch = self.patch_embedding(batch)       # (B, N, E)
        batch = self.special_tokens_pos_encoding(batch)     # (B, N+x, E)
        batch = self.vit_encoders(batch)

        # collect CLS tokens
        batch = batch[:, 0, :]  # (B, E)

        return self.mlp_head(batch)        # (batch_size, num_classes)


class DistilledVisionTransformer(VisionTransformer):
    def __init__(
            self, embed_dim, img_size, patch_size, 
            num_classes, channels=3, attention_heads=2, 
            num_encoders=2, mlp_ratio=4, dropout=0
            ):
        super().__init__()
    
    def forward(self, batch):
        """Expected input shape = (B,C,H,W)"""
