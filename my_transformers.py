
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import TransformerEncoder, VisionTransformerEncoder, PatchEmbedding 
from modules import AddSpecialTokensAndPositionEncoding, positional_encoding, CorruptDeitOutputHead

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
            num_encoders=2, mlp_ratio=4, dropout=0, drop_path=0,
            erase_prob=0
            ):
        assert num_classes > 0, "num_classes must be at least 1"
        super().__init__()
        self.patch_embedding = PatchEmbedding(embed_dim, patch_size, channels=channels, erase_prob=erase_prob)
        self.special_tokens_pos_encoding = AddSpecialTokensAndPositionEncoding(embed_dim, img_size, patch_size)
        self.vit_encoders = nn.Sequential(*[
            VisionTransformerEncoder(embed_dim, attention_heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path) for _ in range(num_encoders)
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


class DistillVisionTransformer(VisionTransformer):
    def __init__(
            self, embed_dim, img_size, patch_size, 
            num_classes, channels=3, attention_heads=2, 
            num_encoders=2, mlp_ratio=4, dropout=0, drop_path=0,
            erase_prob=0
            ):
        super().__init__(embed_dim, img_size, patch_size, num_classes, channels=channels, attention_heads=attention_heads,
                        num_encoders=num_encoders, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path, erase_prob=erase_prob
                        )
        self.special_tokens_pos_encoding = AddSpecialTokensAndPositionEncoding(
                        embed_dim, img_size, patch_size, distillation=True
                        )
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.distill_head = nn.Linear(embed_dim, num_classes)
    
    def process_forward(self, batch):
        """Expected input shape = (B,C,H,W)"""
        batch = self.patch_embedding(batch)       # (B, N, E)
        batch = self.special_tokens_pos_encoding(batch)     # (B, N+x, E)
        batch = self.vit_encoders(batch)

        # return cls, distill tokens. all of shape=(B, E)
        return batch[:, 0, :], batch[:, 1, :]
    
    def forward(self, batch):
        """Expected input shape = (B,C,H,W)"""
        batch, distill_tokens = self.process_forward(batch)
        batch = self.cls_head(batch)                            # (B, num_classes) cls
        distill_tokens = self.distill_head(distill_tokens)      # (B, num_classes) distill
        if self.training:
            return batch, distill_tokens    
        return (batch + distill_tokens) / 2     # during testing
        

class CorruptDistillVisionTransformer(DistillVisionTransformer):
    def __init__(
            self, embed_dim, img_size, patch_size, 
            num_classes, channels=3, attention_heads=2, 
            num_encoders=2, mlp_ratio=4, dropout=0, drop_path=0,
            erase_prob=0, num_img_types=0, head_strategy=0
            ):
        assert head_strategy > 0 and head_strategy <= 3, "allowed head_strategy range: [1,3]"
        assert num_img_types > 0, "num_img_types should be greater that 0"
        super().__init__(embed_dim, img_size, patch_size, num_classes, channels=channels, attention_heads=attention_heads,
                        num_encoders=num_encoders, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path, erase_prob=erase_prob
                        )
        self.special_tokens_pos_encoding = AddSpecialTokensAndPositionEncoding(
                        embed_dim, img_size, patch_size, distillation=True, corruption=True
                        )
        self.corrupt_head = nn.Linear(embed_dim, num_img_types)
        self.output_head = CorruptDeitOutputHead(head_strategy, num_classes, num_img_types)
        
        self.sim_cls_corr_end, self.sim_cls_distill_end = 0, 0

    def process_forward(self, batch):
        """Expected input shape = (B,C,H,W)"""
        batch = self.patch_embedding(batch)       # (B, N, E)
        batch = self.special_tokens_pos_encoding(batch)     # (B, N+x, E)
        batch = self.vit_encoders(batch)

        # return cls, distill tokens. all of shape=(B, E)
        return batch[:, 0, :], batch[:, 1, :], batch[:, 2, :]
    
    def forward(self, batch):
        """Expected input shape = (B,C,H,W)"""
        batch, distill_tokens, corrupt_tokens = self.process_forward(batch)     # all (B,E)
        
        cls_normalised = F.normalize(batch, dim=1)
        self.sim_cls_distill_end = (cls_normalised * F.normalize(distill_tokens, dim=1)).sum(dim=1).mean()
        self.sim_cls_corr_end = (cls_normalised * F.normalize(corrupt_tokens, dim=1)).sum(dim=1).mean()
        
        batch = self.cls_head(batch)                            # (B, num_classes)
        distill_tokens = self.distill_head(distill_tokens)      # (B, num_classes)
        corrupt_tokens = self.corrupt_head(corrupt_tokens)      # (B, num_types)
        return self.output_head(batch, distill_tokens, corrupt_tokens)
            

