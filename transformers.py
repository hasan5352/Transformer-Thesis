import torch.nn as nn
from modules import TransformerEncoder, positional_encoding

class SentimentTransformer(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_classes, heads=2, num_encoders=2, mask=False, pad_token=0):
        super().__init__()
        self.num_encoders = num_encoders
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token)
        self.transformer_encoders = nn.Sequential(*[TransformerEncoder(embedding_dim, heads, mask=mask) for _ in range(num_encoders)])
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