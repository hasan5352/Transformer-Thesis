import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax


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
    
    def forward(self, batch):   # batch = (batch_size, words, embeds)
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
        return self.attention_output_layer(concat_batch)       # (batch_size, words, embeds)


# Encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, heads, mask=False):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, heads, mask=mask)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.ff_layer1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.ff_layer2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, batch):
        # First part: attention + residual + Norm
        batch = batch + self.multi_head_attention(batch)    # (batch_size, words, embeds)
        batch = self.layer_norm1(batch)  # same size

        # second part: ANN + residual + Norm
        relu = nn.ReLU(inplace=True)
        batch1 = self.ff_layer2(relu(self.ff_layer1(batch)))   # same size
        return self.layer_norm2(batch + batch1)
    
