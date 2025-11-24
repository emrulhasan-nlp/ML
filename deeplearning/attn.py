import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim) or (seq_len, input_dim)
        """
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        d_k = K.size(-1)
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    

if __name__ == "__main__":
    # Example usage
    batch_size = 8
    seq_len = 5
    input_dim = 10
    output_dim = 8

    x = torch.randn(batch_size, seq_len, input_dim)
    attn = Attention(input_dim, output_dim)
    output, weights = attn(x)

    print(output[0])  # Output shape
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, output_dim)
    print("Attention weights shape:", weights.shape)  # Expected: (batch_size, seq_len, seq_len)
