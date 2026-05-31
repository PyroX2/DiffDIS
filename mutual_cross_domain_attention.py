import torch
from torch import nn
import torch.nn.functional as F

class MutualCrossDomainAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Expects x of shape: [Batch, Seq_Len, Embed_Dim]
        Crucial: The batch must be structured with masks in the first half 
        and edges in the second half, e.g., torch.cat([masks, edges], dim=0)
        """
        B, N, C = x.shape
        assert B % 2 == 0, "Batch size must be even to split into masks and edges."
        half_B = B // 2

        # Split batch into mask tokens and edge tokens
        x_m, x_e = x[:half_B], x[half_B:]

        # Project and reshape for multi-head attention: [Batch, Heads, Seq_Len, Head_Dim]
        def project_and_reshape(tensor, proj_layer):
            return proj_layer(tensor).view(half_B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q_m = project_and_reshape(x_m, self.q_proj)
        k_m = project_and_reshape(x_m, self.k_proj)
        v_m = project_and_reshape(x_m, self.v_proj)

        q_e = project_and_reshape(x_e, self.q_proj)
        k_e = project_and_reshape(x_e, self.k_proj)
        v_e = project_and_reshape(x_e, self.v_proj)

        # Mutual Attention: Mask queries Edge K/V, Edge queries Mask K/V
        # Using PyTorch's optimized scaled dot-product attention
        attn_m = F.scaled_dot_product_attention(q_m, k_e, v_e)
        attn_e = F.scaled_dot_product_attention(q_e, k_m, v_m)

        # Reshape back to [Batch, Seq_Len, Embed_Dim]
        attn_m = attn_m.transpose(1, 2).reshape(half_B, N, C)
        attn_e = attn_e.transpose(1, 2).reshape(half_B, N, C)

        # Re-concatenate along batch dimension to maintain original shape
        out = torch.cat([attn_m, attn_e], dim=0)
        
        return self.out_proj(out)