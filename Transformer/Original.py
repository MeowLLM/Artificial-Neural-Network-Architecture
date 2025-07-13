import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class RotaryPositionalEmbedding(nn.Module):
    # This is a simplified RoPE, full DeepSeek RoPE might be more complex
    def __init__(self, dim, max_position_embeddings=512, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_position_embeddings ({self.max_position_embeddings}).")

        if self.seq_len_cached is None or seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :].to(x.dtype) # (1, 1, seq_len, dim)
            self.sin_cached = emb.sin()[None, None, :, :].to(x.dtype) # (1, 1, seq_len, dim)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, position_ids=None):
        # x: (batch_size, num_heads, seq_len, head_dim) or (batch_size, seq_len, head_dim)
        cos, sin = self._update_cos_sin_tables(x, x.shape[-2])
        # Apply RoPE directly
        x_rope = (x * cos) + (self.rotate_half(x) * sin)
        return x_rope

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent_kv, dropout_rate=0.1, rope_dim=64):
        super(MultiHeadLatentAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_latent_kv = d_latent_kv # d_c in DeepSeek's paper
        self.rope_dim = rope_dim # Dimension for Rotary Positional Embedding

        # Query projection (standard)
        self.wq = nn.Linear(d_model, d_model)

        # DeepSeek MLA specific projections for KV compression
        # h_t -> c_t_KV
        self.w_down_kv = nn.Linear(d_model, d_latent_kv)
        # c_t_KV -> K, V (for attention computation)
        self.w_up_k = nn.Linear(d_latent_kv, d_model)
        self.w_up_v = nn.Linear(d_latent_kv, d_model)

        # Final output projection
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # RoPE for decoupled keys (simplified)
        self.rope = RotaryPositionalEmbedding(rope_dim)
        # Additional projection for the RoPE-enabled part of the key
        self.w_k_rope = nn.Linear(d_model, rope_dim * num_heads) # Project to (num_heads * rope_dim)

    def forward(self, query_input, kv_input, mask=None, past_key_values=None):
        # query_input: (batch_size, seq_len, d_model)
        # kv_input: (batch_size, seq_len_kv, d_model) - for self-attention, seq_len_kv == seq_len

        batch_size, query_seq_len, _ = query_input.size()
        kv_seq_len = kv_input.size(1)

        # 1. Project Query, and compressed Latent KV
        q = self.wq(query_input).view(batch_size, query_seq_len, self.num_heads, self.d_k).transpose(1, 2) # (B, H, Lq, Dk)

        # MLA: Compress KV input to latent space
        c_kv = self.w_down_kv(kv_input) # (B, Lkv, d_latent_kv)

        # 2. Handle KV caching (crucial for MLA's memory benefits)
        # In a real inference loop, past_key_values would contain cached c_kv.
        # For simplicity here, we recompute c_kv for the whole sequence.
        # For actual decoding, you'd append current c_kv to past_c_kv and pass that.
        # This implementation assumes full sequence input for training/initial pass.

        # 3. Decompress latent KV to full K and V
        k = self.w_up_k(c_kv).view(batch_size, kv_seq_len, self.num_heads, self.d_k).transpose(1, 2) # (B, H, Lkv, Dk)
        v = self.w_up_v(c_kv).view(batch_size, kv_seq_len, self.num_heads, self.d_k).transpose(1, 2) # (B, H, Lkv, Dk)

        # 4. Decoupled RoPE for a portion of the key (simplified)
        # This is where DeepSeek's RoPE handling gets more intricate.
        # We'll apply RoPE to a *part* of the key dimension, as suggested.
        # For full fidelity, you'd need to properly split and combine.
        k_rope_input = self.w_k_rope(kv_input).view(batch_size, kv_seq_len, self.num_heads, self.rope_dim).transpose(1, 2)
        k_rope_output = self.rope(k_rope_input)

        # Combine RoPE part with the main key (simplified placeholder)
        # A full implementation would carefully integrate k_rope_output into 'k'
        # For demonstration, let's just use the RoPE part for the first `rope_dim` of K for each head
        k[:, :, :, :self.rope_dim] = k_rope_output

        # 5. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask needs to be broadcastable to (B, H, Lq, Lkv)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.num_heads * self.d_k)
        output = self.wo(context)
        return output, attention_weights

class DecoderBlockMLA(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, d_latent_kv, dropout_rate=0.1, rope_dim=64):
        super(DecoderBlockMLA, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, num_heads, d_latent_kv, dropout_rate, rope_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, tgt_mask):
        # Masked Multi-Head Latent Attention (MLA)
        normed_x = self.norm1(x)
        # In self-attention, query_input and kv_input are both normed_x
        attn_output, _ = self.attn(normed_x, normed_x, mask=tgt_mask)
        x = x + self.dropout1(attn_output)

        # Feed Forward
        normed_x = self.norm2(x)
        ff_output = self.ff(normed_x)
        x = x + self.dropout2(ff_output)
        return x

class TransformerDecoderOnlyMLA(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, d_latent_kv, dropout_rate=0.1, max_len=5000, rope_dim=64):
        super(TransformerDecoderOnlyMLA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len) # Standard PE, or consider RoPE integrated earlier
        self.dropout = nn.Dropout(dropout_rate)

        self.decoder_layers = nn.ModuleList([
            DecoderBlockMLA(d_model, num_heads, d_ff, d_latent_kv, dropout_rate, rope_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.unembedding = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        seq_len = src.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)

        x = self.embedding(src)
        x = self.positional_encoding(x) # You might integrate RoPE directly into MLA instead of this standard PE
        x = self.dropout(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, tgt_mask)

        x = self.final_norm(x)
        output_logits = self.unembedding(x)
        return output_logits
