"""
轻量 Encoder-Decoder Transformer
用于: YOLOv8 检测结果 token 序列 → 结构化 Score JSON token 序列

设计:
- Encoder: 接收每行 token 序列 (行内按 x 排序的几何 token)
- Decoder: 自回归生成结构化 token 流
- 规模: d_model=128, nhead=4, 2层 encoder + 2层 decoder
- 参数量: ~1M
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import VOCAB_SIZE, PAD_ID


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1), :]


class JianpuTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        # src: (B, S), tgt: (B, T)
        src_pad_mask = (src == PAD_ID)  # (B, S)
        tgt_pad_mask = (tgt == PAD_ID)  # (B, T)
        T = tgt.size(1)
        # causal mask for decoder
        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )
        return src_pad_mask, tgt_pad_mask, causal_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        # src: (B, S), tgt: (B, T)
        src_pad_mask, tgt_pad_mask, causal_mask = self._make_masks(src, tgt)

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        logits = self.fc_out(out)  # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 256,
        bos_id: int = 1,
        eos_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """自回归生成"""
        self.eval()
        B = src.size(0)
        device = src.device

        src_pad_mask = (src == PAD_ID)

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # 初始化 decoder 输入
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_pad_mask = (ys == PAD_ID)
            T = ys.size(1)
            causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool),
                diagonal=1,
            )

            tgt_emb = self.embedding(ys) * math.sqrt(self.d_model)
            tgt_emb = self.pos_decoder(tgt_emb)

            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            logits = self.fc_out(out[:, -1, :])  # (B, V)
            logits = logits / max(temperature, 1e-8)

            # top-k / top-p
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            next_token = next_token.squeeze(-1)  # (B,)

            # 强制已完成的样本保持 EOS
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            ys = torch.cat([ys, next_token.unsqueeze(-1)], dim=1)
            finished = finished | (next_token == eos_id)

            if finished.all():
                break

        return ys


if __name__ == "__main__":
    # 简单测试
    model = JianpuTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    src = torch.randint(0, VOCAB_SIZE, (2, 32))
    tgt = torch.randint(0, VOCAB_SIZE, (2, 16))
    logits = model(src, tgt)
    print(f"Logits shape: {logits.shape}")

    out = model.generate(src, max_len=20)
    print(f"Generated shape: {out.shape}")
