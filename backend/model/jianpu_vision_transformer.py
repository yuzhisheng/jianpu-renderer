"""A compact image-to-sequence Transformer for numbered musical notation.

The implementation is inspired by the architectural ideas in TrOMR/HOMR but
is written for this project and its jianpu vocabulary.  It consumes a cropped
score row directly; no detector tokens are required.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Tuple

import torch
from torch import nn

from model.jianpu_event_vocabulary import (
    BOS_ID, BRANCHES, BRANCH_TOKENS, CTC_ID_TO_TOKEN, CTC_TOKENS, EOS_ID,
    ID_TO_TOKEN, JianpuEvent, NOTE_ID, PAD_ID, TOKEN_TO_ID,
)


@dataclass
class VisionTransformerConfig:
    image_height: int = 128
    max_width: int = 1536
    d_model: int = 192
    nhead: int = 6
    decoder_layers: int = 4
    dim_feedforward: int = 768
    dropout: float = 0.1
    max_seq_len: int = 192
    # Silver pitch-skeleton labels supervise only pitch reliably.  Branches can
    # be enabled as gold/synthetic modifier labels are added.
    active_branches: Tuple[str, ...] = ("pitch",)

    def to_dict(self) -> dict:
        return asdict(self)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


def sinusoidal_1d(length: int, dimension: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    result = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor[:result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)


class JianpuVisionTransformer(nn.Module):
    def __init__(self, config: VisionTransformerConfig | None = None):
        super().__init__()
        self.config = config or VisionTransformerConfig()
        d = self.config.d_model
        self.visual = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128),
            ConvBlock(128, d),
        )
        self.branch_embeddings = nn.ModuleDict({
            branch: nn.Embedding(len(tokens), d, padding_idx=0)
            for branch, tokens in BRANCH_TOKENS.items()
        })
        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, self.config.decoder_layers)
        self.decoder_norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({
            branch: nn.Linear(d, len(tokens)) for branch, tokens in BRANCH_TOKENS.items()
        })
        ctc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout, activation="gelu", batch_first=True,
        )
        self.ctc_encoder = nn.TransformerEncoder(
            ctc_layer, max(1, self.config.decoder_layers // 2), enable_nested_tensor=False,
        )
        self.ctc_head = nn.Linear(d, len(CTC_TOKENS))

    def encode_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        batch, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1)
        h_dim = self.config.d_model // 2
        w_dim = self.config.d_model - h_dim
        pos_h = sinusoidal_1d(height, h_dim, features.device, features.dtype)
        pos_w = sinusoidal_1d(width, w_dim, features.device, features.dtype)
        position = torch.cat([
            pos_h[:, None, :].expand(height, width, h_dim),
            pos_w[None, :, :].expand(height, width, w_dim),
        ], dim=-1)
        return features + position

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode_feature_map(images)
        return features.reshape(features.shape[0], -1, features.shape[-1])

    def ctc_logits(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode_feature_map(images).mean(dim=1)
        return self.ctc_head(self.ctc_encoder(features))

    @torch.no_grad()
    def generate_ctc(
        self, images: torch.Tensor, input_lengths: torch.Tensor | None = None,
        beam_width: int = 1, token_bonus: float = 0.0,
    ) -> List[List[str]]:
        self.eval()
        logits = self.ctc_logits(images)
        ids = logits.argmax(-1)
        if input_lengths is None:
            input_lengths = torch.full(
                (images.shape[0],), ids.shape[1], dtype=torch.long, device=ids.device,
            )
        output = []
        for row_index, (row, length) in enumerate(zip(ids, input_lengths)):
            if beam_width > 1:
                token_ids = ctc_prefix_beam_search(
                    logits[row_index, :int(length)].log_softmax(-1).float().cpu(),
                    beam_width, token_bonus,
                )
                output.append([CTC_ID_TO_TOKEN[value] for value in token_ids])
                continue
            previous = 0
            tokens = []
            for value in row[:int(length)].tolist():
                if value and value != previous:
                    tokens.append(CTC_ID_TO_TOKEN[value])
                previous = value
            output.append(tokens)
        return output

    def decode(self, memory: torch.Tensor, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        kinds = inputs["kind"]
        hidden = sum(self.branch_embeddings[branch](inputs[branch]) for branch in BRANCHES)
        hidden = hidden + sinusoidal_1d(
            hidden.shape[1], self.config.d_model, hidden.device, hidden.dtype,
        ).unsqueeze(0)
        length = hidden.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), 1)
        decoded = self.decoder(
            hidden, memory, tgt_mask=causal, tgt_key_padding_mask=kinds.eq(PAD_ID),
        )
        decoded = self.decoder_norm(decoded)
        return {branch: self.heads[branch](decoded) for branch in BRANCHES}

    def forward(self, images: torch.Tensor, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.decode(self.encode_image(images), inputs)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int | None = None) -> List[List[JianpuEvent]]:
        self.eval()
        memory = self.encode_image(images)
        batch = images.shape[0]
        inputs = {
            branch: torch.full(
                (batch, 1), BOS_ID if branch == "kind" else TOKEN_TO_ID[branch]["NONE"],
                dtype=torch.long, device=images.device,
            ) for branch in BRANCHES
        }
        completed = torch.zeros(batch, dtype=torch.bool, device=images.device)
        results: List[List[JianpuEvent]] = [[] for _ in range(batch)]
        for _ in range((max_len or self.config.max_seq_len) - 1):
            logits = self.decode(memory, inputs)
            next_ids = {branch: value[:, -1].argmax(-1) for branch, value in logits.items()}
            for index in range(batch):
                if completed[index]:
                    continue
                kind_id = int(next_ids["kind"][index])
                if kind_id == EOS_ID:
                    completed[index] = True
                    continue
                if kind_id in {PAD_ID, BOS_ID}:
                    kind_id = TOKEN_TO_ID["kind"]["UNKNOWN"]
                kind = ID_TO_TOKEN["kind"][kind_id]
                event = JianpuEvent(
                    kind=kind,
                    pitch=ID_TO_TOKEN["pitch"][int(next_ids["pitch"][index])]
                    if kind_id == NOTE_ID and "pitch" in self.config.active_branches else "NONE",
                    accidental=ID_TO_TOKEN["accidental"][int(next_ids["accidental"][index])]
                    if kind_id == NOTE_ID and "accidental" in self.config.active_branches else "NONE",
                    octave=ID_TO_TOKEN["octave"][int(next_ids["octave"][index])]
                    if kind_id == NOTE_ID and "octave" in self.config.active_branches else "NONE",
                    duration=ID_TO_TOKEN["duration"][int(next_ids["duration"][index])]
                    if kind_id == NOTE_ID and "duration" in self.config.active_branches else "NONE",
                    articulation=ID_TO_TOKEN["articulation"][int(next_ids["articulation"][index])]
                    if kind_id == NOTE_ID and "articulation" in self.config.active_branches else "NONE",
                )
                results[index].append(event)
            for branch in BRANCHES:
                neutral = TOKEN_TO_ID[branch]["NONE"] if branch != "kind" else EOS_ID
                values = torch.where(completed, torch.full_like(next_ids[branch], neutral), next_ids[branch])
                inputs[branch] = torch.cat([inputs[branch], values[:, None]], dim=1)
            if bool(completed.all()):
                break
        return results

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _logadd(*values: float) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor, beam_width: int = 10, token_bonus: float = 0.0,
) -> List[int]:
    """Small dependency-free CTC prefix beam search (blank id is zero)."""
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}
    for timestep in log_probs.tolist():
        candidates: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for prefix, (blank_score, symbol_score) in beams.items():
            old_blank, old_symbol = candidates.get(prefix, (-math.inf, -math.inf))
            candidates[prefix] = (
                _logadd(old_blank, blank_score + timestep[0], symbol_score + timestep[0]),
                old_symbol,
            )
            for token in range(1, len(timestep)):
                probability = timestep[token]
                if prefix and token == prefix[-1]:
                    pb, ps = candidates.get(prefix, (-math.inf, -math.inf))
                    candidates[prefix] = (pb, _logadd(ps, symbol_score + probability))
                    extended = prefix + (token,)
                    epb, eps = candidates.get(extended, (-math.inf, -math.inf))
                    candidates[extended] = (
                        epb, _logadd(eps, blank_score + probability + token_bonus),
                    )
                else:
                    extended = prefix + (token,)
                    epb, eps = candidates.get(extended, (-math.inf, -math.inf))
                    candidates[extended] = (
                        epb, _logadd(
                            eps, blank_score + probability + token_bonus,
                            symbol_score + probability + token_bonus,
                        ),
                    )
        beams = dict(sorted(
            candidates.items(), key=lambda item: _logadd(*item[1]), reverse=True,
        )[:beam_width])
    return list(max(beams.items(), key=lambda item: _logadd(*item[1]))[0])
