from __future__ import annotations

from loguru import logger
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class WordLevelMHA(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_heads: int = 2,
        qk_dim: int = 32,
        v_dim: int = 64,
        num_chunks: int = 8,
        dropout: float = 0.1,
        max_relative_distance: int = 8,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_chunks = num_chunks
        self.chunk_dim = embedding_dim // num_chunks
        
        if embedding_dim % num_chunks != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_chunks ({num_chunks})"
            )
        
        self.q_proj = nn.Linear(self.chunk_dim, qk_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(self.chunk_dim, qk_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(self.chunk_dim, v_dim * num_heads, bias=False)
        
        self.out_proj = nn.Linear(v_dim * num_heads, self.chunk_dim, bias=False)
        
        self.layer_norm = nn.LayerNorm(self.chunk_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = qk_dim ** -0.5
        
        self.relative_pos_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * num_chunks - 1)
        )
        self._init_relative_pos_bias()
        
        self.register_buffer('pos_encoding', self._create_sinusoidal_pe())
        
        self.register_buffer('rel_pos_index', self._create_relative_position_index())
    
    def _init_relative_pos_bias(self):
        with torch.no_grad():
            center = self.num_chunks - 1
            for i in range(2 * self.num_chunks - 1):
                distance = abs(i - center)
                self.relative_pos_bias.data[:, i] = -0.5 * distance
    
    def _create_sinusoidal_pe(self) -> torch.Tensor:
        pe = torch.zeros(self.num_chunks, self.chunk_dim)
        position = torch.arange(0, self.num_chunks, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.chunk_dim, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / self.chunk_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _create_relative_position_index(self) -> torch.Tensor:
        coords = torch.arange(self.num_chunks)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)
        relative_coords = relative_coords + (self.num_chunks - 1)
        return relative_coords
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(
                f"WordLevelMHA expects 2D input [batch, embedding_dim], got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )
        batch_size = x.size(0)
        if batch_size == 0:
            raise ValueError("WordLevelMHA received empty batch (batch_size=0)")
        if x.size(1) != self.embedding_dim:
            raise ValueError(
                f"WordLevelMHA embedding dimension mismatch: expected {self.embedding_dim}, got {x.size(1)}"
            )

        x_chunks = x.view(batch_size, self.num_chunks, self.chunk_dim)
        
        x_chunks = x_chunks + self.pos_encoding.unsqueeze(0)
        
        Q = self.q_proj(x_chunks)
        K = self.k_proj(x_chunks)
        V = self.v_proj(x_chunks)
        
        Q = Q.view(batch_size, self.num_chunks, self.num_heads, self.qk_dim).transpose(1, 2)
        K = K.view(batch_size, self.num_chunks, self.num_heads, self.qk_dim).transpose(1, 2)
        V = V.view(batch_size, self.num_chunks, self.num_heads, self.v_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        rel_bias = self.relative_pos_bias[:, self.rel_pos_index]
        attn_scores = attn_scores + rel_bias.unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, self.num_chunks, self.v_dim * self.num_heads
        )
        
        attn_output = self.out_proj(attn_output)
        
        attn_output = self.layer_norm(x_chunks + attn_output)
        
        output = attn_output.view(batch_size, self.embedding_dim)
        
        return output, attn_weights

class UnifiedTagClassifier(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 1024,
        pattern_dim: int = 2030,
        num_tags: int = 406,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.pattern_dim = pattern_dim
        self.num_tags = num_tags
        self.dropout_rate = dropout

        self.emb_fc1 = nn.Linear(embedding_dim, 512)
        self.emb_bn1 = nn.BatchNorm1d(512)
        self.emb_fc2 = nn.Linear(512, 384)
        self.emb_bn2 = nn.BatchNorm1d(384)

        self.pattern_fc1 = nn.Linear(pattern_dim, 256)
        self.pattern_bn1 = nn.BatchNorm1d(256)
        self.pattern_fc2 = nn.Linear(256, 128)
        self.pattern_bn2 = nn.BatchNorm1d(128)

        self.sparse_mha = WordLevelMHA(
            embedding_dim=embedding_dim,
            num_heads=2,
            qk_dim=32,
            v_dim=64,
            num_chunks=8,
            dropout=dropout * 0.2,
        )
        
        self._last_attn_weights: Optional[torch.Tensor] = None

        fusion_dim = 384 + 128 + embedding_dim
        self.fusion_fc1 = nn.Linear(fusion_dim, 768)
        self.fusion_bn1 = nn.BatchNorm1d(768)
        self.fusion_fc2 = nn.Linear(768, 384)
        self.fusion_bn2 = nn.BatchNorm1d(384)
        self.output = nn.Linear(384, num_tags)

        self.dropout = nn.Dropout(dropout)

    def _validate_inputs(self, embeddings: torch.Tensor, patterns: torch.Tensor) -> None:
        if embeddings.dim() != 2:
            raise ValueError(
                f"Expected embeddings to be 2D [batch, embedding_dim], got shape {tuple(embeddings.shape)}"
            )
        if embeddings.size(0) == 0:
            raise ValueError("UnifiedTagClassifier received empty batch (batch_size=0)")
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.size(1)}"
            )
        if patterns.dim() != 2:
            raise ValueError(
                f"Expected patterns to be 2D [batch, pattern_dim], got shape {tuple(patterns.shape)}"
            )
        if patterns.size(1) != self.pattern_dim:
            raise ValueError(
                f"Pattern dimension mismatch: expected {self.pattern_dim}, got {patterns.size(1)}. "
                f"This usually means the n-gram features were generated with a different number of tags."
            )
        if embeddings.size(0) != patterns.size(0):
            raise ValueError(
                f"Batch size mismatch: embeddings has {embeddings.size(0)}, patterns has {patterns.size(0)}"
            )

    def forward(
        self, 
        embeddings: torch.Tensor, 
        patterns: torch.Tensor, 
        return_features: bool = False, 
        return_embedding_features: bool = False
    ):
        self._validate_inputs(embeddings, patterns)
        
        emb = F.relu(self.emb_bn1(self.emb_fc1(embeddings)))
        emb = F.relu(self.emb_bn2(self.emb_fc2(emb)))
        emb_features = emb
        emb = self.dropout(emb)

        pat = F.relu(self.pattern_bn1(self.pattern_fc1(patterns)))
        pat = F.relu(self.pattern_bn2(self.pattern_fc2(pat)))
        pat = self.dropout(pat)

        attn, attn_weights = self.sparse_mha(embeddings)
        self._last_attn_weights = attn_weights
        attn = self.dropout(attn)

        combined = torch.cat([emb, pat, attn], dim=1)
        fused = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        fused = F.relu(self.fusion_bn2(self.fusion_fc2(fused)))
        fused = self.dropout(fused)

        logits = self.output(fused)
        
        if return_embedding_features:
            return logits, emb_features, fused
        if return_features:
            return logits, fused
        return logits
    
    def encode_for_faiss(self, embeddings: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        if was_training:
            logger.warning("encode_for_faiss called while model is in training mode. Switching to eval.")
            self.eval()
        
        try:
            with torch.no_grad():
                if embeddings.size(1) != self.embedding_dim:
                    raise ValueError(
                        f"Expected embedding dim {self.embedding_dim}, got {embeddings.size(1)}"
                    )
                emb = F.relu(self.emb_bn1(self.emb_fc1(embeddings)))
                emb = F.relu(self.emb_bn2(self.emb_fc2(emb)))
                return F.normalize(emb, p=2, dim=1)
        finally:
            if was_training:
                self.train()

    def get_attention_weights(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.size(1)}"
            )
        with torch.no_grad():
            _, attn_weights = self.sparse_mha(embeddings)
            return attn_weights

    def predict_with_uncertainty(
        self, 
        embeddings: torch.Tensor, 
        patterns: torch.Tensor,
        n_samples: int = 10,
        return_all_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self._validate_inputs(embeddings, patterns)
        
        was_training = self.training
        self.eval()
        
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        try:
            all_probs = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    logits = self.forward(embeddings, patterns)
                    probs = F.softmax(logits, dim=1)
                    all_probs.append(probs)
            
            stacked = torch.stack(all_probs, dim=0)
            
            mean_probs = stacked.mean(dim=0)
            uncertainty = stacked.var(dim=0)
            
            if return_all_samples:
                return mean_probs, uncertainty, stacked
            return mean_probs, uncertainty, None
            
        finally:
            if was_training:
                self.train()
            else:
                self.eval()

    def get_config(self) -> dict:
        return {
            'embedding_dim': self.embedding_dim,
            'pattern_dim': self.pattern_dim,
            'num_tags': self.num_tags,
            'dropout': self.dropout_rate,
        }

class SupConLoss(nn.Module):
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        features = F.normalize(features, dim=1)
        
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.view(-1, 1)
        mask_positives = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(batch_size, device=device)
        mask_positives = mask_positives - mask_self
        
        num_positives = mask_positives.sum(dim=1)
        
        has_positives = num_positives > 0
        if not has_positives.any():
            return torch.tensor(0.0, device=device)
        
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        exp_sim = torch.exp(sim_matrix) * (1 - mask_self)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (num_positives + 1e-8)
        
        loss = -mean_log_prob_pos[has_positives].mean()
        
        return loss
