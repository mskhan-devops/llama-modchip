#!/usr/bin/env python3
"""
eakv_quantizer.py — Standalone Analytical Simulator for Entropy-Guided Asymmetric KV Quantization
=================================================================================================
NOTE: This file is an offline analytical simulator used to evaluate per-cell Q8_0/Q4_0/FP16
quantization error prior to production deployment.
In the active production llama-server path, asymmetric KV quantization is applied natively via
llama.cpp's `--cache-type-k q8_0 --cache-type-v q4_0` flags combined with the `_MM_HINT_NTA`
L3 isolation patch in `patches/llama-modchip-v3.2.patch`.
"""

import math
import struct
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# FNV-1a 64-bit Constants
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
MASK_64 = 0xffffffffffffffff

def fnv1a_hash(tokens: List[int]) -> int:
    h = FNV_OFFSET
    for t in tokens:
        raw = struct.pack("i", t)
        for b in raw:
            h = ((h ^ b) * FNV_PRIME) & MASK_64
    return h

# Precomputed 4-bit unpacking lookup table (256 bytes -> 2 signed float32 values)
_Q4_UNPACK_LUT = np.empty((256, 2), dtype=np.float32)
for _b in range(256):
    _low = _b & 0x0F
    if _low >= 8:
        _low -= 16
    _high = (_b >> 4) & 0x0F
    if _high >= 8:
        _high -= 16
    _Q4_UNPACK_LUT[_b, 0] = _low
    _Q4_UNPACK_LUT[_b, 1] = _high

class EAKVCell:
    """
    Dual-format KV memory cell representing a single token's key and value projections.
    Maps directly to `struct llama_kv_cell_rc` in C++.
    """
    __slots__ = ("token_id", "pos", "precision", "k_fp16", "v_fp16", "k_q4_bytes", "v_q4_bytes", "k_scale", "v_scale", "entropy")
    
    def __init__(
        self,
        token_id: int,
        pos: int,
        precision: str,
        k_fp16: Optional[np.ndarray],
        v_fp16: Optional[np.ndarray],
        entropy: float,
        k_q4_bytes: Optional[bytes] = None,
        v_q4_bytes: Optional[bytes] = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        self.token_id = token_id
        self.pos = pos              # Virtual Positional Index (original coordinate m)
        self.precision = precision  # 'FP16' or 'Q4_0'
        self.entropy = entropy
        
        if precision == 'FP16':
            self.k_fp16 = k_fp16.astype(np.float16) if (k_fp16 is not None and k_fp16.dtype != np.float16) else k_fp16
            self.v_fp16 = v_fp16.astype(np.float16) if (v_fp16 is not None and v_fp16.dtype != np.float16) else v_fp16
            self.k_q4_bytes = None
            self.v_q4_bytes = None
            self.k_scale = 1.0
            self.v_scale = 1.0
        else: # Q4_0
            if k_q4_bytes is not None and v_q4_bytes is not None:
                self.k_q4_bytes = k_q4_bytes
                self.v_q4_bytes = v_q4_bytes
                self.k_scale = k_scale
                self.v_scale = v_scale
            else:
                self.k_q4_bytes, self.k_scale = self._quantize_q4_0(k_fp16)
                self.v_q4_bytes, self.v_scale = self._quantize_q4_0(v_fp16)
            self.k_fp16 = None
            self.v_fp16 = None

    @staticmethod
    def _quantize_q4_0(vec: np.ndarray) -> Tuple[bytes, float]:
        """Quantizes a 128-dim float vector into 64 bytes of packed 4-bit signed integers + 1 float scale."""
        max_val = np.max(np.abs(vec))
        if max_val == 0.0:
            scale = 1.0
        else:
            scale = float(max_val) / 7.0  # Signed 4-bit range [-7, +7]
        
        scaled = np.clip(np.round(vec / scale), -7, 7).astype(np.int8)
        # Pack pairs of 4-bit nibbles into single bytes
        low = scaled[0::2] & 0x0F
        high = (scaled[1::2] & 0x0F) << 4
        packed = (low | high).astype(np.uint8)
        return packed.tobytes(), scale

    def dequantize_k(self) -> np.ndarray:
        """Dequantizes Key vector back to float32 for attention calculation."""
        if self.precision == 'FP16':
            return self.k_fp16.astype(np.float32)
        raw = np.frombuffer(self.k_q4_bytes, dtype=np.uint8)
        return _Q4_UNPACK_LUT[raw].reshape(-1) * self.k_scale

    def dequantize_v(self) -> np.ndarray:
        """Dequantizes Value vector back to float32 for attention aggregation."""
        if self.precision == 'FP16':
            return self.v_fp16.astype(np.float32)
        raw = np.frombuffer(self.v_q4_bytes, dtype=np.uint8)
        return _Q4_UNPACK_LUT[raw].reshape(-1) * self.v_scale

    def byte_size(self) -> int:
        """Returns physical memory consumption in bytes."""
        # 128 dimensions * 2 (K & V)
        if self.precision == 'FP16':
            return 128 * 2 * 2 + 8  # 512 bytes + 8 bytes header per head
        else:
            return 64 * 2 + 8 + 8   # 128 bytes + 8 bytes scales + 8 bytes header = 144 bytes per head

class EAKVEngine:
    """
    Entropy-Guided Asymmetric KV Quantization & Attention Execution Engine.
    Employs RouteCrack 6-level FNV-64 hash cascades, Lookahead Guard, and VPI RoPE.
    """
    HEAD_DIM = 128
    ROPE_BASE = 1000000.0  # RoPE theta base for Qwen models
    LOOKAHEAD_GUARD_TOKENS = 2  # 2 tokens before transition promoted to FP16
    
    def __init__(self, n_layers: int = 48, n_kv_heads: int = 8):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        # Storage: layer_idx -> head_idx -> List[EAKVCell]
        self.cells: List[List[List[EAKVCell]]] = [
            [[] for _ in range(n_kv_heads)] for _ in range(n_layers)
        ]
        # Precomputed frequency angles for RoPE (head_dim=128)
        dim_indices = np.arange(0, self.HEAD_DIM, 2)
        self.inv_freq = 1.0 / (self.ROPE_BASE ** (dim_indices / self.HEAD_DIM))

    def apply_rope(self, vec: np.ndarray, pos: int) -> np.ndarray:
        """Applies Rotary Position Embedding (RoPE) at absolute virtual position pos (supports 1D or ND)."""
        angles = pos * self.inv_freq
        cos = np.cos(angles)
        sin = np.sin(angles)
        
        # Split into even and odd components along last axis
        v_even = vec[..., 0::2]
        v_odd  = vec[..., 1::2]
        
        rot_even = v_even * cos - v_odd * sin
        rot_odd  = v_even * sin + v_odd * cos
        
        out = np.empty_like(vec)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out

    def classify_entropy_stream(self, tokens: List[int]) -> List[Tuple[float, str]]:
        """
        Classifies incoming tokens using RouteCrack zero-FLOP heuristics
        and applies the 2-Token Lookahead Guard Band.
        Returns: List of (entropy_score, precision_label)
        """
        n = len(tokens)
        raw_classes = []
        
        # 1. First pass: RouteCrack hash cascade confidence
        # Simulated statistical predictability: tokens with repeating structural patterns have low entropy
        for i, t in enumerate(tokens):
            window = tokens[max(0, i-4) : i+1]
            h = fnv1a_hash(window)
            # Determinist syntactic pattern if hash satisfies deterministic modulo or common ASCII range
            # Token < 500: common C syntax, punctuation, keywords, braces
            if t < 500 or (h % 10 < 7):
                entropy = 0.15 # Low entropy / deterministic
                precision = 'Q4_0'
            else:
                entropy = 0.85 # High entropy / semantic decision (e.g. hex offset)
                precision = 'FP16'
            raw_classes.append((entropy, precision))
            
        # 2. Second pass: Apply 2-Token Lookahead Guard
        # If precision transitions from Q4_0 -> FP16, promote the 2 preceding tokens to FP16
        final_classes = list(raw_classes)
        for i in range(n - 1):
            if final_classes[i][1] == 'Q4_0' and final_classes[i+1][1] == 'FP16':
                for guard_idx in range(max(0, i - self.LOOKAHEAD_GUARD_TOKENS + 1), i + 1):
                    final_classes[guard_idx] = (final_classes[guard_idx][0], 'FP16')
                    
        return final_classes

    @staticmethod
    def _quantize_q4_0_batch(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized batch quantization of shape (..., 128) into packed uint8 nibbles and scales."""
        max_val = np.max(np.abs(tensor), axis=-1)
        scale = np.where(max_val == 0.0, 1.0, max_val / 7.0)
        scaled = np.clip(np.round(tensor / scale[..., np.newaxis]), -7, 7).astype(np.int8)
        low = scaled[..., 0::2] & 0x0F
        high = (scaled[..., 1::2] & 0x0F) << 4
        packed = (low | high).astype(np.uint8)
        return packed, scale

    def ingest_token(self, token_id: int, pos: int, precision: str, entropy: float, layer_k_projections: np.ndarray, layer_v_projections: np.ndarray):
        """
        Ingests a token into all layers and KV heads with specified precision and virtual position pos.
        Uses vectorized RoPE rotation and batched Q4_0 quantization across all (layers, heads).
        """
        # Vectorized RoPE rotation across all (layers, heads) in a single tensor operation
        k_rot_all = self.apply_rope(layer_k_projections, pos=pos)
        
        if precision == 'FP16':
            k_fp16_all = k_rot_all.astype(np.float16)
            v_fp16_all = layer_v_projections.astype(np.float16)
            for l in range(self.n_layers):
                for h in range(self.n_kv_heads):
                    cell = EAKVCell(
                        token_id=token_id,
                        pos=pos,
                        precision='FP16',
                        k_fp16=k_fp16_all[l, h],
                        v_fp16=v_fp16_all[l, h],
                        entropy=entropy,
                    )
                    self.cells[l][h].append(cell)
        else: # Q4_0: Batched vectorized quantization across all layers and heads
            k_packed, k_scales = self._quantize_q4_0_batch(k_rot_all)
            v_packed, v_scales = self._quantize_q4_0_batch(layer_v_projections)
            
            for l in range(self.n_layers):
                for h in range(self.n_kv_heads):
                    cell = EAKVCell(
                        token_id=token_id,
                        pos=pos,
                        precision='Q4_0',
                        k_fp16=None,
                        v_fp16=None,
                        entropy=entropy,
                        k_q4_bytes=k_packed[l, h].tobytes(),
                        v_q4_bytes=v_packed[l, h].tobytes(),
                        k_scale=float(k_scales[l, h]),
                        v_scale=float(v_scales[l, h]),
                    )
                    self.cells[l][h].append(cell)

    def compute_attention(self, layer_idx: int, head_idx: int, query_vec: np.ndarray, query_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes attention output and attention weights against the EAKV Cache using VPI RoPE.
        """
        # Apply RoPE to query vector at position query_pos
        q_rot = self.apply_rope(query_vec, pos=query_pos)
        
        cells = self.cells[layer_idx][head_idx]
        seq_len = len(cells)
        if seq_len == 0:
            return np.zeros(self.HEAD_DIM, dtype=np.float32), np.array([])
        
        # Dequantize Keys and compute dot products
        scores = np.empty(seq_len, dtype=np.float32)
        v_matrix = np.empty((seq_len, self.HEAD_DIM), dtype=np.float32)
        
        for i, c in enumerate(cells):
            k_vec = c.dequantize_k()
            v_vec = c.dequantize_v()
            v_matrix[i] = v_vec
            scores[i] = np.dot(q_rot, k_vec) / math.sqrt(self.HEAD_DIM)
            
        # Softmax
        scores_max = np.max(scores)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores)
        
        # Weighted sum of Values
        out_vec = np.dot(weights, v_matrix)
        return out_vec, weights

    def memory_footprint_bytes(self) -> int:
        """Returns total memory consumed by all cells across all layers and heads."""
        total = 0
        for l in range(self.n_layers):
            for h in range(self.n_kv_heads):
                for c in self.cells[l][h]:
                    total += c.byte_size()
        return total

if __name__ == "__main__":
    print("[*] EAKV Engine Module loaded successfully.")
