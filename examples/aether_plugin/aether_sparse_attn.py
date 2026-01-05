# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AETHER Sparse Attention Kernel for TRT Plugin Injection
========================================================

This is the source Triton kernel for AOT compilation to PTX.
Designed for integration with TensorRT-LLM via custom plugin.

Reference:
    Sharma, T. (2024). AETHER - Adaptive Event-driven Threshold Hybrid
    Entangled Rendering. DOI: 10.13141/RG.2.2.14811.27684
"""

import torch
import triton
import triton.language as tl


@triton.jit
def aether_sparse_attention_kernel(
    # Input tensors
    Q_ptr, K_ptr, V_ptr,
    # Precomputed metadata
    Mean_ptr, Rad_ptr, Conc_ptr,
    # Output tensor
    Out_ptr,
    # Q strides (B, H, D)
    stride_q_b, stride_q_h, stride_q_d,
    # K strides (B, H, S, D)
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    # V strides (B, H, S, D)
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    # Mean strides (B, H, N_blocks, D)
    stride_m_b, stride_m_h, stride_m_blk, stride_m_d,
    # Radius strides (B, H, N_blocks)
    stride_r_b, stride_r_h, stride_r_blk,
    # Concentration strides (B, H, N_blocks)
    stride_c_b, stride_c_h, stride_c_blk,
    # Out strides (B, H, D)
    stride_o_b, stride_o_h, stride_o_d,
    # Dimensions
    batch_size, num_heads, seq_len,
    # Constexpr config
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    SCALE: tl.constexpr,
    THRESHOLD: tl.constexpr,
    USE_CONCENTRATION: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
):
    """
    Fused block-sparse attention kernel with AETHER scoring.
    
    This kernel:
    1. Scores all KV blocks using attention potential (center + deviation)
    2. Selects blocks above threshold (or local window for causal)
    3. Computes attention only over selected blocks
    4. Uses online softmax for numerical stability
    
    Grid: (batch_size, num_heads, 1)
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    dim_offsets = tl.arange(0, HEAD_DIM)
    
    # Load query vector
    q_offset = batch_idx * stride_q_b + head_idx * stride_q_h
    q = tl.load(Q_ptr + q_offset + dim_offsets * stride_q_d).to(tl.float32)
    q_norm = tl.sqrt(tl.sum(q * q) + 1e-8)
    
    # Initialize online softmax accumulators
    m_i = tl.zeros([1], dtype=tl.float32) - 1e9  # Max score
    l_i = tl.zeros([1], dtype=tl.float32)        # Sum of exp
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32) # Output accumulator
    
    # Process each block
    for block_idx in range(N_BLOCKS):
        # Load block metadata
        m_offset = batch_idx * stride_m_b + head_idx * stride_m_h + block_idx * stride_m_blk
        mean = tl.load(Mean_ptr + m_offset + dim_offsets * stride_m_d).to(tl.float32)
        
        r_offset = batch_idx * stride_r_b + head_idx * stride_r_h + block_idx * stride_r_blk
        radius = tl.load(Rad_ptr + r_offset).to(tl.float32)
        
        # Compute attention potential (scoring)
        center_score = tl.sum(q * mean) * SCALE
        deviation = q_norm * radius * SCALE
        
        if USE_CONCENTRATION:
            c_offset = batch_idx * stride_c_b + head_idx * stride_c_h + block_idx * stride_c_blk
            concentration = tl.load(Conc_ptr + c_offset).to(tl.float32)
            # Safety clamp: maps [0.1, 1.0] -> [0.37, 1.0]
            concentration_factor = 0.3 + (0.7 * concentration)
            deviation = deviation * concentration_factor
        
        potential = center_score + deviation
        
        # Determine if block should be attended
        if IS_CAUSAL:
            is_local = block_idx >= (N_BLOCKS - LOCAL_WINDOW)
            is_sink = (block_idx == 0)  # Attention sink
            should_attend = (potential > THRESHOLD) | is_local | is_sink
        else:
            should_attend = potential > THRESHOLD
        
        # Process block if selected
        if should_attend:
            for token_offset in range(BLOCK_SIZE):
                token_idx = block_idx * BLOCK_SIZE + token_offset
                
                # Bounds check
                if token_idx < seq_len:
                    # Load key
                    k_offset = batch_idx * stride_k_b + head_idx * stride_k_h + token_idx * stride_k_s
                    key = tl.load(K_ptr + k_offset + dim_offsets * stride_k_d).to(tl.float32)
                    
                    # Compute attention score
                    attn_score = tl.sum(q * key) * SCALE
                    
                    # Online softmax update
                    m_new = tl.maximum(m_i, attn_score)
                    alpha = tl.exp(m_i - m_new)
                    beta = tl.exp(attn_score - m_new)
                    l_i = alpha * l_i + beta
                    
                    # Load value and accumulate
                    v_offset = batch_idx * stride_v_b + head_idx * stride_v_h + token_idx * stride_v_s
                    value = tl.load(V_ptr + v_offset + dim_offsets * stride_v_d).to(tl.float32)
                    acc = alpha * acc + beta * value
                    
                    m_i = m_new
    
    # Normalize output
    out = acc / (l_i + 1e-8)
    
    # Store result
    o_offset = batch_idx * stride_o_b + head_idx * stride_o_h
    tl.store(Out_ptr + o_offset + dim_offsets * stride_o_d, out.to(Out_ptr.dtype.element_ty))


@triton.jit
def aether_block_score_kernel(
    # Input
    Q_ptr, Mean_ptr, Rad_ptr, Conc_ptr,
    # Output
    Score_ptr, Mask_ptr,
    # Strides
    stride_q_b, stride_q_h, stride_q_d,
    stride_m_b, stride_m_h, stride_m_blk, stride_m_d,
    stride_r_b, stride_r_h, stride_r_blk,
    stride_c_b, stride_c_h, stride_c_blk,
    stride_s_b, stride_s_h, stride_s_blk,
    stride_mask_b, stride_mask_h, stride_mask_blk,
    # Config
    HEAD_DIM: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    SCALE: tl.constexpr,
    THRESHOLD: tl.constexpr,
    USE_CONCENTRATION: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
):
    """
    Standalone block scoring kernel for debugging/analysis.
    
    Outputs per-block scores and binary mask.
    """
    block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    dim_offsets = tl.arange(0, HEAD_DIM)
    
    # Load query
    q = tl.load(Q_ptr + bh_idx * stride_q_h + dim_offsets * stride_q_d).to(tl.float32)
    q_norm = tl.sqrt(tl.sum(q * q) + 1e-8)
    
    # Load metadata
    mean = tl.load(Mean_ptr + bh_idx * stride_m_h + block_idx * stride_m_blk + dim_offsets * stride_m_d).to(tl.float32)
    radius = tl.load(Rad_ptr + bh_idx * stride_r_h + block_idx * stride_r_blk).to(tl.float32)
    
    # Compute score
    center_score = tl.sum(q * mean) * SCALE
    deviation = q_norm * radius * SCALE
    
    if USE_CONCENTRATION:
        conc = tl.load(Conc_ptr + bh_idx * stride_c_h + block_idx * stride_c_blk).to(tl.float32)
        deviation = deviation * (0.3 + 0.7 * conc)
    
    potential = center_score + deviation
    
    # Determine mask
    if IS_CAUSAL:
        is_local = block_idx >= (N_BLOCKS - LOCAL_WINDOW)
        is_sink = (block_idx == 0)
        mask = (potential > THRESHOLD) | is_local | is_sink
    else:
        mask = potential > THRESHOLD
    
    # Store
    tl.store(Score_ptr + bh_idx * stride_s_h + block_idx * stride_s_blk, potential)
    tl.store(Mask_ptr + bh_idx * stride_mask_h + block_idx * stride_mask_blk, mask)


# Python wrapper for testing
def run_aether_sparse_attention(
    query: torch.Tensor,      # (B, H, D)
    keys: torch.Tensor,       # (B, H, S, D)
    values: torch.Tensor,     # (B, H, S, D)
    block_means: torch.Tensor,  # (B, H, N_blocks, D)
    block_radii: torch.Tensor,  # (B, H, N_blocks)
    block_conc: torch.Tensor,   # (B, H, N_blocks)
    threshold: float = 0.15,
    block_size: int = 64,
    is_causal: bool = True,
    local_window: int = 4,
) -> torch.Tensor:
    """
    Python wrapper for testing the kernel.
    """
    B, H, D = query.shape
    _, _, S, _ = keys.shape
    N_blocks = S // block_size
    scale = 1.0 / (D ** 0.5)
    
    output = torch.zeros(B, H, D, device=query.device, dtype=query.dtype)
    
    grid = (B, H, 1)
    
    aether_sparse_attention_kernel[grid](
        query, keys, values,
        block_means, block_radii, block_conc,
        output,
        # Q strides
        query.stride(0), query.stride(1), query.stride(2),
        # K strides
        keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3),
        # V strides
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        # Mean strides
        block_means.stride(0), block_means.stride(1), block_means.stride(2), block_means.stride(3),
        # Radius strides
        block_radii.stride(0), block_radii.stride(1), block_radii.stride(2),
        # Conc strides
        block_conc.stride(0), block_conc.stride(1), block_conc.stride(2),
        # Out strides
        output.stride(0), output.stride(1), output.stride(2),
        # Dimensions
        B, H, S,
        # Constexpr
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        N_BLOCKS=N_blocks,
        SCALE=scale,
        THRESHOLD=threshold,
        USE_CONCENTRATION=True,
        IS_CAUSAL=is_causal,
        LOCAL_WINDOW=local_window,
    )
    
    return output


if __name__ == "__main__":
    # Quick test
    import torch.nn.functional as F
    
    device = torch.device("cuda")
    B, H, S, D = 1, 8, 512, 128
    block_size = 64
    N_blocks = S // block_size
    
    query = torch.randn(B, H, D, device=device, dtype=torch.float16)
    keys = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    values = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    
    # Compute metadata
    keys_blocked = keys.view(B, H, N_blocks, block_size, D)
    means = F.normalize(keys_blocked.mean(dim=3), dim=-1)
    keys_norm = F.normalize(keys_blocked, dim=-1)
    distances = ((keys_norm - means.unsqueeze(3)) ** 2).sum(dim=-1)
    radii = distances.max(dim=-1).values.sqrt()
    alignment = (keys_norm * means.unsqueeze(3)).sum(dim=-1)
    conc = alignment.mean(dim=-1).clamp(0.1, 1.0)
    
    output = run_aether_sparse_attention(
        query, keys, values,
        means.float(), radii.float(), conc.float(),
        threshold=0.1,
        block_size=block_size,
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
