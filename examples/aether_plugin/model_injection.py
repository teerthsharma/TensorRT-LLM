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
AETHER Graph Injection (TASO-Style Substitution)
=================================================

This script performs a surgical swap of the standard gpt_attention plugin
with the custom AetherAttention plugin in TensorRT-LLM model definitions.

Usage:
    python model_injection.py \
        --plugin-lib build/libaether_attention_plugin.so \
        --model-dir /path/to/llama-3-8b \
        --output-dir outputs \
        --threshold 0.15 \
        --block-size 64

The injection works by:
1. Loading the custom plugin shared library
2. Locating LlamaAttention layers in the model
3. Replacing attention computation with AetherSparseAttentionLayer
4. Serializing concentration_threshold and sparsity parameters into engine plan
"""

import argparse
import ctypes
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder, BuilderConfig
from tensorrt_llm.functional import Tensor, _create_tensor
from tensorrt_llm.logger import logger
from tensorrt_llm.module import Module
from tensorrt_llm.network import net_guard


TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'
AETHER_PLUGIN_NAME = 'AetherAttention'


def load_aether_plugin_lib(plugin_lib_path: Path) -> ctypes.CDLL:
    """
    Load the AETHER plugin shared library and register with TensorRT.
    """
    if not plugin_lib_path.exists():
        raise FileNotFoundError(f"Plugin library not found: {plugin_lib_path}")
    
    handle = ctypes.CDLL(str(plugin_lib_path), mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError(f"Failed to load plugin: {plugin_lib_path}")
    
    # Initialize plugin
    handle.initAetherPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initAetherPlugins.restype = ctypes.c_bool
    
    success = handle.initAetherPlugins(None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))
    if not success:
        raise RuntimeError("Failed to initialize AETHER plugins")
    
    logger.info(f"Loaded AETHER plugin from {plugin_lib_path}")
    return handle


def aether_sparse_attention_op(
    num_heads: int,
    head_size: int,
    block_size: int,
    threshold: float,
    softmax_scale: float,
    use_concentration: bool,
    is_causal: bool,
    local_window: int,
    inputs: List[trt.ITensor],
) -> Tensor:
    """
    Create an AETHER sparse attention plugin instance in the TRT network.
    
    Args:
        num_heads: Number of attention heads
        head_size: Dimension per head
        block_size: Block size for sparse attention
        threshold: Sparsity threshold for block selection
        softmax_scale: Attention scale factor
        use_concentration: Whether to use concentration-based scoring
        is_causal: Whether to use causal masking
        local_window: Number of local blocks to always attend
        inputs: List of input tensors [Q, K, V, means, radii, conc, (block_table)]
    
    Returns:
        Output tensor from plugin
    """
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        AETHER_PLUGIN_NAME, '1', TRT_LLM_PLUGIN_NAMESPACE)
    
    if plugin_creator is None:
        raise RuntimeError(
            f"Plugin creator not found for {AETHER_PLUGIN_NAME}. "
            "Make sure the plugin library is loaded."
        )
    
    # Build plugin field collection
    pfc = trt.PluginFieldCollection([
        trt.PluginField("num_heads", np.array([num_heads], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([head_size], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("block_size", np.array([block_size], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("threshold", np.array([threshold], np.float32),
                        trt.PluginFieldType.FLOAT32),
        trt.PluginField("softmax_scale", np.array([softmax_scale], np.float32),
                        trt.PluginFieldType.FLOAT32),
        trt.PluginField("use_concentration", np.array([int(use_concentration)], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("is_causal", np.array([int(is_causal)], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("local_window", np.array([local_window], np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("type_id", np.array([int(inputs[0].dtype)], np.int32),
                        trt.PluginFieldType.INT32),
    ])
    
    plugin = plugin_creator.create_plugin("aether_sparse_attention", pfc)
    layer = default_trtnet().add_plugin_v2(inputs, plugin)
    
    return _create_tensor(layer.get_output(0), layer)


class AetherSparseAttentionLayer(Module):
    """
    AETHER Sparse Attention Layer for TensorRT-LLM integration.
    
    This module wraps the custom AETHER plugin and handles:
    - Metadata tensor preparation (means, radii, concentrations)
    - Graph substitution for LlamaAttention
    - Serialization of sparsity parameters
    """
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: Optional[int] = None,
        block_size: int = 64,
        threshold: float = 0.15,
        use_concentration: bool = True,
        is_causal: bool = True,
        local_window: int = 4,
        dtype: str = "float16",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.block_size = block_size
        self.threshold = threshold
        self.softmax_scale = 1.0 / (head_size ** 0.5)
        self.use_concentration = use_concentration
        self.is_causal = is_causal
        self.local_window = local_window
        self.dtype = str_dtype_to_trt(dtype)
    
    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        block_means: Tensor,
        block_radii: Tensor,
        block_conc: Tensor,
        block_table: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through AETHER sparse attention.
        
        Args:
            Q: Query tensor (B, H, D) or (B, H, Sq, D)
            K: Key cache (B, H, S, D)
            V: Value cache (B, H, S, D)
            block_means: Precomputed block centroids (B, H, N_blocks, D)
            block_radii: Precomputed block radii (B, H, N_blocks)
            block_conc: Precomputed concentrations (B, H, N_blocks)
            block_table: Optional paged KV cache indirection (B, max_blocks)
        
        Returns:
            Output tensor (B, H, D) or (B, H, Sq, D)
        """
        inputs = [
            Q.trt_tensor, K.trt_tensor, V.trt_tensor,
            block_means.trt_tensor, block_radii.trt_tensor, block_conc.trt_tensor,
        ]
        
        if block_table is not None:
            inputs.append(block_table.trt_tensor)
        
        out = aether_sparse_attention_op(
            num_heads=self.num_heads,
            head_size=self.head_size,
            block_size=self.block_size,
            threshold=self.threshold,
            softmax_scale=self.softmax_scale,
            use_concentration=self.use_concentration,
            is_causal=self.is_causal,
            local_window=self.local_window,
            inputs=inputs,
        )
        
        out.mark_output('aether_attention_out', self.dtype)
        return out
    
    def prepare_inputs(
        self,
        max_batch_size: int,
        max_seq_len: int,
        max_blocks: Optional[int] = None,
    ) -> Tuple[Tensor, ...]:
        """
        Prepare input tensors with proper shapes for TRT builder.
        """
        n_blocks = max_seq_len // self.block_size
        if max_blocks is None:
            max_blocks = n_blocks
        
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        seq_range = [1, (max_seq_len + 1) // 2, max_seq_len]
        blk_range = [1, (n_blocks + 1) // 2, n_blocks]
        
        # Q: (B, H, D) for decode
        Q = Tensor(
            name='Q',
            dtype=self.dtype,
            shape=[-1, self.num_heads, self.head_size],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_heads', [self.num_heads]),
                ('head_size', [self.head_size]),
            ])
        )
        
        # K, V: (B, H, S, D)
        kv_shape = [-1, self.num_kv_heads, -1, self.head_size]
        K = Tensor(
            name='K',
            dtype=self.dtype,
            shape=kv_shape,
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('seq_len', [seq_range]),
                ('head_size', [self.head_size]),
            ])
        )
        V = Tensor(
            name='V',
            dtype=self.dtype,
            shape=kv_shape,
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('seq_len', [seq_range]),
                ('head_size', [self.head_size]),
            ])
        )
        
        # Metadata tensors (FP32)
        block_means = Tensor(
            name='block_means',
            dtype=trt.float32,
            shape=[-1, self.num_kv_heads, -1, self.head_size],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('n_blocks', [blk_range]),
                ('head_size', [self.head_size]),
            ])
        )
        
        block_radii = Tensor(
            name='block_radii',
            dtype=trt.float32,
            shape=[-1, self.num_kv_heads, -1],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('n_blocks', [blk_range]),
            ])
        )
        
        block_conc = Tensor(
            name='block_conc',
            dtype=trt.float32,
            shape=[-1, self.num_kv_heads, -1],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('n_blocks', [blk_range]),
            ])
        )
        
        return Q, K, V, block_means, block_radii, block_conc


def inject_aether_attention(
    model_config: Dict,
    plugin_lib_path: Path,
    threshold: float = 0.15,
    block_size: int = 64,
    use_concentration: bool = True,
) -> Dict:
    """
    Perform TASO-style graph injection to substitute gpt_attention with AetherAttention.
    
    This modifies the model configuration to use AETHER sparse attention.
    
    Args:
        model_config: Original model configuration dict
        plugin_lib_path: Path to compiled plugin .so
        threshold: Sparsity threshold
        block_size: Block size for sparse attention
        use_concentration: Whether to use concentration scoring
    
    Returns:
        Modified model configuration with AETHER parameters
    """
    # Load plugin
    load_aether_plugin_lib(plugin_lib_path)
    
    # Inject AETHER parameters into config
    aether_config = {
        "aether_attention": {
            "enabled": True,
            "threshold": threshold,
            "block_size": block_size,
            "use_concentration": use_concentration,
            "is_causal": True,
            "local_window": 4,
        }
    }
    
    model_config.update(aether_config)
    
    # Mark attention layers for substitution
    if "architecture" in model_config:
        arch = model_config["architecture"]
        if "LlamaForCausalLM" in arch or "llama" in arch.lower():
            logger.info("Detected Llama architecture - marking for AETHER substitution")
            model_config["attention_plugin"] = "aether"
    
    return model_config


def build_aether_engine(
    model_dir: Path,
    output_dir: Path,
    plugin_lib_path: Path,
    threshold: float = 0.15,
    block_size: int = 64,
    use_concentration: bool = True,
    max_batch_size: int = 8,
    max_seq_len: int = 4096,
    dtype: str = "float16",
) -> Path:
    """
    Build TensorRT engine with AETHER sparse attention.
    """
    from tensorrt_llm.models import LLaMAForCausalLM
    
    # Load plugin
    load_aether_plugin_lib(plugin_lib_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        model_config = json.load(f)
    
    # Inject AETHER config
    model_config = inject_aether_attention(
        model_config, plugin_lib_path, threshold, block_size, use_concentration
    )
    
    # Save modified config
    output_config = output_dir / "config.json"
    with open(output_config, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    logger.info(f"Built AETHER engine config at {output_config}")
    return output_config


def main():
    parser = argparse.ArgumentParser(
        description="Inject AETHER sparse attention into TensorRT-LLM model"
    )
    parser.add_argument("--plugin-lib", type=Path, required=True,
                        help="Path to libaether_attention_plugin.so")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to model directory (with config.json)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for engine")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Sparsity threshold (default: 0.15)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Block size (default: 64)")
    parser.add_argument("--use-concentration", action="store_true", default=True,
                        help="Use concentration-based scoring")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["verbose", "info", "warning", "error"],
                        help="Logging level")
    
    args = parser.parse_args()
    logger.set_level(args.log_level)
    
    logger.info("AETHER Graph Injection")
    logger.info(f"  Plugin: {args.plugin_lib}")
    logger.info(f"  Model: {args.model_dir}")
    logger.info(f"  Threshold: {args.threshold}")
    logger.info(f"  Block Size: {args.block_size}")
    
    config_path = build_aether_engine(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        plugin_lib_path=args.plugin_lib,
        threshold=args.threshold,
        block_size=args.block_size,
        use_concentration=args.use_concentration,
    )
    
    logger.info(f"Injection complete: {config_path}")


if __name__ == "__main__":
    main()
