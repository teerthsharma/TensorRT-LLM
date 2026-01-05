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
AETHER Plugin Python Wrapper
============================

Python wrapper for loading and using the AETHER sparse attention TensorRT plugin.
Similar pattern to examples/openai_triton/manual_plugin/plugin.py.
"""

import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import Tensor, _create_tensor
from tensorrt_llm.module import Module


TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'
AETHER_PLUGIN_NAME = 'AetherAttention'

# Global handle for plugin library
_aether_plugin_handle = None


def _load_aether_plugin_lib(plugin_lib_path: Optional[Path] = None):
    """
    Load the AETHER plugin shared library.
    
    Args:
        plugin_lib_path: Path to .so file. If None, looks in default location.
    """
    global _aether_plugin_handle
    
    if _aether_plugin_handle is not None:
        return _aether_plugin_handle
    
    if plugin_lib_path is None:
        # Default location relative to this file
        plugin_lib_path = Path(__file__).parent / 'build' / 'libaether_attention_plugin.so'
    
    if not plugin_lib_path.exists():
        raise FileNotFoundError(
            f"AETHER plugin library not found at {plugin_lib_path}. "
            "Please build the plugin first with: cd build && cmake .. && make"
        )
    
    handle = ctypes.CDLL(str(plugin_lib_path), mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError(f"Failed to load AETHER plugin: {plugin_lib_path}")
    
    # Initialize plugin registration
    handle.initAetherPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initAetherPlugins.restype = ctypes.c_bool
    
    success = handle.initAetherPlugins(None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))
    if not success:
        raise RuntimeError("Failed to initialize AETHER plugins")
    
    _aether_plugin_handle = handle
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
    Create AETHER sparse attention plugin layer.
    
    Args:
        num_heads: Number of attention heads
        head_size: Head dimension
        block_size: Block size for sparse attention
        threshold: Sparsity threshold
        softmax_scale: Attention scale (1/sqrt(d))
        use_concentration: Enable concentration scoring
        is_causal: Enable causal masking
        local_window: Local window size
        inputs: [Q, K, V, means, radii, conc, (block_table)]
    
    Returns:
        Output tensor
    """
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        AETHER_PLUGIN_NAME, '1', TRT_LLM_PLUGIN_NAMESPACE)
    
    if plugin_creator is None:
        raise RuntimeError(
            f"Plugin '{AETHER_PLUGIN_NAME}' not found. "
            "Call _load_aether_plugin_lib() first."
        )
    
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
    
    plugin = plugin_creator.create_plugin("aether_attention", pfc)
    layer = default_trtnet().add_plugin_v2(inputs, plugin)
    
    return _create_tensor(layer.get_output(0), layer)


class AetherSparseAttentionLayer(Module):
    """
    TensorRT-LLM Module for AETHER Sparse Attention.
    
    This layer wraps the custom plugin for easy integration
    into TensorRT-LLM model definitions.
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
        """Forward pass."""
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
        
        out.mark_output('out', self.dtype)
        return out
    
    def prepare_inputs(
        self,
        max_batch_size: int,
        max_seq_len: int,
    ) -> List[Tensor]:
        """Prepare input tensors for TRT builder."""
        n_blocks = max_seq_len // self.block_size
        
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        seq_range = [1, (max_seq_len + 1) // 2, max_seq_len]
        blk_range = [1, (n_blocks + 1) // 2, n_blocks]
        
        Q = Tensor(
            name='Q', dtype=self.dtype,
            shape=[-1, self.num_heads, self.head_size],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_heads', [self.num_heads]),
                ('head_size', [self.head_size]),
            ]))
        
        K = Tensor(
            name='K', dtype=self.dtype,
            shape=[-1, self.num_kv_heads, -1, self.head_size],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('seq_len', [seq_range]),
                ('head_size', [self.head_size]),
            ]))
        
        V = Tensor(
            name='V', dtype=self.dtype,
            shape=[-1, self.num_kv_heads, -1, self.head_size],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('seq_len', [seq_range]),
                ('head_size', [self.head_size]),
            ]))
        
        block_means = Tensor(
            name='block_means', dtype=trt.float32,
            shape=[-1, self.num_kv_heads, -1, self.head_size],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('n_blocks', [blk_range]),
                ('head_size', [self.head_size]),
            ]))
        
        block_radii = Tensor(
            name='block_radii', dtype=trt.float32,
            shape=[-1, self.num_kv_heads, -1],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('n_blocks', [blk_range]),
            ]))
        
        block_conc = Tensor(
            name='block_conc', dtype=trt.float32,
            shape=[-1, self.num_kv_heads, -1],
            dim_range=OrderedDict([
                ('batch_size', [bs_range]),
                ('num_kv_heads', [self.num_kv_heads]),
                ('n_blocks', [blk_range]),
            ]))
        
        return [Q, K, V, block_means, block_radii, block_conc]


# Auto-load plugin when module is imported
try:
    _load_aether_plugin_lib()
except FileNotFoundError:
    pass  # Plugin not built yet, will fail on first use
