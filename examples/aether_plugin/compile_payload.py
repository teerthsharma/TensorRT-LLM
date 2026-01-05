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
AETHER Sparse Attention - Triton AOT Compiler
==============================================

Compiles the AETHER sparse attention kernel to PTX for embedding in C++ TRT plugin.

Usage:
    python compile_payload.py \
        --kernel-file aether_sparse_attn.py \
        --kernel-name aether_sparse_attention_kernel \
        --output-dir aot/sm89 \
        --sm-version 89 \
        --dtype fp16 \
        --head-dim 128 \
        --block-size 64

Outputs:
    - aot/sm89/aether_sparse_kernel.ptx    (Raw PTX assembly)
    - kernel_payload.h                      (C header with embedded PTX)
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import triton
from triton.compiler import CompiledKernel, ASTSource


@dataclass
class KernelMetadata:
    """Metadata extracted from compiled Triton kernel."""
    name: str
    ptx: str
    shared_memory: int
    num_registers: int
    num_warps: int
    signature: str
    constants: dict


def compile_triton_to_ptx(
    kernel_file: Path,
    kernel_name: str,
    sm_version: int,
    head_dim: int,
    block_size: int,
    n_blocks: int,
    dtype: str = "fp16",
) -> KernelMetadata:
    """
    Compile a Triton kernel to PTX using AOT compilation.
    
    Args:
        kernel_file: Path to .py file containing Triton kernel
        kernel_name: Name of the kernel function
        sm_version: CUDA compute capability (e.g., 89 for sm_89)
        head_dim: Attention head dimension (constexpr)
        block_size: Block size for sparse attention (constexpr)
        n_blocks: Number of blocks (constexpr)
        dtype: Data type (fp16 or fp32)
    
    Returns:
        KernelMetadata with PTX and configuration
    """
    # Import the kernel module
    import importlib.util
    spec = importlib.util.spec_from_file_location("kernel_module", kernel_file)
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)
    
    kernel_fn = getattr(kernel_module, kernel_name)
    
    # Define constexpr values
    constexprs = {
        "HEAD_DIM": head_dim,
        "BLOCK_SIZE": block_size,
        "N_BLOCKS": n_blocks,
        "SCALE": 1.0 / (head_dim ** 0.5),
        "USE_CONCENTRATION": True,
        "IS_CAUSAL": True,
        "LOCAL_WINDOW": 4,
    }
    
    # Build signature based on dtype
    if dtype == "fp16":
        ptr_type = "*fp16:16"
        scalar_type = "fp16"
    else:
        ptr_type = "*fp32:16"
        scalar_type = "fp32"
    
    # AETHER kernel signature:
    # Q_ptr, K_ptr, V_ptr, Mean_Ptr, Rad_Ptr, Conc_Ptr, Out_ptr,
    # stride_q_b, stride_q_h, stride_q_d, ...
    # plus scalar config params
    signature = ", ".join([
        # Pointers
        ptr_type,  # Q
        ptr_type,  # K
        ptr_type,  # V
        "*fp32:16",  # Mean (always fp32)
        "*fp32:16",  # Radius
        "*fp32:16",  # Concentration
        ptr_type,  # Out
        # Strides (i32)
        "i32", "i32", "i32",  # Q strides
        "i32", "i32", "i32", "i32",  # K strides
        "i32", "i32", "i32", "i32",  # V strides
        "i32", "i32", "i32", "i32",  # Mean strides
        "i32", "i32", "i32",  # Radius strides
        "i32", "i32", "i32",  # Concentration strides
        "i32", "i32", "i32",  # Out strides
        # Scalars
        "i32", "i32", "i32",  # batch_size, num_heads, seq_len
        # Constexprs are encoded in signature
        f"{head_dim}",  # HEAD_DIM
        f"{block_size}",  # BLOCK_SIZE
        f"{n_blocks}",  # N_BLOCKS
    ])
    
    # Compile using triton compile
    target = f"cuda:{sm_version}"
    
    try:
        # Try modern Triton API
        compiled = triton.compile(
            kernel_fn,
            signature=signature,
            constants=constexprs,
            target=target,
        )
        ptx = compiled.asm["ptx"]
        shared_memory = compiled.shared
        num_warps = compiled.num_warps
        num_registers = getattr(compiled, "num_regs", 32)
    except Exception as e:
        print(f"Warning: Modern API failed ({e}), trying legacy method...")
        # Fallback: Use triton.tools.compile
        from triton.tools import compile as triton_aot_compile
        
        result = triton_aot_compile(
            str(kernel_file),
            kernel_name,
            signature,
            target,
            num_warps=4,
            num_stages=2,
        )
        ptx = result["ptx"]
        shared_memory = result.get("shared", 0)
        num_warps = 4
        num_registers = 32
    
    return KernelMetadata(
        name=kernel_name,
        ptx=ptx,
        shared_memory=shared_memory,
        num_registers=num_registers,
        num_warps=num_warps,
        signature=signature,
        constants=constexprs,
    )


def generate_c_header(
    metadata: KernelMetadata,
    output_path: Path,
    kernel_func_name: str = "aether_sparse_attn",
) -> None:
    """
    Generate C header file with embedded PTX string.
    
    The generated header includes:
    - PTX string as const char*
    - Kernel configuration (shared mem, registers, warps)
    - load_*/unload_* wrapper functions for CUDA Driver API
    """
    # Escape PTX for C string
    escaped_ptx = metadata.ptx.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n\"\n\"")
    
    header_content = f'''/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AUTO-GENERATED by compile_payload.py - DO NOT EDIT
 *
 * AETHER Sparse Attention Kernel Payload
 * Target: sm_{metadata.constants.get("SM_VERSION", "89")}
 * Head Dim: {metadata.constants.get("HEAD_DIM", 128)}
 * Block Size: {metadata.constants.get("BLOCK_SIZE", 64)}
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {{
#endif

// Kernel configuration
#define {kernel_func_name.upper()}_SHARED_MEM {metadata.shared_memory}
#define {kernel_func_name.upper()}_NUM_REGS {metadata.num_registers}
#define {kernel_func_name.upper()}_NUM_WARPS {metadata.num_warps}
#define {kernel_func_name.upper()}_BLOCK_SIZE ({metadata.num_warps} * 32)

// Embedded PTX assembly
static const char {kernel_func_name}_ptx[] =
"{escaped_ptx}";

// CUDA module and function handles
static CUmodule {kernel_func_name}_mod = NULL;
static CUfunction {kernel_func_name}_func = NULL;

// Load PTX module
static inline CUresult load_{kernel_func_name}(void) {{
    if ({kernel_func_name}_mod != NULL) {{
        return CUDA_SUCCESS;  // Already loaded
    }}
    CUresult err = cuModuleLoadData(&{kernel_func_name}_mod, {kernel_func_name}_ptx);
    if (err != CUDA_SUCCESS) {{
        return err;
    }}
    err = cuModuleGetFunction(&{kernel_func_name}_func, {kernel_func_name}_mod, "{metadata.name}");
    return err;
}}

// Unload PTX module
static inline void unload_{kernel_func_name}(void) {{
    if ({kernel_func_name}_mod != NULL) {{
        cuModuleUnload({kernel_func_name}_mod);
        {kernel_func_name}_mod = NULL;
        {kernel_func_name}_func = NULL;
    }}
}}

// Get kernel function handle
static inline CUfunction get_{kernel_func_name}_func(void) {{
    return {kernel_func_name}_func;
}}

// Get shared memory size
static inline int get_{kernel_func_name}_shared_mem(void) {{
    return {kernel_func_name.upper()}_SHARED_MEM;
}}

#ifdef __cplusplus
}}
#endif
'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header_content)
    print(f"Generated: {output_path}")


def generate_ptx_file(metadata: KernelMetadata, output_path: Path) -> None:
    """Write raw PTX to file for debugging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(metadata.ptx)
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile AETHER Triton kernel to PTX for TRT plugin embedding"
    )
    parser.add_argument("--kernel-file", type=Path, required=True,
                        help="Path to Triton kernel .py file")
    parser.add_argument("--kernel-name", type=str, required=True,
                        help="Name of kernel function")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for generated files")
    parser.add_argument("--sm-version", type=int, default=89,
                        help="CUDA SM version (default: 89 for RTX 4060)")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "fp32"],
                        help="Data type (default: fp16)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Attention head dimension (default: 128)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Sparse attention block size (default: 64)")
    parser.add_argument("--n-blocks", type=int, default=64,
                        help="Number of blocks (default: 64)")
    
    args = parser.parse_args()
    
    print(f"Compiling {args.kernel_file}::{args.kernel_name}")
    print(f"  Target: sm_{args.sm_version}")
    print(f"  Config: HEAD_DIM={args.head_dim}, BLOCK_SIZE={args.block_size}")
    
    # Compile kernel
    metadata = compile_triton_to_ptx(
        kernel_file=args.kernel_file,
        kernel_name=args.kernel_name,
        sm_version=args.sm_version,
        head_dim=args.head_dim,
        block_size=args.block_size,
        n_blocks=args.n_blocks,
        dtype=args.dtype,
    )
    
    print(f"  Shared Memory: {metadata.shared_memory} bytes")
    print(f"  Registers: {metadata.num_registers}")
    print(f"  Warps: {metadata.num_warps}")
    
    # Generate output files
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Raw PTX file
    ptx_path = output_dir / f"aether_sparse_kernel_{args.dtype}.ptx"
    generate_ptx_file(metadata, ptx_path)
    
    # C header with embedded PTX
    header_path = args.kernel_file.parent / "kernel_payload.h"
    generate_c_header(metadata, header_path, f"aether_sparse_{args.dtype}")
    
    print("\nCompilation complete!")
    print(f"  PTX: {ptx_path}")
    print(f"  Header: {header_path}")


if __name__ == "__main__":
    main()
