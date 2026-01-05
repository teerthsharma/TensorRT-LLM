/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AUTO-GENERATED PLACEHOLDER - Replace with actual compiled PTX
 *
 * This is a placeholder header. To generate the real payload:
 *   python compile_payload.py \
 *       --kernel-file aether_sparse_attn.py \
 *       --kernel-name aether_sparse_attention_kernel \
 *       --output-dir aot/sm89 \
 *       --sm-version 89
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Kernel configuration (placeholder values)
#define AETHER_SPARSE_FP16_SHARED_MEM 49152
#define AETHER_SPARSE_FP16_NUM_REGS 64
#define AETHER_SPARSE_FP16_NUM_WARPS 4
#define AETHER_SPARSE_FP16_BLOCK_SIZE (AETHER_SPARSE_FP16_NUM_WARPS * 32)

// Placeholder PTX - replace with actual compiled output
static const char aether_sparse_fp16_ptx[] =
"// PLACEHOLDER PTX - Run compile_payload.py to generate\n"
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
".entry aether_sparse_attention_kernel(\n"
"    .param .u64 Q_ptr,\n"
"    .param .u64 K_ptr,\n"
"    .param .u64 V_ptr\n"
") {\n"
"    ret;\n"
"}\n";

// Module handles
static CUmodule aether_sparse_fp16_mod = NULL;
static CUfunction aether_sparse_fp16_func = NULL;

// Load PTX module
static inline CUresult load_aether_sparse_fp16(void) {
    if (aether_sparse_fp16_mod != NULL) {
        return CUDA_SUCCESS;
    }
    CUresult err = cuModuleLoadData(&aether_sparse_fp16_mod, aether_sparse_fp16_ptx);
    if (err != CUDA_SUCCESS) {
        return err;
    }
    err = cuModuleGetFunction(&aether_sparse_fp16_func, aether_sparse_fp16_mod, 
                              "aether_sparse_attention_kernel");
    return err;
}

// Unload PTX module
static inline void unload_aether_sparse_fp16(void) {
    if (aether_sparse_fp16_mod != NULL) {
        cuModuleUnload(aether_sparse_fp16_mod);
        aether_sparse_fp16_mod = NULL;
        aether_sparse_fp16_func = NULL;
    }
}

// Get kernel function
static inline CUfunction get_aether_sparse_fp16_func(void) {
    return aether_sparse_fp16_func;
}

// Get shared memory size
static inline int get_aether_sparse_fp16_shared_mem(void) {
    return AETHER_SPARSE_FP16_SHARED_MEM;
}

#ifdef __cplusplus
}
#endif
