#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AETHER Sparse Attention TensorRT-LLM Plugin Deployment
# =======================================================
# Target: Llama-3-8B (Int4) on sm_89 (RTX 4060)
# Workflow: Triton -> PTX -> C Header -> TRT Plugin -> Graph Injection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_ROOT="${TRITON_ROOT:-$(python -c 'import triton; print(triton.__path__[0])')}"
BUILD_DIR="${SCRIPT_DIR}/build"
AOT_DIR="${SCRIPT_DIR}/aot"
SM_VERSION="${SM_VERSION:-89}"

echo "=========================================="
echo " AETHER TRT Plugin Deployment"
echo " Target: sm_${SM_VERSION}"
echo "=========================================="

# =============================================================================
# STEP 1: Compile Triton AOT to PTX + C Header
# =============================================================================
echo ""
echo "[STEP 1/4] Compiling Triton kernel to PTX..."

mkdir -p "${AOT_DIR}/sm${SM_VERSION}"

# Compile AETHER sparse attention kernel for FP16
python "${SCRIPT_DIR}/compile_payload.py" \
    --kernel-file "${SCRIPT_DIR}/aether_sparse_attn.py" \
    --kernel-name "aether_sparse_attention_kernel" \
    --output-dir "${AOT_DIR}/sm${SM_VERSION}" \
    --sm-version "${SM_VERSION}" \
    --dtype "fp16" \
    --head-dim 128 \
    --block-size 64

# Link generated headers
echo "[STEP 1/4] Linking AOT headers..."
if [ -f "${TRITON_ROOT}/../tools/link.py" ]; then
    python "${TRITON_ROOT}/../tools/link.py" \
        "${AOT_DIR}/sm${SM_VERSION}"/*.h \
        -o "${AOT_DIR}/aether_kernel_fp16"
else
    echo "Warning: Triton link.py not found, using generated headers directly"
fi

echo "[STEP 1/4] ✓ PTX compilation complete"
echo "  → ${AOT_DIR}/sm${SM_VERSION}/aether_sparse_kernel.ptx"
echo "  → ${SCRIPT_DIR}/kernel_payload.h"

# =============================================================================
# STEP 2: Compile C++ Plugin (.so)
# =============================================================================
echo ""
echo "[STEP 2/4] Building C++ TensorRT plugin..."

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSM_VERSION="${SM_VERSION}" \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME:-/usr/local/cuda}"

make -j"$(nproc)"

if [ -f "${BUILD_DIR}/libaether_attention_plugin.so" ]; then
    echo "[STEP 2/4] ✓ Plugin build complete"
    echo "  → ${BUILD_DIR}/libaether_attention_plugin.so"
else
    echo "[STEP 2/4] ✗ Plugin build FAILED"
    exit 1
fi

cd "${SCRIPT_DIR}"

# =============================================================================
# STEP 3: Python Graph Injection
# =============================================================================
echo ""
echo "[STEP 3/4] Running graph injection..."

MODEL_DIR="${MODEL_DIR:-/models/llama-3-8b-int4}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"

python "${SCRIPT_DIR}/model_injection.py" \
    --plugin-lib "${BUILD_DIR}/libaether_attention_plugin.so" \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --threshold 0.15 \
    --block-size 64 \
    --use-concentration

echo "[STEP 3/4] ✓ Graph injection complete"
echo "  → ${OUTPUT_DIR}/config.json"

# =============================================================================
# STEP 4: E2E Verification Test
# =============================================================================
echo ""
echo "[STEP 4/4] Running E2E verification..."

# Single-GPU test
python "${SCRIPT_DIR}/run_verification.py" \
    --engine-dir "${OUTPUT_DIR}" \
    --prompts "The quick brown fox" "In the beginning" \
    --max-output-len 64

# Multi-GPU test (if available)
if command -v mpirun &> /dev/null && [ "${WORLD_SIZE:-1}" -gt 1 ]; then
    echo ""
    echo "[STEP 4/4] Running distributed verification with mpirun..."
    mpirun -n "${WORLD_SIZE}" \
        --allow-run-as-root \
        python "${SCRIPT_DIR}/run_verification.py" \
            --engine-dir "${OUTPUT_DIR}" \
            --prompts "The future of AI is" \
            --max-output-len 128 \
            --distributed
fi

echo ""
echo "=========================================="
echo " AETHER Deployment Complete"
echo " Engine: ${OUTPUT_DIR}"
echo "=========================================="
