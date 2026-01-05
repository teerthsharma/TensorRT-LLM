# AETHER Sparse Attention TensorRT-LLM Plugin

Inject custom Triton kernels into TensorRT-LLM C++ runtime via ahead-of-time PTX compilation.

## Overview

This plugin enables block-sparse attention using the AETHER (Adaptive Event-driven Threshold Hybrid Entangled Rendering) algorithm within TensorRT-LLM inference pipelines.

**Target Configuration:**
- Model: Llama-3-8B (Int4 quantization)
- Hardware: sm_89 (RTX 4060 / Ada Lovelace)
- Backend: TensorRT-LLM C++ Runtime

## Architecture

```
┌─────────────────────┐
│  aether_sparse_attn.py  │  Triton Source Kernel
└──────────┬──────────┘
           │ triton.compile (AOT)
           ▼
┌─────────────────────┐
│  kernel_payload.h   │  Embedded PTX String
└──────────┬──────────┘
           │ cuModuleLoadData
           ▼
┌─────────────────────┐
│ AetherAttentionPlugin.cpp │  TRT IPluginV2DynamicExt
└──────────┬──────────┘
           │ cuLaunchKernel
           ▼
┌─────────────────────┐
│  TensorRT-LLM Engine │  Llama-3-8B with AETHER
└─────────────────────┘
```

## Quick Start

```bash
# Full deployment (Triton→PTX→Plugin→Engine)
bash deploy_aether_plugin.sh
```

## Step-by-Step

### 1. Compile Triton to PTX

```bash
python compile_payload.py \
    --kernel-file aether_sparse_attn.py \
    --kernel-name aether_sparse_attention_kernel \
    --output-dir aot/sm89 \
    --sm-version 89 \
    --dtype fp16
```

### 2. Build C++ Plugin

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSM_VERSION=89
make -j$(nproc)
# Output: build/libaether_attention_plugin.so
```

### 3. Inject into Model

```bash
python model_injection.py \
    --plugin-lib build/libaether_attention_plugin.so \
    --model-dir /path/to/llama-3-8b \
    --output-dir outputs \
    --threshold 0.15
```

### 4. Run Verification

```bash
python run_verification.py \
    --engine-dir outputs \
    --prompts "The future of AI"
```

## File Structure

```
examples/aether_plugin/
├── deploy_aether_plugin.sh    # Master deployment script
├── aether_sparse_attn.py      # Triton source kernel
├── compile_payload.py         # AOT compilation
├── kernel_payload.h           # Generated PTX header
├── AetherAttentionPlugin.h    # C++ plugin header
├── AetherAttentionPlugin.cpp  # C++ plugin implementation
├── tritonPlugins.cpp          # Plugin registration
├── CMakeLists.txt             # Build configuration
├── model_injection.py         # Graph substitution
├── plugin.py                  # Python wrapper
├── run_verification.py        # E2E test harness
└── README.md                  # This file
```

## Key Implementation Details

### ABI Packing (C++ → Triton)

The critical bridge in `enqueue()` maps TensorRT `void* inputs[]` to Triton kernel arguments:

```cpp
void* kernelArgs[] = {
    (void*) &Q, (void*) &K, (void*) &V,
    (void*) &means, (void*) &radii, (void*) &conc,
    (void*) &Out,
    // Strides...
    (void*) &batchSize, (void*) &mNumHeads, (void*) &seqLen,
};

cuLaunchKernel(mKernel, gridX, gridY, gridZ,
               blockX, blockY, blockZ,
               sharedMem, stream, kernelArgs, nullptr);
```

### Paged KV Cache

When using paged KV cache, the kernel performs indirection via `block_table`:

```cpp
// Physical address = block_table[batch, block_idx] * block_size + offset
if (mUsePagedKVCache) {
    blockTable = reinterpret_cast<int32_t const*>(inputs[6]);
}
```

### Quantization

Handles Int4/Int8 inputs while computing in FP16:

```cpp
// Input may be quantized (Int4/Int8), but kernel computes in FP16
T const* Q = reinterpret_cast<T const*>(inputs[0]);  // T = half
```

## References

- Sharma, T. (2024). *AETHER - Adaptive Event-driven Threshold Hybrid Entangled Rendering*. DOI: [10.13141/RG.2.2.14811.27684](https://doi.org/10.13141/RG.2.2.14811.27684)
- [TensorRT-LLM OpenAI Triton Plugin Example](../openai_triton/manual_plugin/)
- [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

## License

Apache-2.0
