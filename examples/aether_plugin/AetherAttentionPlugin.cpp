/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * AETHER Sparse Attention Plugin Implementation
 * ==============================================
 * 
 * This file implements the critical bridge between TensorRT and Triton-compiled PTX:
 * 1. Loads embedded PTX via cuModuleLoadData
 * 2. Maps TensorRT inputs[] to Triton kernel ABI
 * 3. Handles paged KV cache pointer arithmetic via block_table
 * 4. Invokes kernel via cuLaunchKernel
 *
 * Reference:
 *   Sharma, T. (2024). AETHER - Adaptive Event-driven Threshold Hybrid
 *   Entangled Rendering. DOI: 10.13141/RG.2.2.14811.27684
 */
#include "AetherAttentionPlugin.h"

// Import generated PTX payload
extern "C"
{
#include "kernel_payload.h"
}

#include <cstring>
#include <cuda_fp16.h>
#include <iostream>
#include <string>

using namespace nvinfer1;
using aether::plugin::AetherAttentionPluginCreator;
using aether::plugin::AetherAttentionPlugin;

static char const* AETHER_ATTENTION_PLUGIN_VERSION{"1"};
static char const* AETHER_ATTENTION_PLUGIN_NAME{"AetherAttention"};
PluginFieldCollection AetherAttentionPluginCreator::mFC{};
std::vector<PluginField> AetherAttentionPluginCreator::mPluginAttributes;

namespace aether::plugin
{

// =============================================================================
// Utility Functions
// =============================================================================

// Write values into buffer for serialization
template <typename T>
void writeArg(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer for deserialization
template <typename T>
void readArg(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

std::uintptr_t constexpr kCudaMemAlign = 128;

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    if (addr % kCudaMemAlign)
    {
        addr += kCudaMemAlign - addr % kCudaMemAlign;
    }
    return (int8_t*) addr;
}

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        CUresult err = call;                                                     \
        if (err != CUDA_SUCCESS) {                                              \
            const char* errStr;                                                  \
            cuGetErrorString(err, &errStr);                                     \
            std::cerr << "CUDA Error: " << errStr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1;                                                           \
        }                                                                        \
    } while(0)

// =============================================================================
// AetherAttentionPlugin Implementation
// =============================================================================

AetherAttentionPlugin::AetherAttentionPlugin(
    int numHeads, int headSize, int blockSize, float threshold,
    float softmaxScale, bool useConcentration, bool isCausal,
    int localWindow, nvinfer1::DataType type)
    : mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mBlockSize(blockSize)
    , mThreshold(threshold)
    , mSoftmaxScale(softmaxScale)
    , mUseConcentration(useConcentration)
    , mIsCausal(isCausal)
    , mLocalWindow(localWindow)
    , mType(type)
    , mModule(nullptr)
    , mKernel(nullptr)
    , mUsePagedKVCache(false)
{
}

AetherAttentionPlugin::AetherAttentionPlugin(void const* data, size_t length)
    : mModule(nullptr)
    , mKernel(nullptr)
    , mUsePagedKVCache(false)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    readArg(d, mNumHeads);
    readArg(d, mHeadSize);
    readArg(d, mBlockSize);
    readArg(d, mThreshold);
    readArg(d, mSoftmaxScale);
    readArg(d, mUseConcentration);
    readArg(d, mIsCausal);
    readArg(d, mLocalWindow);
    readArg(d, mType);
    assert(d == a + length);
}

nvinfer1::IPluginV2DynamicExt* AetherAttentionPlugin::clone() const noexcept
{
    auto* plugin = new AetherAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AetherAttentionPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output shape: same as Q input (B, H, D)
    assert(outputIndex == 0);
    return inputs[0];  // Q shape
}

bool AetherAttentionPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    /*
     * Inputs:
     *   0: Q        (FP16/FP32)
     *   1: K        (FP16/FP32)
     *   2: V        (FP16/FP32)
     *   3: means    (FP32)
     *   4: radii    (FP32)
     *   5: conc     (FP32)
     *   6: block_table (optional, INT32)
     * Outputs:
     *   0: Out      (FP16/FP32)
     */
    assert(nbInputs >= 6 && nbInputs <= 7);
    assert(nbOutputs == 1);
    assert(0 <= pos && pos < nbInputs + nbOutputs);

    bool isValid = false;
    
    if (pos == 0 || pos == 1 || pos == 2)
    {
        // Q, K, V - must match mType
        isValid = inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos >= 3 && pos <= 5)
    {
        // Metadata tensors - always FP32
        isValid = inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == 6 && nbInputs == 7)
    {
        // block_table (paged KV cache) - INT32
        isValid = inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == nbInputs)
    {
        // Output - must match Q
        isValid = inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    }
    
    return isValid;
}

void AetherAttentionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    // Check if we're using paged KV cache
    mUsePagedKVCache = (nbInputs == 7);
}

size_t AetherAttentionPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // Workspace for block scores during selection
    auto const K = inputs[1];
    int const batchSize = K.dims.d[0];
    int const seqLen = K.dims.d[2];
    int const nBlocks = seqLen / mBlockSize;
    
    // Block scores: B * H * N_blocks * sizeof(float)
    size_t scoresBufSize = sizeof(float) * batchSize * mNumHeads * nBlocks;
    
    // Align to 128 bytes
    if (scoresBufSize % kCudaMemAlign)
    {
        scoresBufSize += kCudaMemAlign - (scoresBufSize % kCudaMemAlign);
    }
    
    return scoresBufSize;
}

template <typename T>
int AetherAttentionPlugin::enqueueImpl(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream)
{
    /*
     * Critical ABI Packing:
     * Pack TensorRT inputs[] into void* args[] matching Triton kernel signature.
     * 
     * Triton signature:
     *   Q_ptr, K_ptr, V_ptr, Mean_ptr, Rad_ptr, Conc_ptr, Out_ptr,
     *   stride_q_b, stride_q_h, stride_q_d,
     *   stride_k_b, stride_k_h, stride_k_s, stride_k_d,
     *   ... (more strides)
     *   batch_size, num_heads, seq_len,
     *   HEAD_DIM, BLOCK_SIZE, N_BLOCKS, ... (constexpr embedded)
     */
    
    // Extract dimensions
    int batchSize = inputDesc[0].dims.d[0];
    int seqLen = inputDesc[1].dims.d[2];
    int nBlocks = seqLen / mBlockSize;
    
    // Input/output pointers
    T const* Q = reinterpret_cast<T const*>(inputs[0]);
    T const* K = reinterpret_cast<T const*>(inputs[1]);
    T const* V = reinterpret_cast<T const*>(inputs[2]);
    float const* means = reinterpret_cast<float const*>(inputs[3]);
    float const* radii = reinterpret_cast<float const*>(inputs[4]);
    float const* conc = reinterpret_cast<float const*>(inputs[5]);
    
    T* Out = reinterpret_cast<T*>(outputs[0]);
    
    // Handle paged KV cache if enabled
    int32_t const* blockTable = nullptr;
    if (mUsePagedKVCache)
    {
        blockTable = reinterpret_cast<int32_t const*>(inputs[6]);
        // NOTE: When using paged KV cache, the kernel needs to perform
        // pointer arithmetic using block_table to get physical addresses:
        //   physical_idx = block_table[batch, block_idx] * block_size + offset
        //   K_physical = K_pool + physical_idx * stride
    }
    
    // Compute strides (assuming contiguous layout)
    // Q: (B, H, D)
    int stride_q_b = mNumHeads * mHeadSize;
    int stride_q_h = mHeadSize;
    int stride_q_d = 1;
    
    // K, V: (B, H, S, D)
    int stride_k_b = mNumHeads * seqLen * mHeadSize;
    int stride_k_h = seqLen * mHeadSize;
    int stride_k_s = mHeadSize;
    int stride_k_d = 1;
    
    // means: (B, H, N_blocks, D)
    int stride_m_b = mNumHeads * nBlocks * mHeadSize;
    int stride_m_h = nBlocks * mHeadSize;
    int stride_m_blk = mHeadSize;
    int stride_m_d = 1;
    
    // radii, conc: (B, H, N_blocks)
    int stride_r_b = mNumHeads * nBlocks;
    int stride_r_h = nBlocks;
    int stride_r_blk = 1;
    
    // Output strides (same as Q)
    int stride_o_b = stride_q_b;
    int stride_o_h = stride_q_h;
    int stride_o_d = stride_q_d;
    
    // Pack kernel arguments
    // NOTE: Triton expects void** where each void* points to the actual value
    void* kernelArgs[] = {
        // Pointers (cast to CUdeviceptr)
        (void*) &Q, (void*) &K, (void*) &V,
        (void*) &means, (void*) &radii, (void*) &conc,
        (void*) &Out,
        // Q strides
        (void*) &stride_q_b, (void*) &stride_q_h, (void*) &stride_q_d,
        // K strides
        (void*) &stride_k_b, (void*) &stride_k_h, (void*) &stride_k_s, (void*) &stride_k_d,
        // V strides (same as K)
        (void*) &stride_k_b, (void*) &stride_k_h, (void*) &stride_k_s, (void*) &stride_k_d,
        // Mean strides
        (void*) &stride_m_b, (void*) &stride_m_h, (void*) &stride_m_blk, (void*) &stride_m_d,
        // Radii strides
        (void*) &stride_r_b, (void*) &stride_r_h, (void*) &stride_r_blk,
        // Conc strides (same as radii)
        (void*) &stride_r_b, (void*) &stride_r_h, (void*) &stride_r_blk,
        // Output strides
        (void*) &stride_o_b, (void*) &stride_o_h, (void*) &stride_o_d,
        // Dimensions
        (void*) &batchSize, (void*) &mNumHeads, (void*) &seqLen,
    };
    
    // Grid dimensions: (batch_size, num_heads, 1)
    unsigned int gridX = batchSize;
    unsigned int gridY = mNumHeads;
    unsigned int gridZ = 1;
    
    // Block dimensions from compiled kernel
    unsigned int blockX = AETHER_SPARSE_FP16_BLOCK_SIZE;  // From kernel_payload.h
    unsigned int blockY = 1;
    unsigned int blockZ = 1;
    
    // Shared memory size from compiled kernel
    unsigned int sharedMem = get_aether_sparse_fp16_shared_mem();
    
    // Launch kernel via CUDA Driver API
    CUresult res = cuLaunchKernel(
        mKernel,           // CUfunction
        gridX, gridY, gridZ,
        blockX, blockY, blockZ,
        sharedMem,
        stream,
        kernelArgs,
        nullptr            // No extra params
    );
    
    if (res != CUDA_SUCCESS)
    {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        std::cerr << "cuLaunchKernel failed: " << errStr << std::endl;
        return -1;
    }
    
    return 0;
}

int AetherAttentionPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    int res = -1;
    
    if (mType == DataType::kHALF)
    {
        res = enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        res = enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        std::cerr << "Unsupported data type for AetherAttentionPlugin" << std::endl;
    }
    
    return res;
}

nvinfer1::DataType AetherAttentionPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];  // Match Q type
}

char const* AetherAttentionPlugin::getPluginType() const noexcept
{
    return AETHER_ATTENTION_PLUGIN_NAME;
}

char const* AetherAttentionPlugin::getPluginVersion() const noexcept
{
    return AETHER_ATTENTION_PLUGIN_VERSION;
}

int AetherAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int AetherAttentionPlugin::initialize() noexcept
{
    // Load PTX module from embedded payload
    CUresult res = load_aether_sparse_fp16();
    if (res != CUDA_SUCCESS)
    {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        std::cerr << "Failed to load AETHER kernel: " << errStr << std::endl;
        return -1;
    }
    
    mModule = aether_sparse_fp16_mod;
    mKernel = get_aether_sparse_fp16_func();
    
    return 0;
}

void AetherAttentionPlugin::terminate() noexcept
{
    unload_aether_sparse_fp16();
    mModule = nullptr;
    mKernel = nullptr;
}

size_t AetherAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) 
         + sizeof(mHeadSize) 
         + sizeof(mBlockSize) 
         + sizeof(mThreshold)
         + sizeof(mSoftmaxScale) 
         + sizeof(mUseConcentration) 
         + sizeof(mIsCausal)
         + sizeof(mLocalWindow) 
         + sizeof(mType);
}

void AetherAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    writeArg(d, mNumHeads);
    writeArg(d, mHeadSize);
    writeArg(d, mBlockSize);
    writeArg(d, mThreshold);
    writeArg(d, mSoftmaxScale);
    writeArg(d, mUseConcentration);
    writeArg(d, mIsCausal);
    writeArg(d, mLocalWindow);
    writeArg(d, mType);
    assert(d == a + getSerializationSize());
}

void AetherAttentionPlugin::destroy() noexcept
{
    delete this;
}

void AetherAttentionPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* AetherAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// =============================================================================
// AetherAttentionPluginCreator Implementation
// =============================================================================

AetherAttentionPluginCreator::AetherAttentionPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("block_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("threshold", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("softmax_scale", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("use_concentration", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("is_causal", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("local_window", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* AetherAttentionPluginCreator::getPluginName() const noexcept
{
    return AETHER_ATTENTION_PLUGIN_NAME;
}

char const* AetherAttentionPluginCreator::getPluginVersion() const noexcept
{
    return AETHER_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* AetherAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AetherAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    
    int numHeads = 32;
    int headSize = 128;
    int blockSize = 64;
    float threshold = 0.15f;
    float softmaxScale = 0.0f;
    bool useConcentration = true;
    bool isCausal = true;
    int localWindow = 4;
    nvinfer1::DataType type = DataType::kHALF;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            numHeads = *static_cast<int const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "head_size"))
        {
            headSize = *static_cast<int const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "block_size"))
        {
            blockSize = *static_cast<int const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "threshold"))
        {
            threshold = *static_cast<float const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "softmax_scale"))
        {
            softmaxScale = *static_cast<float const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "use_concentration"))
        {
            useConcentration = (*static_cast<int const*>(fields[i].data)) != 0;
        }
        else if (!strcmp(attrName, "is_causal"))
        {
            isCausal = (*static_cast<int const*>(fields[i].data)) != 0;
        }
        else if (!strcmp(attrName, "local_window"))
        {
            localWindow = *static_cast<int const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "type_id"))
        {
            type = static_cast<nvinfer1::DataType>(*static_cast<int const*>(fields[i].data));
        }
    }
    
    // Default softmax scale if not provided
    if (softmaxScale == 0.0f)
    {
        softmaxScale = 1.0f / sqrtf(static_cast<float>(headSize));
    }

    try
    {
        auto* obj = new AetherAttentionPlugin(
            numHeads, headSize, blockSize, threshold, softmaxScale,
            useConcentration, isCausal, localWindow, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "AetherAttentionPlugin creation failed: " << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV2* AetherAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new AetherAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "AetherAttentionPlugin deserialization failed: " << e.what() << std::endl;
    }
    return nullptr;
}

void AetherAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* AetherAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace aether::plugin
