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
 * AETHER Sparse Attention Plugin for TensorRT-LLM
 * ================================================
 * Custom plugin that loads Triton-compiled PTX and invokes via CUDA Driver API.
 *
 * Reference:
 *   Sharma, T. (2024). AETHER - Adaptive Event-driven Threshold Hybrid
 *   Entangled Rendering. DOI: 10.13141/RG.2.2.14811.27684
 */
#pragma once

#include <NvInferRuntime.h>

#include <cassert>
#include <set>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace aether::plugin
{

/**
 * @brief AETHER Sparse Attention Plugin
 * 
 * Implements block-sparse attention using Event Radar scoring.
 * Loads Triton-compiled PTX at initialization and invokes via cuLaunchKernel.
 * 
 * Inputs:
 *   0: Q - Query tensor (B, H, D)
 *   1: K - Key cache (B, H, S, D)
 *   2: V - Value cache (B, H, S, D)
 *   3: block_means - Precomputed block centroids (B, H, N_blocks, D)
 *   4: block_radii - Precomputed block radii (B, H, N_blocks)
 *   5: block_conc - Precomputed concentrations (B, H, N_blocks)
 *   6: block_table (optional) - Paged KV cache indirection (B, max_blocks)
 * 
 * Outputs:
 *   0: Out - Attention output (B, H, D)
 */
class AetherAttentionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    AetherAttentionPlugin(
        int numHeads,
        int headSize,
        int blockSize,
        float threshold,
        float softmaxScale,
        bool useConcentration,
        bool isCausal,
        int localWindow,
        nvinfer1::DataType type);

    AetherAttentionPlugin(void const* data, size_t length);

    ~AetherAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    template <typename T>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;

    // Plugin parameters
    int mNumHeads;
    int mHeadSize;
    int mBlockSize;
    float mThreshold;
    float mSoftmaxScale;
    bool mUseConcentration;
    bool mIsCausal;
    int mLocalWindow;
    nvinfer1::DataType mType;

    // CUDA module handles (loaded from embedded PTX)
    CUmodule mModule;
    CUfunction mKernel;
    
    // Flag for paged KV cache mode
    bool mUsePagedKVCache;
};

/**
 * @brief Plugin Creator for AetherAttentionPlugin
 */
class AetherAttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    AetherAttentionPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace aether::plugin
