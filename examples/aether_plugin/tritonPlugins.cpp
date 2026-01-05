/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AETHER Plugin Registration
 */
#include "AetherAttentionPlugin.h"
#include <NvInferRuntime.h>

namespace aether::plugin
{

// Extern creator instance
static AetherAttentionPluginCreator aetherAttentionPluginCreator;

extern "C"
{

// Plugin initialization function - called by TensorRT LLM
bool initAetherPlugins(void* logger, const char* libNamespace)
{
    nvinfer1::ILogger* trtLogger = static_cast<nvinfer1::ILogger*>(logger);
    
    // Register plugin creator
    getPluginRegistry()->registerCreator(aetherAttentionPluginCreator, libNamespace);
    
    if (trtLogger)
    {
        trtLogger->log(nvinfer1::ILogger::Severity::kINFO, 
            "AETHER Sparse Attention plugin registered successfully");
    }
    
    return true;
}

} // extern "C"

} // namespace aether::plugin

// Macro for auto-registration with TensorRT
REGISTER_TENSORRT_PLUGIN(aether::plugin::AetherAttentionPluginCreator);
