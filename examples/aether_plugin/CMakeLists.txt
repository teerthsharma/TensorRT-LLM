# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CMakeLists.txt for AETHER Sparse Attention TensorRT Plugin

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aether_attention_plugin CUDA CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
set(SM_VERSION "89" CACHE STRING "CUDA SM version")
set(CMAKE_CUDA_ARCHITECTURES ${SM_VERSION})

# Find packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT
if(NOT DEFINED TRT_ROOT)
    set(TRT_ROOT /usr/local/tensorrt CACHE PATH "TensorRT root directory")
endif()

find_path(TRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TRT_ROOT}/include ${CUDA_TOOLKIT_ROOT_DIR}/../TensorRT/include
    DOC "TensorRT include directory"
)

find_library(TRT_LIB
    NAMES nvinfer
    PATHS ${TRT_ROOT}/lib ${CUDA_TOOLKIT_ROOT_DIR}/../TensorRT/lib
    DOC "TensorRT library"
)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TRT_ROOT to TensorRT installation directory.")
endif()

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib: ${TRT_LIB}")

# TensorRT-LLM common headers (optional, for helper macros)
if(DEFINED TRTLLM_ROOT)
    set(TRTLLM_INCLUDE_DIRS
        ${TRTLLM_ROOT}/cpp/include
        ${TRTLLM_ROOT}/cpp/tensorrt_llm/common
    )
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/aot
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRTLLM_INCLUDE_DIRS}
)

# Source files
set(PLUGIN_SOURCES
    AetherAttentionPlugin.cpp
    tritonPlugins.cpp
)

# Create shared library
add_library(aether_attention_plugin SHARED ${PLUGIN_SOURCES})

# Link libraries
target_link_libraries(aether_attention_plugin
    ${TRT_LIB}
    CUDA::cuda_driver
    CUDA::cudart
)

# Compiler flags
target_compile_options(aether_attention_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Preprocessor definitions
target_compile_definitions(aether_attention_plugin PRIVATE
    SM_VERSION=${SM_VERSION}
)

# Install
install(TARGETS aether_attention_plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES 
    AetherAttentionPlugin.h
    kernel_payload.h
    DESTINATION include/aether
)

# Print configuration
message(STATUS "")
message(STATUS "=== AETHER Plugin Configuration ===")
message(STATUS "SM Version: ${SM_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "===================================")
