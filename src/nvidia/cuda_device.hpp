#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <string>

/** execute and check a CUDA api command
*
* @param id gpu id (thread id)
* @param ... CUDA api command
*/
#define CUDA_CHECK(id, ...) {                                                                             \
    cudaError_t error = __VA_ARGS__;                                                                      \
    if(error!=cudaSuccess){                                                                               \
        std::cerr << "[CUDA] Error gpu " << id << ": <" << __FUNCTION__ << ">:" << __LINE__ << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
        throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error))); \
    }                                                                                                     \
}                                                                                                         \
( (void) 0 )

/** execute and check a CUDA kernel
*
* @param id gpu id (thread id)
* @param ... CUDA kernel call
*/
#define CUDA_CHECK_KERNEL(id, ...)      \
    __VA_ARGS__;                        \
    CUDA_CHECK(id, cudaGetLastError())

#define CU_CHECK(id, ...) {                                                                             \
    CUresult result = __VA_ARGS__;                                                                      \
    if(result != CUDA_SUCCESS){                                                                         \
        const char* s;                                                                                  \
        cuGetErrorString(result, &s);                                                                   \
        std::cerr << "[CUDA] Error gpu " << ctx->device_id << ": <" << __FUNCTION__ << ">:" << __LINE__ << " \"" << (s ? s : "unknown error") << "\"" << std::endl; \
        throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(s ? s : "unknown error")); \
    }                                                                                                   \
}                                                                                                       \
( (void) 0 )

