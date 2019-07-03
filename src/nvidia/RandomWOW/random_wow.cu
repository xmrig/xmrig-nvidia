/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "randomx.h"
#include "configuration.h"
#include "../cryptonight.h"
#include "../cuda_device.hpp"
#include "../workers/Workers.h"

namespace randomx {
    constexpr int mantissaSize = 52;
    constexpr int exponentSize = 11;
    constexpr uint64_t mantissaMask = (1ULL << mantissaSize) - 1;
    constexpr uint64_t exponentMask = (1ULL << exponentSize) - 1;
    constexpr int exponentBias = 1023;
    constexpr int dynamicExponentBits = 4;
    constexpr int staticExponentBits = 4;
    constexpr uint64_t constExponentBits = 0x300;
    constexpr uint64_t dynamicMantissaMask = (1ULL << (mantissaSize + dynamicExponentBits)) - 1;

    constexpr int RegistersCount = 8;
    constexpr int RegisterCountFlt = RegistersCount / 2;
    constexpr int RegisterNeedsDisplacement = 5; //x86 r13 register

    constexpr int CacheLineSize = RANDOMX_DATASET_ITEM_SIZE;
    constexpr uint32_t DatasetExtraItems = RANDOMX_DATASET_EXTRA_SIZE / RANDOMX_DATASET_ITEM_SIZE;

    constexpr uint32_t ConditionMask = ((1 << RANDOMX_JUMP_BITS) - 1);
    constexpr int ConditionOffset = RANDOMX_JUMP_OFFSET;
    constexpr int StoreL3Condition = 14;
}

#include "blake2b_cuda.hpp"
#include "aes_cuda.hpp"
#include "randomx_cuda.hpp"

void randomx_prepare(nvid_ctx *ctx, const uint8_t* seed_hash, xmrig::Variant variant, uint32_t batch_size)
{
    const size_t dataset_size = randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;
    if (!ctx->d_rx_dataset) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_dataset, dataset_size));
    }
    if (!ctx->d_rx_hashes) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_hashes, batch_size * 64));
    }
    if (!ctx->d_rx_entropy) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_entropy, batch_size * (128 + 2048)));
    }
    if (!ctx->d_rx_vm_states) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_vm_states, batch_size * 2048));
    }
    if (!ctx->d_rx_rounding) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_rounding, batch_size * sizeof(uint32_t)));
    }

    randomx_dataset* dataset = Workers::getDataset(seed_hash, variant);
    if ((memcmp(ctx->rx_dataset_seedhash, seed_hash, sizeof(ctx->rx_dataset_seedhash)) != 0) || (ctx->rx_variant != variant)) {
        memcpy(ctx->rx_dataset_seedhash, seed_hash, sizeof(ctx->rx_dataset_seedhash));
        ctx->rx_variant = variant;
        CUDA_CHECK(ctx->device_id, cudaMemcpy(ctx->d_rx_dataset, randomx_get_dataset_memory(dataset), dataset_size, cudaMemcpyHostToDevice));
    }
}

__global__ void find_shares(const void* hashes, uint64_t target, uint32_t* shares)
{
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t* p = (const uint64_t*) hashes;

    if (p[global_index * 4 + 3] < target) {
        const uint32_t idx = atomicInc(shares, 0xFFFFFFFF) + 1;
        if (idx < 10) {
            shares[idx] = global_index;
        }
    }
}

void randomx_hash(nvid_ctx *ctx, uint32_t nonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, uint32_t batch_size)
{
    CUDA_CHECK_KERNEL(ctx->device_id, blake2b_initial_hash<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_input, ctx->inputlen, nonce));
    CUDA_CHECK_KERNEL(ctx->device_id, fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_long_state, batch_size));
    CUDA_CHECK(ctx->device_id, cudaMemset(ctx->d_rx_rounding, 0, batch_size * sizeof(uint32_t)));

    for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i) {
        CUDA_CHECK_KERNEL(ctx->device_id, fillAes4Rx4<ENTROPY_SIZE, false><<<batch_size / 32, 32 * 4>>>(ctx->d_rx_hashes, ctx->d_rx_entropy, batch_size));

        CUDA_CHECK_KERNEL(ctx->device_id, init_vm<8><<<batch_size / 4, 4 * 8>>>(ctx->d_rx_entropy, ctx->d_rx_vm_states));
        for (int j = 0, n = 1 << ctx->device_bfactor; j < n; ++j) {
            CUDA_CHECK_KERNEL(ctx->device_id, execute_vm<8, false> << <batch_size / 2, 2 * 8 >> > (ctx->d_rx_vm_states, ctx->d_rx_rounding, ctx->d_long_state, ctx->d_rx_dataset, batch_size, RANDOMX_PROGRAM_ITERATIONS >> ctx->device_bfactor, j == 0, j == n - 1));
        }

        if (i == RANDOMX_PROGRAM_COUNT - 1) {
            CUDA_CHECK_KERNEL(ctx->device_id, hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, VM_STATE_SIZE, 64><<<batch_size / 32, 32 * 4>>>(ctx->d_long_state, ctx->d_rx_vm_states, batch_size));
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 32><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        } else {
            CUDA_CHECK_KERNEL(ctx->device_id, blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 64><<<batch_size / 32, 32>>>(ctx->d_rx_hashes, ctx->d_rx_vm_states));
        }
    }

    CUDA_CHECK(ctx->device_id, cudaMemset(ctx->d_result_nonce, 0, 10 * sizeof(uint32_t)));
    CUDA_CHECK_KERNEL(ctx->device_id, find_shares<<<batch_size / 32, 32>>>(ctx->d_rx_hashes, target, ctx->d_result_nonce));
    CUDA_CHECK(ctx->device_id, cudaDeviceSynchronize());

    CUDA_CHECK(ctx->device_id, cudaMemcpy(resnonce, ctx->d_result_nonce, 10 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    *rescount = resnonce[0];
    if (*rescount > 9) {
        *rescount = 9;
    }

    for (int i = 0; i < *rescount; i++) {
        resnonce[i] = resnonce[i + 1] + nonce;
    }
}
