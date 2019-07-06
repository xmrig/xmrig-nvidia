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
#include "../CryptoNight_constants.h"
#include "../cryptonight.h"
#include "../cuda_device.hpp"
#include "../workers/Workers.h"

void randomx_prepare(nvid_ctx *ctx, const uint8_t* seed_hash, xmrig::Variant variant, uint32_t batch_size)
{
    randomx_dataset* dataset = Workers::getDataset(seed_hash, variant);
    const size_t dataset_size = randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;
    if (!ctx->d_rx_dataset) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_dataset, dataset_size));
    }
    if (!ctx->d_long_state) {
        ctx->d_scratchpads_size = batch_size * rx_select_memory(variant);
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_long_state, ctx->d_scratchpads_size));
    }
    if (!ctx->d_rx_hashes) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_hashes, batch_size * 64));
    }
    if (!ctx->d_rx_entropy) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_entropy, batch_size * (128 + 2560)));
    }
    if (!ctx->d_rx_vm_states) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_vm_states, batch_size * 2560));
    }
    if (!ctx->d_rx_rounding) {
        CUDA_CHECK(ctx->device_id, cudaMalloc(&ctx->d_rx_rounding, batch_size * sizeof(uint32_t)));
    }

    if ((memcmp(ctx->rx_dataset_seedhash, seed_hash, sizeof(ctx->rx_dataset_seedhash)) != 0) || (ctx->rx_variant != variant)) {
        memcpy(ctx->rx_dataset_seedhash, seed_hash, sizeof(ctx->rx_dataset_seedhash));
        ctx->rx_variant = variant;
        CUDA_CHECK(ctx->device_id, cudaMemcpy(ctx->d_rx_dataset, randomx_get_dataset_memory(dataset), dataset_size, cudaMemcpyHostToDevice));
    }
}
