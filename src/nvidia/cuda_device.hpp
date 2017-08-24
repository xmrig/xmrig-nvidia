
#pragma once

#include <cuda_runtime.h>


static inline void exit_if_cudaerror(int thr_id, const char *fn, int line)
{
    const cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        return;
    }

    printf("\nGPU #%d: %s\n%s line %d\n", thr_id, cudaGetErrorString(err), fn, line);
    exit(1);
}
