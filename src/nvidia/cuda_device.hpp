
#pragma once

#include <cuda_runtime.h>

static inline void exit_if_cudaerror(int thr_id, const char *file, int line)
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("\nGPU %d: %s\n%s line %d\n", thr_id, cudaGetErrorString(err), file, line);
		exit(1);
	}
}
