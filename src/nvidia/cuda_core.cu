/* XMRig
* Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
* Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
* Copyright 2014      Lucas Jones <https://github.com/lucasjones>
* Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
* Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
* Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
* Copyright 2018      Lee Clagett <https://github.com/vtnerd>
* Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaCryptonightR_gen.h"
#include "common/log/Log.h"
#include "common/utils/timestamp.h"

#ifdef _WIN32
#include <Windows.h>
static void compat_usleep(int waitTime)
{
    if (waitTime > 0) {
        if (waitTime > 100) {
            // use a waitable timer for larger intervals > 0.1ms

            HANDLE timer;
            LARGE_INTEGER ft;

            ft.QuadPart = -10ll * int64_t(waitTime); // Convert to 100 nanosecond interval, negative value indicates relative time

            timer = CreateWaitableTimer(nullptr, TRUE, nullptr);
            SetWaitableTimer(timer, &ft, 0, nullptr, nullptr, 0);
            WaitForSingleObject(timer, INFINITE);
            CloseHandle(timer);
        } else {
            // use a polling loop for short intervals <= 0.1ms

            LARGE_INTEGER perfCnt, start, now;
            int64_t elapsed;

            QueryPerformanceFrequency(&perfCnt);
            QueryPerformanceCounter(&start);
            do {
                SwitchToThread();
                QueryPerformanceCounter(&now);
                elapsed = static_cast<int64_t>(((now.QuadPart - start.QuadPart) / static_cast<float>(perfCnt.QuadPart) * 1000 * 1000));
            } while (elapsed < static_cast<int64_t>(waitTime));
        }
    }
}
#else
#include <unistd.h>
static inline void compat_usleep(int waitTime)
{
    usleep(static_cast<uint64_t>(waitTime));
}
#endif

#include "cryptonight.h"
#include "cuda_extra.h"
#include "cuda_aes.hpp"
#include "cuda_device.hpp"
#include "cuda_fast_int_math_v2.hpp"
#include "cuda_fast_div_heavy.hpp"
#include "cuda_cryptonight_gpu.hpp"
#include "common/xmrig.h"
#include "crypto/CryptoNight_constants.h"

#if defined(__x86_64__) || defined(_M_AMD64) || defined(__LP64__)
#   define _ASM_PTR_ "l"
#else
#   define _ASM_PTR_ "r"
#endif

/* sm_2X is limited to 2GB due to the small TLB
 * therefore we never use 64bit indices
 */
#if defined(XMR_STAK_LARGEGRID) && (__CUDA_ARCH__ >= 300)
typedef uint64_t IndexType;
#else
typedef int IndexType;
#endif

template< typename T >
__device__ __forceinline__ T loadGlobal64( T * const addr )
{
#   if (__CUDA_ARCH__ < 700)
    T x;
    asm volatile( "ld.global.cg.u64 %0, [%1];" : "=l"( x ) : _ASM_PTR_(addr));
    return x;
#   else
    return *addr;
#   endif
}

template< typename T >
__device__ __forceinline__ T loadGlobal32( T * const addr )
{
#   if (__CUDA_ARCH__ < 700)
    T x;
    asm volatile( "ld.global.cg.u32 %0, [%1];" : "=r"( x ) : _ASM_PTR_(addr));
    return x;
#   else
    return *addr;
#   endif
}

template< typename T >
__device__ __forceinline__ void storeGlobal32( T* addr, T const & val )
{
#   if (__CUDA_ARCH__ < 700)
    asm volatile( "st.global.cg.u32 [%0], %1;" : : _ASM_PTR_(addr), "r"( val ) );
#   else
    *addr = val;
#   endif
}

template< typename T >
__device__ __forceinline__ void storeGlobal64( T* addr, T const & val )
{
#   if (__CUDA_ARCH__ < 700)
    asm volatile("st.global.cg.u64 [%0], %1;" : : _ASM_PTR_(addr), "l"(val));
#   else
    *addr = val;
#   endif
}

template<size_t ITERATIONS, uint32_t MEM>
__global__ void cryptonight_core_gpu_phase1( int threads, int bfactor, int partidx, uint32_t * __restrict__ long_state, uint32_t * __restrict__ ctx_state2, uint32_t * __restrict__ ctx_key1 )
{
    __shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init( sharedMemory );
    __syncthreads( );

    const int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 3;
    const int sub = ( threadIdx.x & 7 ) << 2;

    const int batchsize = MEM >> bfactor;
    const int start = partidx * batchsize;
    const int end = start + batchsize;

    if ( thread >= threads )
        return;

    uint32_t key[40], text[4];

    MEMCPY8( key, ctx_key1 + thread * 40, 20 );

    if (partidx == 0) {
        // first round
        MEMCPY8( text, ctx_state2 + thread * 50 + sub + 16, 2 );
    }
    else {
        // load previous text data
        MEMCPY8( text, &long_state[( (uint64_t) thread * MEM) + sub + start - 32], 2 );
    }

    __syncthreads( );
    for (int i = start; i < end; i += 32) {
        cn_aes_pseudo_round_mut( sharedMemory, text, key );
        MEMCPY8(&long_state[((uint64_t) thread * MEM) + (sub + i)], text, 2);
    }
}

/** avoid warning `unused parameter` */
template< typename T >
__forceinline__ __device__ void unusedVar( const T& )
{
}

/** shuffle data for
 *
 * - this method can be used with all compute architectures
 * - for <sm_30 shared memory is needed
 *
 * group_n - must be a power of 2!
 *
 * @param ptr pointer to shared memory, size must be `threadIdx.x * sizeof(uint32_t)`
 *            value can be NULL for compute architecture >=sm_30
 * @param sub thread number within the group, range [0:group_n]
 * @param value value to share with other threads within the group
 * @param src thread number within the group from where the data is read, range [0:group_n]
 */
template<size_t group_n>
__forceinline__ __device__ uint32_t shuffle(volatile uint32_t* ptr,const uint32_t sub,const int val,const uint32_t src)
{
#   if ( __CUDA_ARCH__ < 300 )
    ptr[sub] = val;
    return ptr[src & (group_n-1)];
#   else
    unusedVar( ptr );
    unusedVar( sub );
#   if (__CUDACC_VER_MAJOR__ >= 9)
    return __shfl_sync(__activemask(), val, src, group_n);
#   else
    return __shfl( val, src, group_n );
#   endif
#   endif
}


template<size_t group_n>
__forceinline__ __device__ uint64_t shuffle64(volatile uint32_t* ptr,const uint32_t sub,const int val,const uint32_t src, const uint32_t src2)
{
    uint64_t tmp;
    ((uint32_t*)&tmp)[0] = shuffle<group_n>(ptr, sub, val, src);
    ((uint32_t*)&tmp)[1] = shuffle<group_n>(ptr, sub, val, src2);
    return tmp;
}

struct u64 : public uint2
{

    __forceinline__ __device__ u64(){}

    __forceinline__ __device__ u64( const uint32_t x0, const uint32_t x1)
    {
        uint2::x = x0;
        uint2::y = x1;
    }

    __forceinline__ __device__ operator uint64_t() const
    {
        return *((uint64_t*)this);
    }

    __forceinline__ __device__ u64( const uint64_t x0)
    {
        ((uint64_t*)&this->x)[0] = x0;
    }

    __forceinline__ __device__ u64 operator^=(const u64& other)
    {
        uint2::x ^= other.x;
        uint2::y ^= other.y;

        return *this;
    }

    __forceinline__ __device__ u64 operator+(const u64& other) const
    {
        u64 tmp;
        ((uint64_t*)&tmp.x)[0] = ((uint64_t*)&(this->x))[0] + ((uint64_t*)&(other.x))[0];

        return tmp;
    }

    __forceinline__ __device__ u64 operator+=(const uint64_t& other)
    {
        return ((uint64_t*)&this->x)[0] += other;
    }

    __forceinline__ __device__ void print(int i) const
    {
        if(i<2)
            printf("gpu: %lu\n", ((uint64_t*)&this->x)[0]);
    }
};

/** cryptonight with two threads per hash
 */
template<size_t ITERATIONS, uint32_t MEM, uint32_t MASK, xmrig::Algo ALGO, xmrig::Variant VARIANT>
#ifdef XMR_STAK_THREADS
__launch_bounds__( XMR_STAK_THREADS * 2 )
#endif
__global__ void cryptonight_core_gpu_phase2_double(
        int threads,
        int bfactor,
        int partidx,
        uint32_t *d_long_state,
        uint32_t *d_ctx_a,
        uint32_t *d_ctx_b,
        uint32_t * d_ctx_state,
        uint32_t startNonce,
        uint32_t * __restrict__ d_input
        )
{
    __shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init( sharedMemory );

#   if( __CUDA_ARCH__ < 300 )
    extern __shared__ uint64_t externShared[];
    // 8 x 64bit values
    volatile uint64_t* myChunks = (volatile uint64_t*)(externShared + (threadIdx.x >> 1) * 8);
    volatile uint32_t* sPtr = (volatile uint32_t*)(externShared + (blockDim.x >> 1) * 8)  + (threadIdx.x & 0xFFFFFFFE);
#   else
    extern __shared__ uint64_t chunkMem[];
    volatile uint32_t* sPtr = NULL;
    // 8 x 64bit values
    volatile uint64_t* myChunks = (volatile uint64_t*)(chunkMem + (threadIdx.x >> 1) * 8);
#   endif

    __syncthreads( );

    const uint64_t tid    = (blockDim.x * blockIdx.x + threadIdx.x);
    const uint32_t thread = tid >> 1;
    const uint32_t sub    = tid & 1;

    if (thread >= threads) {
        return;
    }

    uint8_t *l0              = (uint8_t*)&d_long_state[(IndexType) thread * MEM];
    uint64_t ax0             = ((uint64_t*)(d_ctx_a + thread * 4))[sub];
    uint32_t idx0            = shuffle<2>(sPtr, sub, static_cast<uint32_t>(ax0), 0);
    uint64_t bx0             = ((uint64_t*)(d_ctx_b + thread * 16))[sub];
    uint64_t bx1             = ((uint64_t*)(d_ctx_b + thread * 16 + 4))[sub];
    uint64_t division_result = ((uint64_t*)(d_ctx_b + thread * 16 + 4 * 2))[0];
    uint32_t sqrt_result     = (d_ctx_b + thread * 16 + 4 * 2 + 2)[0];

    const int batchsize      = (ITERATIONS * 2) >> ( 1 + bfactor );
    const int start          = partidx * batchsize;
    const int end            = start + batchsize;

    uint64_t* ptr0;
    for (int i = start; i < end; ++i) {
        ptr0 = (uint64_t *)&l0[idx0 & (MASK - 0x30)];

        ((ulonglong4*)myChunks)[sub] = ((ulonglong4*)ptr0)[sub];

        uint32_t idx1 = (idx0 & 0x30) >> 3;
        const u64 cx  = myChunks[ idx1 + sub ];
        const u64 cx2 = myChunks[ idx1 + ((sub + 1) & 1) ];

        u64 cx_aes = ax0 ^ u64(
            t_fn0( cx.x & 0xff ) ^ t_fn1( (cx.y >> 8) & 0xff ) ^ t_fn2( (cx2.x >> 16) & 0xff ) ^ t_fn3( (cx2.y >> 24 ) ),
            t_fn0( cx.y & 0xff ) ^ t_fn1( (cx2.x >> 8) & 0xff ) ^ t_fn2( (cx2.y >> 16) & 0xff ) ^ t_fn3( (cx.x >> 24 ) )
        );

        {
            const uint64_t chunk1 = myChunks[idx1 ^ 2 + sub];
            const uint64_t chunk2 = myChunks[idx1 ^ 4 + sub];
            const uint64_t chunk3 = myChunks[idx1 ^ 6 + sub];

#           if (__CUDACC_VER_MAJOR__ >= 9)
            __syncwarp();
#           else
            __syncthreads();
#           endif

            myChunks[idx1 ^ 2 + sub] = ((VARIANT == xmrig::VARIANT_RWZ) ? chunk1 : chunk3) + bx1;
            myChunks[idx1 ^ 4 + sub] = ((VARIANT == xmrig::VARIANT_RWZ) ? chunk3 : chunk1) + bx0;
            myChunks[idx1 ^ 6 + sub] = chunk2 + ax0;
        }

        myChunks[idx1 + sub] = cx_aes ^ bx0;

        ((ulonglong4*)ptr0)[sub] = ((ulonglong4*)myChunks)[sub];

        idx0 = shuffle<2>(sPtr, sub, cx_aes.x, 0);
        idx1 = (idx0 & 0x30) >> 3;
        ptr0 = (uint64_t *)&l0[idx0 & MASK & (MASK - 0x30)];

        ((ulonglong4*)myChunks)[sub] = ((ulonglong4*)ptr0)[sub];

        uint64_t cx_mul;
        ((uint32_t*)&cx_mul)[0] = shuffle<2>(sPtr, sub, cx_aes.x , 0);
        ((uint32_t*)&cx_mul)[1] = shuffle<2>(sPtr, sub, cx_aes.y , 0);

        if (sub == 1) {
            // Use division and square root results from the _previous_ iteration to hide the latency
            ((uint32_t*)&division_result)[1] ^= sqrt_result;
            ((uint64_t*)myChunks)[idx1]      ^= division_result;

            const uint32_t dd = (static_cast<uint32_t>(cx_mul) + (sqrt_result << 1)) | 0x80000001UL;
            division_result = fast_div_v2(cx_aes, dd);

            // Use division_result as an input for the square root to prevent parallel implementation in hardware
            sqrt_result = fast_sqrt_v2(cx_mul + division_result);
        }

#       if (__CUDACC_VER_MAJOR__ >= 9)
        __syncwarp();
#       else
        __syncthreads( );
#       endif

        uint64_t c = ((uint64_t*)myChunks)[idx1 + sub];

        {
            uint64_t cl = ((uint64_t*)myChunks)[idx1];
            // sub 0 -> hi, sub 1 -> lo
            uint64_t res = sub == 0 ? __umul64hi( cx_mul, cl ) : cx_mul * cl;

            const uint64_t chunk1 = myChunks[ idx1 ^ 2 + sub ] ^ res;
            uint64_t chunk2       = myChunks[ idx1 ^ 4 + sub ];
            res ^= ((uint64_t*)&chunk2)[0];
            const uint64_t chunk3 = myChunks[ idx1 ^ 6 + sub ];

#           if (__CUDACC_VER_MAJOR__ >= 9)
            __syncwarp();
#           else
            __syncthreads( );
#           endif

            myChunks[idx1 ^ 2 + sub] = ((VARIANT == xmrig::VARIANT_RWZ) ? chunk1 : chunk3) + bx1;
            myChunks[idx1 ^ 4 + sub] = ((VARIANT == xmrig::VARIANT_RWZ) ? chunk3 : chunk1) + bx0;
            myChunks[idx1 ^ 6 + sub] = chunk2 + ax0;

            ax0 += res;
        }

        bx1 = bx0;
        bx0 = cx_aes;

        myChunks[idx1 + sub] = ax0;

        ((ulonglong4*)ptr0)[sub] = ((ulonglong4*)myChunks)[sub];

        ax0 ^= c;
        idx0 = shuffle<2>(sPtr, sub, static_cast<uint32_t>(ax0), 0);
    }

    if (bfactor > 0) {
        ((uint64_t*)(d_ctx_a + thread * 4))[sub]      = ax0;
        ((uint64_t*)(d_ctx_b + thread * 16))[sub]     = bx0;
        ((uint64_t*)(d_ctx_b + thread * 16 + 4))[sub] = bx1;

        if (sub == 1) {
            // must be valid only for `sub == 1`
            ((uint64_t*)(d_ctx_b + thread * 16 + 4 * 2))[0] = division_result;
            (d_ctx_b + thread * 16 + 4 * 2 + 2)[0]          = sqrt_result;
        }
    }
}


template<size_t ITERATIONS, uint32_t MEM, uint32_t MASK, xmrig::Algo ALGO, xmrig::Variant VARIANT, xmrig::Variant BASE>
#ifdef XMR_STAK_THREADS
__launch_bounds__( XMR_STAK_THREADS * 4 )
#endif
__global__ void cryptonight_core_gpu_phase2_quad(
        int threads,
        int bfactor,
        int partidx,
        uint32_t *d_long_state,
        uint32_t *d_ctx_a,
        uint32_t *d_ctx_b,
        uint32_t *d_ctx_state,
        uint32_t startNonce,
        uint32_t *__restrict__ d_input
        )
{
    __shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init( sharedMemory );

    __syncthreads( );

    const int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 2;
    const uint32_t nonce = startNonce + thread;
    const int sub = threadIdx.x & 3;
    const int sub2 = sub & 2;

#if( __CUDA_ARCH__ < 300 )
        extern __shared__ uint32_t shuffleMem[];
        volatile uint32_t* sPtr = (volatile uint32_t*)(shuffleMem + (threadIdx.x& 0xFFFFFFFC));
#else
        volatile uint32_t* sPtr = NULL;
#endif
    if (thread >= threads) {
        return;
    }

    int i, k;
    uint32_t j;
    const int batchsize = (ITERATIONS * 2) >> (2 + bfactor);
    const int start = partidx * batchsize;
    const int end = start + batchsize;
    uint32_t * long_state = &d_long_state[(IndexType) thread * MEM];
    uint32_t a, d[2], idx0;
    uint32_t t1[2], t2[2], res;

    uint32_t tweak1_2[2];
    if (BASE == xmrig::VARIANT_1) {
        uint32_t * state = d_ctx_state + thread * 50;
        tweak1_2[0] = (d_input[8] >> 24) | (d_input[9] << 8);
        tweak1_2[0] ^= state[48];
        tweak1_2[1] = nonce;
        tweak1_2[1] ^= state[49];
    }

    a = (d_ctx_a + thread * 4)[sub];
    idx0 = shuffle<4>(sPtr,sub, a, 0);
    if (ALGO == xmrig::CRYPTONIGHT_HEAVY) {
        if (partidx != 0) {
            // state is stored after all ctx_b states
            idx0 = *(d_ctx_b + threads * 4 + thread);
        }
    }

    d[1] = (d_ctx_b + thread * 4)[sub];

    #pragma unroll 2
    for (i = start; i < end; ++i) {
        #pragma unroll 2
        for (int x = 0; x < 2; ++x) {
            j = ((idx0 & MASK) >> 2) + sub;

            if (VARIANT == xmrig::VARIANT_TUBE) {
                uint32_t k[4];
                k[0] = ~loadGlobal32<uint32_t>(long_state + j);
                k[1] = shuffle<4>(sPtr,sub, k[0], sub + 1);
                k[2] = shuffle<4>(sPtr,sub, k[0], sub + 2);
                k[3] = shuffle<4>(sPtr,sub, k[0], sub + 3);

                #pragma unroll 4
                for (int i = 0; i < 4; ++i) {
                    // only calculate the key if all data are up to date
                    if (i == sub) {
                        d[x] = a ^
                            t_fn0(k[0] & 0xff) ^
                            t_fn1((k[1] >> 8) & 0xff) ^
                            t_fn2((k[2] >> 16) & 0xff) ^
                            t_fn3((k[3] >> 24));
                    }
                    // the last shuffle is not needed
                    if (i != 3) {
                        /* avoid negative number for modulo
                         * load valid key (k) depending on the round
                         */
                        k[(4 - sub + i) % 4] = shuffle<4>(sPtr,sub, k[0] ^ d[x], i);
                    }
                }
            } else {
                const uint32_t x_0 = loadGlobal32<uint32_t>(long_state + j);
                const uint32_t x_1 = shuffle<4>(sPtr,sub, x_0, sub + 1);
                const uint32_t x_2 = shuffle<4>(sPtr,sub, x_0, sub + 2);
                const uint32_t x_3 = shuffle<4>(sPtr,sub, x_0, sub + 3);
                d[x] = a ^
                    t_fn0(x_0 & 0xff) ^
                    t_fn1((x_1 >> 8) & 0xff) ^
                    t_fn2((x_2 >> 16) & 0xff) ^
                    t_fn3((x_3 >> 24));
            }

            //XOR_BLOCKS_DST(c, b, &long_state[j]);
            t1[0] = shuffle<4>(sPtr,sub, d[x], 0);

            const uint32_t z = d[0] ^ d[1];
            if (BASE == xmrig::VARIANT_1) {
                const uint32_t table = 0x75310U;
                const uint32_t index = ((z >> (VARIANT == xmrig::VARIANT_XTL ? 27 : 26)) & 12) | ((z >> 23) & 2);
                const uint32_t fork_7 = z ^ ((table >> index) & 0x30U) << 24;
                storeGlobal32(long_state + j, sub == 2 ? fork_7 : z);
            }
            else {
                storeGlobal32(long_state + j, z);
            }

            //MUL_SUM_XOR_DST(c, a, &long_state[((uint32_t *)c)[0] & MASK]);
            j = ( ( *t1 & MASK ) >> 2 ) + sub;

            uint32_t yy[2];
            *( (uint64_t*) yy ) = loadGlobal64<uint64_t>( ( (uint64_t *) long_state )+( j >> 1 ) );
            uint32_t zz[2];
            zz[0] = shuffle<4>(sPtr,sub, yy[0], 0);
            zz[1] = shuffle<4>(sPtr,sub, yy[1], 0);

            t1[1] = shuffle<4>(sPtr,sub, d[x], 1);
            #pragma unroll
            for (k = 0; k < 2; k++) {
                t2[k] = shuffle<4>(sPtr,sub, a, k + sub2);
            }

            *( (uint64_t *) t2 ) += sub2 ? ( *( (uint64_t *) t1 ) * *( (uint64_t*) zz ) ) : __umul64hi( *( (uint64_t *) t1 ), *( (uint64_t*) zz ) );

            res = *( (uint64_t *) t2 )  >> ( sub & 1 ? 32 : 0 );

            if (BASE == xmrig::VARIANT_1) {
                const uint32_t tweaked_res = tweak1_2[sub & 1] ^ res;
                uint32_t long_state_update = sub2 ? tweaked_res : res;

                if (VARIANT == xmrig::VARIANT_TUBE || VARIANT == xmrig::VARIANT_RTO) {
                    uint32_t value = shuffle<4>(sPtr,sub, long_state_update, sub & 1) ^ long_state_update;
                    long_state_update = sub >= 2 ? value : long_state_update;
                }

                storeGlobal32(long_state + j, long_state_update);
            }
            else {
                storeGlobal32(long_state + j, res);
            }

            a = ( sub & 1 ? yy[1] : yy[0] ) ^ res;
            idx0 = shuffle<4>(sPtr,sub, a, 0);
            if (ALGO == xmrig::CRYPTONIGHT_HEAVY) {
                int64_t n = loadGlobal64<uint64_t>( ( (uint64_t *) long_state ) + (( idx0 & MASK ) >> 3));
                int32_t d = loadGlobal32<uint32_t>( (uint32_t*)(( (uint64_t *) long_state ) + (( idx0 & MASK) >> 3) + 1u ));
                int64_t q = fast_div_heavy(n, d | 0x5);

                if (sub & 1) {
                    storeGlobal64<uint64_t>( ( (uint64_t *) long_state ) + (( idx0 & MASK ) >> 3), n ^ q );
                }

                if (VARIANT == xmrig::VARIANT_XHV) {
                    d = ~d;
                }

                idx0 = d ^ q;
            }
        }
    }

    if (bfactor > 0) {
        (d_ctx_a + thread * 4)[sub] = a;
        (d_ctx_b + thread * 4)[sub] = d[1];
        if (ALGO == xmrig::CRYPTONIGHT_HEAVY) {
            if (sub&1) {
                *(d_ctx_b + threads * 4 + thread) = idx0;
            }
        }
    }
}

template<size_t ITERATIONS, uint32_t MEM, xmrig::Algo ALGO>
__global__ void cryptonight_core_gpu_phase3( int threads, int bfactor, int partidx, const uint32_t * __restrict__ long_state, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2 )
{
    __shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init( sharedMemory );
    __syncthreads( );

    int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 3;
    int subv = ( threadIdx.x & 7 );
    int sub = subv << 2;

    const int batchsize = MEM >> bfactor;
    const int start = (partidx % (1 << bfactor)) * batchsize;
    const int end = start + batchsize;

    if ( thread >= threads )
        return;

    uint32_t key[40], text[4];
    MEMCPY8( key, d_ctx_key2 + thread * 40, 20 );
    MEMCPY8( text, d_ctx_state + thread * 50 + sub + 16, 2 );

    __syncthreads( );

#   if ( __CUDA_ARCH__ < 300 )
    extern __shared__ uint32_t shuffleMem[];
    volatile uint32_t* sPtr = (volatile uint32_t*)(shuffleMem + (threadIdx.x& 0xFFFFFFF8));
#   else
    volatile uint32_t* sPtr = NULL;
#   endif

    for (int i = start; i < end; i += 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            text[j] ^= long_state[((IndexType) thread * MEM) + ( sub + i + j)];
        }

        cn_aes_pseudo_round_mut( sharedMemory, text, key );

        if (ALGO == xmrig::CRYPTONIGHT_HEAVY) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                text[j] ^= shuffle<8>(sPtr, subv, text[j], (subv+1) & 7);
            }
        }
    }

    MEMCPY8( d_ctx_state + thread * 50 + sub + 16, text, 2 );
}

template<xmrig::Algo ALGO, xmrig::Variant VARIANT>
void cryptonight_core_gpu_hash(nvid_ctx* ctx, uint32_t nonce)
{
    constexpr size_t MASK         = xmrig::cn_select_mask<ALGO>();
    constexpr size_t ITERATIONS   = xmrig::cn_select_iter<ALGO, VARIANT>();
    constexpr size_t MEM          = xmrig::cn_select_memory<ALGO>() / 4;
    constexpr xmrig::Variant BASE = xmrig::cn_base_variant<VARIANT>();

    dim3 grid(ctx->device_blocks);
    dim3 block(ctx->device_threads);
    dim3 block2(ctx->device_threads << 1);
    dim3 block4(ctx->device_threads << 2);
    dim3 block8(ctx->device_threads << 3);

    int partcount = 1 << ctx->device_bfactor;

    /* bfactor for phase 1 and 3
     *
     * phase 1 and 3 consume less time than phase 2, therefore we begin with the
     * kernel splitting if the user defined a `bfactor >= 5`
     */
    int bfactorOneThree = ctx->device_bfactor - 4;
    if (bfactorOneThree < 0) {
        bfactorOneThree = 0;
    }

    const int partcountOneThree = 1 << bfactorOneThree;
    for (int i = 0; i < partcountOneThree; i++) {
        CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase1<ITERATIONS, MEM><<< grid, block8 >>>( ctx->device_blocks*ctx->device_threads,
            bfactorOneThree, i,
            ctx->d_long_state,
            (ALGO == xmrig::CRYPTONIGHT_HEAVY ? ctx->d_ctx_state2 : ctx->d_ctx_state),
            ctx->d_ctx_key1));

        if (partcount > 1 && ctx->device_bsleep > 0) {
            compat_usleep(ctx->device_bsleep);
        }
    }

    if (partcount > 1 && ctx->device_bsleep > 0) {
        compat_usleep(ctx->device_bsleep);
    }

    for (int i = 0; i < partcount; i++) {
        if ((VARIANT == xmrig::VARIANT_WOW) || (VARIANT == xmrig::VARIANT_4)) {
            int threads = ctx->device_blocks * ctx->device_threads;
            void* args[] = { &threads, &ctx->device_bfactor, &i, &ctx->d_long_state, &ctx->d_ctx_a, &ctx->d_ctx_b, &ctx->d_ctx_state, &nonce, &ctx->d_input };
            CU_CHECK(ctx->device_id, cuLaunchKernel(
                ctx->kernel,
                grid.x, grid.y, grid.z,
                block2.x, block2.y, block2.z,
                sizeof(uint64_t) * block.x * 8 + block.x * sizeof(uint32_t) * static_cast<int>(ctx->device_arch[0] < 3), nullptr,
                args, 0
            ));
            CU_CHECK(ctx->device_id, cuCtxSynchronize());
        } else if (BASE == xmrig::VARIANT_2) {
            CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase2_double<ITERATIONS, MEM, MASK, ALGO, VARIANT><<<
                grid,
                block2,
                sizeof(uint64_t) * block.x * 8 + block.x * sizeof(uint32_t) * static_cast<int>(ctx->device_arch[0] < 3)
            >>>(
                ctx->device_blocks * ctx->device_threads,
                ctx->device_bfactor,
                i,
                ctx->d_long_state,
                ctx->d_ctx_a,
                ctx->d_ctx_b,
                ctx->d_ctx_state,
                nonce,
                ctx->d_input
                )
            );
        } else {
            CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase2_quad<ITERATIONS, MEM, MASK, ALGO, VARIANT, BASE><<<
                grid,
                block4,
                block4.x * sizeof(uint32_t) * static_cast<int>(ctx->device_arch[0] < 3)
            >>>(
                ctx->device_blocks * ctx->device_threads,
                ctx->device_bfactor,
                i,
                ctx->d_long_state,
                ctx->d_ctx_a,
                ctx->d_ctx_b,
                ctx->d_ctx_state,
                nonce,
                ctx->d_input
                )
            );
        }

        if (partcount > 1 && ctx->device_bsleep > 0) {
            compat_usleep(ctx->device_bsleep);
        }
    }

    const int roundsPhase3 = ALGO == xmrig::CRYPTONIGHT_HEAVY ? partcountOneThree * 2 : partcountOneThree;
    for (int i = 0; i < roundsPhase3; i++) {
        CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase3<ITERATIONS, MEM, ALGO><<<
            grid,
            block8,
            block8.x * sizeof(uint32_t) * static_cast< int >( ctx->device_arch[0] < 3 )
        >>>( ctx->device_blocks*ctx->device_threads,
            bfactorOneThree, i,
            ctx->d_long_state,
            ctx->d_ctx_state, ctx->d_ctx_key2));
    }
}


template<xmrig::Algo ALGO, xmrig::Variant VARIANT>
void cryptonight_core_gpu_hash_gpu(nvid_ctx* ctx, uint32_t nonce)
{
	constexpr size_t ITERATIONS   = xmrig::cn_select_iter<ALGO, VARIANT>();
	constexpr size_t MEMORY       = xmrig::cn_select_memory<ALGO>();

	dim3 grid( ctx->device_blocks );
	dim3 block( ctx->device_threads );
	dim3 block2( ctx->device_threads << 1 );
	dim3 block4( ctx->device_threads << 2 );
	dim3 block8( ctx->device_threads << 3 );

	const uint32_t intensity = ctx->device_blocks * ctx->device_threads;

	CUDA_CHECK_KERNEL(
		ctx->device_id,
		xmrig::cn_gpu::cn_explode_gpu<MEMORY><<<intensity,32>>>((int*)ctx->d_ctx_state, (int*)ctx->d_long_state)
	);

	int partcount = 1 << ctx->device_bfactor;
	for(int i = 0; i < partcount; i++)
	{
		CUDA_CHECK_KERNEL(
			ctx->device_id,
			// 36 x 16byte x numThreads
			xmrig::cn_gpu::cryptonight_core_gpu_phase2_gpu<ITERATIONS, MEMORY>
				<<<ctx->device_blocks, ctx->device_threads * 16,  36 * 16 * ctx->device_threads>>>
				(
					(int*)ctx->d_ctx_state,
					(int*)ctx->d_long_state,
					ctx->device_bfactor,
					i,
					ctx->d_ctx_a,
					ctx->d_ctx_b
				)
		);
	}

	/* bfactor for phase 3
	 *
	 * 3 consume less time than phase 2, therefore we begin with the
	 * kernel splitting if the user defined a `bfactor >= 5`
	 */
	int bfactorOneThree = ctx->device_bfactor - 4;
	if( bfactorOneThree < 0 )
		bfactorOneThree = 0;

	int partcountOneThree = 1 << bfactorOneThree;
	int roundsPhase3 = partcountOneThree * 2;

	for ( int i = 0; i < roundsPhase3; i++ )
	{
		CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase3<ITERATIONS,MEMORY/4, xmrig::CRYPTONIGHT_HEAVY><<<
			grid,
			block8,
			block8.x * sizeof(uint32_t) * static_cast< int >( ctx->device_arch[0] < 3 )
		>>>( ctx->device_blocks*ctx->device_threads,
			bfactorOneThree, i,
			ctx->d_long_state,
			ctx->d_ctx_state, ctx->d_ctx_key2 ));
	}
}

void cryptonight_gpu_hash(nvid_ctx *ctx, xmrig::Algo algo, xmrig::Variant variant, uint64_t height, uint32_t startNonce)
{
    using namespace xmrig;

    if (algo == CRYPTONIGHT) {
        if ((variant == VARIANT_WOW) || (variant == VARIANT_4)) {
            if ((ctx->kernel_variant != variant) || (ctx->kernel_height != height)) {
#               ifdef APP_DEBUG
                const int64_t timeStart = xmrig::steadyTimestamp();
#               endif

                if (ctx->module) {
                    cuModuleUnload(ctx->module);
                }

                std::vector<char> ptx;
                std::string lowered_name;
                CryptonightR_get_program(ptx, lowered_name, variant, height, ctx->device_arch[0], ctx->device_arch[1]);

                CU_CHECK(ctx->device_id, cuModuleLoadDataEx(&ctx->module, ptx.data(), 0, 0, 0));
                CU_CHECK(ctx->device_id, cuModuleGetFunction(&ctx->kernel, ctx->module, lowered_name.c_str()));

                ctx->kernel_variant = variant;
                ctx->kernel_height = height;

                CryptonightR_get_program(ptx, lowered_name, variant, height + 1, ctx->device_arch[0], ctx->device_arch[1], true);

#               ifdef APP_DEBUG
                const int64_t timeFinish = xmrig::steadyTimestamp();
                LOG_INFO("GPU #%d updated CryptonightR in %.3fs", ctx->device_id, (timeFinish - timeStart) / 1000.0);
#               endif
            }
        }

        switch (variant) {
        case VARIANT_0:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_0>(ctx, startNonce);
            break;

        case VARIANT_1:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_1>(ctx, startNonce);
            break;

        case VARIANT_XTL:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_XTL>(ctx, startNonce);
            break;

        case VARIANT_MSR:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_MSR>(ctx, startNonce);
            break;

        case VARIANT_XAO:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_XAO>(ctx, startNonce);
            break;

        case VARIANT_RTO:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_RTO>(ctx, startNonce);
            break;

        case VARIANT_2:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_2>(ctx, startNonce);
            break;

        case VARIANT_HALF:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_HALF>(ctx, startNonce);
            break;

        case VARIANT_GPU:
            cryptonight_core_gpu_hash_gpu<CRYPTONIGHT, VARIANT_GPU>(ctx, startNonce);
            break;

        case VARIANT_WOW:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_WOW>(ctx, startNonce);
            break;

        case VARIANT_4:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_4>(ctx, startNonce);
            break;

        case VARIANT_RWZ:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_RWZ>(ctx, startNonce);
            break;

        case VARIANT_ZLS:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_ZLS>(ctx, startNonce);
            break;

        case VARIANT_DOUBLE:
            cryptonight_core_gpu_hash<CRYPTONIGHT, VARIANT_DOUBLE>(ctx, startNonce);
            break;

        default:
            break;
        }
    }
    else if (algo == CRYPTONIGHT_LITE) {
        switch (variant) {
        case VARIANT_0:
            cryptonight_core_gpu_hash<CRYPTONIGHT_LITE, VARIANT_0>(ctx, startNonce);
            break;

        case VARIANT_1:
            cryptonight_core_gpu_hash<CRYPTONIGHT_LITE, VARIANT_1>(ctx, startNonce);
            break;

        default:
            break;
        }
    }
    else if (algo == CRYPTONIGHT_HEAVY) {
        switch (variant) {
        case VARIANT_0:
            cryptonight_core_gpu_hash<CRYPTONIGHT_HEAVY, VARIANT_0>(ctx, startNonce);
            break;

        case VARIANT_TUBE:
            cryptonight_core_gpu_hash<CRYPTONIGHT_HEAVY, VARIANT_TUBE>(ctx, startNonce);
            break;

        case VARIANT_XHV:
            cryptonight_core_gpu_hash<CRYPTONIGHT_HEAVY, VARIANT_XHV>(ctx, startNonce);
            break;

        default:
            break;
        }
    }
    else if (algo == CRYPTONIGHT_PICO) {
        cryptonight_core_gpu_hash<CRYPTONIGHT_PICO, VARIANT_TRTL>(ctx, startNonce);
    }
}
