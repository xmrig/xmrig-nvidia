/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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


#include "crypto/CryptoNight.h"
#include "crypto/CryptoNight_p.h"
#include "crypto/CryptoNight_test.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "Options.h"


bool (*cryptonight_hash_ctx)(const void *input, size_t size, void *output, cryptonight_ctx *ctx, int variant) = nullptr;


static bool cryptonight_av1_aesni(const void *input, size_t size, void *output, struct cryptonight_ctx *ctx, int variant) {
    return cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, false, true>(input, size, output, ctx, variant);
}


static bool cryptonight_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx, int variant) {
    return cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, true, true>(input, size, output, ctx, variant);
}


#ifndef XMRIG_NO_AEON
static bool cryptonight_lite_av1_aesni(const void *input, size_t size, void *output, cryptonight_ctx *ctx, int variant) {
    return cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, false, false>(input, size, output, ctx, variant);
}


static bool cryptonight_lite_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx, int variant) {
    return cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, true, false>(input, size, output, ctx, variant);
}


bool (*cryptonight_variations[8])(const void *input, size_t size, void *output, cryptonight_ctx *ctx, int variant) = {
            cryptonight_av1_aesni,
            nullptr,
            cryptonight_av3_softaes,
            nullptr,
            cryptonight_lite_av1_aesni,
            nullptr,
            cryptonight_lite_av3_softaes,
            nullptr
        };
#else
bool (*cryptonight_variations[4])(const void *input, size_t size, void *output, cryptonight_ctx *ctx, int variant) = {
            cryptonight_av1_aesni,
            nullptr,
            cryptonight_av3_softaes,
            nullptr
        };
#endif


bool CryptoNight::hash(const Job &job, JobResult &result, cryptonight_ctx *ctx)
{
    bool success = false;
    if (1 < job.size())
    {
        const int variant = ((const uint8_t*)job.blob())[0] >= 7 ? ((const uint8_t*)job.blob())[0] - 6 : 0;
        success = cryptonight_hash_ctx(job.blob(), job.size(), result.result, ctx, variant);
    }
    return success && *reinterpret_cast<uint64_t*>(result.result + 24) < job.target();
}


bool CryptoNight::init(int algo, int variant)
{
    if (variant < 1 || variant > 4) {
        return false;
    }

#   ifndef XMRIG_NO_AEON
    const int index = algo == Options::ALGO_CRYPTONIGHT_LITE ? (variant + 3) : (variant - 1);
#   else
    const int index = variant - 1;
#   endif

    cryptonight_hash_ctx = cryptonight_variations[index];

    return selfTest(algo);
}


bool CryptoNight::hash(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx *ctx, int variant)
{
    return cryptonight_hash_ctx(input, size, output, ctx, variant);
}


bool CryptoNight::selfTest(int algo) {
    if (cryptonight_hash_ctx == nullptr) {
        return false;
    }

    char output[32];

    cryptonight_ctx *ctx = static_cast<cryptonight_ctx*>(_mm_malloc(sizeof(struct cryptonight_ctx), 16));

    const bool success = cryptonight_hash_ctx(test_input, 76, output, ctx, 0);

    _mm_free(ctx);

#   ifdef XMRIG_NO_AEON
    return successs && memcmp(output, test_output0, 32) == 0;
#   else
    return success && memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output1 : test_output0, 32) == 0;
#   endif
}
