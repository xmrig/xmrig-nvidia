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


#include <assert.h>


#include "Cpu.h"
#include "crypto/CryptoNight.h"
#include "crypto/CryptoNight_test.h"
#include "crypto/CryptoNight_x86.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "Options.h"


xmrig::Algo CryptoNight::m_algorithm = xmrig::CRYPTONIGHT;
xmrig::AlgoVerify CryptoNight::m_av  = xmrig::VERIFY_HW_AES;


bool CryptoNight::hash(const Job &job, JobResult &result, cryptonight_ctx *ctx)
{
    fn(job.variant())(job.blob(), job.size(), result.result, ctx);

    return *reinterpret_cast<uint64_t*>(result.result + 24) < job.target();
}


bool CryptoNight::init(xmrig::Algo algorithm)
{
    m_algorithm = algorithm;
    m_av        = Cpu::hasAES() ? xmrig::VERIFY_HW_AES : xmrig::VERIFY_SOFT_AES;

    return selfTest();
}


CryptoNight::cn_hash_fun CryptoNight::fn(xmrig::Algo algorithm, xmrig::AlgoVerify av, xmrig::Variant variant)
{
    using namespace xmrig;

    assert(variant == VARIANT_NONE || variant == VARIANT_V1);

    static const cn_hash_fun func_table[10] = {
        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_NONE>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_NONE>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_V1>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_V1>,

#       ifndef XMRIG_NO_AEON
        cryptonight_single_hash<CRYPTONIGHT_LITE, false, VARIANT_NONE>,
        cryptonight_single_hash<CRYPTONIGHT_LITE, true,  VARIANT_NONE>,

        cryptonight_single_hash<CRYPTONIGHT_LITE, false, VARIANT_V1>,
        cryptonight_single_hash<CRYPTONIGHT_LITE, true,  VARIANT_V1>,
#       else
        nullptr, nullptr, nullptr, nullptr,
#       endif

#       ifndef XMRIG_NO_SUMO
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, false, VARIANT_NONE>,
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_NONE>,
#       else
        nullptr, nullptr,
#       endif
    };

#   ifndef XMRIG_NO_SUMO
    if (algorithm == CRYPTONIGHT_HEAVY) {
        variant = VARIANT_NONE;
    }
#   endif

    return func_table[4 * algorithm + 2 * variant + av - 1];
}


bool CryptoNight::selfTest() {
    if (fn(xmrig::VARIANT_NONE) == nullptr || fn(xmrig::VARIANT_V1) == nullptr) {
        return false;
    }

    uint8_t output[32];
    cryptonight_ctx *ctx = static_cast<cryptonight_ctx *>(_mm_malloc(sizeof(cryptonight_ctx), 16));

    fn(xmrig::VARIANT_NONE)(test_input, 76, output, ctx);

    if (m_algorithm == xmrig::CRYPTONIGHT && memcmp(output, test_output_v0, 32) == 0) {
        fn(xmrig::VARIANT_V1)(test_input, 76, output, ctx);

        _mm_free(ctx);

        return memcmp(output, test_output_v1, 32) == 0;
    }

#   ifndef XMRIG_NO_AEON
    if (m_algorithm == xmrig::CRYPTONIGHT_LITE && memcmp(output, test_output_v0_lite, 32) == 0) {
        fn(xmrig::VARIANT_V1)(test_input, 76, output, ctx);

        _mm_free(ctx);

        return memcmp(output, test_output_v1_lite, 32) == 0;
    }
#   endif

    const bool rc = m_algorithm == xmrig::CRYPTONIGHT_HEAVY && memcmp(output, test_output_heavy, 32) == 0;
    _mm_free(ctx);

    return rc;
}
