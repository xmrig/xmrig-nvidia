/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include <stdlib.h>
#include <string.h>
#include <stdio.h>


#include "nvidia/CudaCLI.h"
#include "nvidia/cryptonight.h"
#include "workers/CudaThread.h"


CudaCLI::CudaCLI() :
    m_count(cuda_get_devicecount())
{
}


bool CudaCLI::setup(std::vector<xmrig::IThread *> &threads, xmrig::Algo algo, bool isCNv2)
{
    if (isEmpty() || m_count == 0) {
        return false;
    }

    if (m_devices.empty()) {
        for (int i = 0; i < m_count; i++) {
            m_devices.push_back(i);
        }
    }

    for (int i = 0; i < static_cast<int>(m_devices.size()); i++) {
        nvid_ctx ctx;
        ctx.device_id      = m_devices[static_cast<size_t>(i)];
        ctx.device_blocks  = blocks(i);
        ctx.device_threads = this->threads(i);
        ctx.device_bfactor = bfactor(i);
        ctx.device_bsleep  = bsleep(i);
        ctx.syncMode       = 3;

        if (cuda_get_deviceinfo(&ctx, algo, isCNv2) != 0) {
            continue;
        }

        threads.push_back(new CudaThread(ctx, affinity(i), algo));
    }

    return true;
}


void CudaCLI::autoConf(std::vector<xmrig::IThread *> &threads, xmrig::Algo algo, bool isCNv2)
{
    if (m_count == 0) {
        return;
    }

    for (int i = 0; i < m_count; i++) {
        nvid_ctx ctx;
        ctx.device_id      = i;
        ctx.device_blocks  = -1;
        ctx.device_threads = -1;
        ctx.device_bfactor = bfactor();
        ctx.device_bsleep  = bsleep();
        ctx.syncMode       = 3;

        if (cuda_get_deviceinfo(&ctx, algo, isCNv2) != 0) {
            continue;
        }

        threads.push_back(new CudaThread(ctx, -1, algo));
    }
}


void CudaCLI::parseDevices(const char *arg)
{
    char *value = strdup(arg);
    char *pch   = strtok(value, ",");

    while (pch != nullptr) {
        const int index = static_cast<int>(strtoul(pch, nullptr, 10));
        if (index < m_count) {
            m_devices.push_back(index);
        }
        else {
            fprintf(stderr, "Non-existent CUDA device #%d specified in --cuda-devices option\n", index);
        }

        pch = strtok(nullptr, ",");
    }

    free(value);
}


void CudaCLI::parseLaunch(const char *arg)
{
    char *value = strdup(arg);
    char *pch   = strtok(value, ",");
    std::vector<char *> tmp;

    while (pch != nullptr) {
        tmp.push_back(pch);
        pch = strtok(nullptr, ",");
    }

    for (char *config : tmp) {
        pch       = strtok(config, "x");
        int count = 0;

        while (pch != nullptr && count < 2) {
            count++;

            const int v = static_cast<int>(strtoul(pch, nullptr, 10));
            if (count == 1) {
                m_threads.push_back(v > 0 ? v : -1);
            }
            else if (count == 2) {
                m_blocks.push_back(v > 0 ? v : -1);
            }

            pch = strtok(nullptr, "x");
        }

        if (count == 1) {
            m_blocks.push_back(-1);
        }
    }

    free(value);
}


int CudaCLI::get(const std::vector<int> &vector, int index, int defaultValue) const
{
    if (vector.empty()) {
        return defaultValue;
    }

    if (vector.size() <= index) {
        return vector.back();
    }

    return vector[index];
}


void CudaCLI::parse(std::vector<int> &vector, const char *arg) const
{
    char *value = strdup(arg);
    char *pch   = strtok(value, ",");

    while (pch != nullptr) {
        vector.push_back(static_cast<int>(strtoul(pch, nullptr, 10)));

        pch = strtok(nullptr, ",");
    }

    free(value);
}
