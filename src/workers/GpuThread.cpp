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

#include <stdio.h>
#include <string.h>

#include "workers/GpuThread.h"


GpuThread::GpuThread() :
    m_affinity(-1),
    m_bfactor(0),
    m_blocks(0),
    m_bsleep(0),
    m_clockRate(0),
    m_index(0),
    m_memoryClockRate(0),
    m_nvmlId(-1),
    m_pciBusID(0),
    m_pciDeviceID(0),
    m_pciDomainID(0),
    m_smx(0),
    m_threadId(0),
    m_threads(0)
{
    memset(m_arch, 0, sizeof(m_arch));
    memset(m_name, 0, sizeof(m_name));
}


GpuThread::GpuThread(const nvid_ctx &ctx, int affinity) :
    m_affinity(affinity),
    m_bfactor(ctx.device_bfactor),
    m_blocks(ctx.device_blocks),
    m_bsleep(ctx.device_bsleep),
    m_clockRate(ctx.device_clockRate),
    m_index(ctx.device_id),
    m_memoryClockRate(ctx.device_memoryClockRate),
    m_nvmlId(-1),
    m_pciBusID(ctx.device_pciBusID),
    m_pciDeviceID(ctx.device_pciDeviceID),
    m_pciDomainID(ctx.device_pciDomainID),
    m_smx(ctx.device_mpcount),
    m_threadId(0),
    m_threads(ctx.device_threads)
{
    memcpy(m_arch, ctx.device_arch, sizeof(m_arch));
    strncpy(m_name, ctx.device_name, sizeof(m_name) - 1);
}


GpuThread::~GpuThread()
{
}


bool GpuThread::init()
{
    if (m_index < 0 || m_blocks < -1 || m_threads < -1 || m_bfactor < 0 || m_bsleep < 0) {
        return false;
    }

    if (cuda_get_devicecount() == 0) {
        return false;
    }

    nvid_ctx ctx;
    ctx.device_id      = m_index;
    ctx.device_blocks  = m_blocks;
    ctx.device_threads = m_threads;
    ctx.device_bfactor = m_bfactor;
    ctx.device_bsleep  = m_bsleep;

    if (cuda_get_deviceinfo(&ctx) != 1) {
        return false;
    }

    memcpy(m_arch, ctx.device_arch, sizeof(m_arch));
    strncpy(m_name, ctx.device_name, sizeof(m_name) - 1);

    m_threads = ctx.device_threads;
    m_blocks  = ctx.device_blocks;
    m_smx     = ctx.device_mpcount;

    m_clockRate       = ctx.device_clockRate;
    m_memoryClockRate = ctx.device_memoryClockRate;
    m_pciBusID        = ctx.device_pciBusID;
    m_pciDeviceID     = ctx.device_pciDeviceID;
    m_pciDomainID     = ctx.device_pciDomainID;

    return true;
}


void GpuThread::limit(int maxUsage, int maxThreads)
{
    if (maxThreads > 0) {
        if (m_threads > maxThreads) {
            m_threads = maxThreads;
        }

        return;
    }

    if (maxUsage < 100) {
        m_threads = (int) m_threads / 100.0 * maxUsage;
    }
}
