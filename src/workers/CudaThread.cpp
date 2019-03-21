/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2019      Spudz76     <https://github.com/Spudz76>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "rapidjson/document.h"
#include "workers/CudaThread.h"


CudaThread::CudaThread() :
    m_bfactor(0),
    m_blocks(0),
    m_bsleep(0),
    m_clockRate(0),
    m_memoryClockRate(0),
    m_nvmlId(-1),
    m_smx(0),
    m_threads(0),
    m_affinity(-1),
    m_index(0),
    m_memoryFree(0),
    m_memoryTotal(0),
    m_threadId(0),
    m_pciBusID(0),
    m_pciDeviceID(0),
    m_pciDomainID(0),
    m_syncMode(3),
    m_algorithm(xmrig::INVALID_ALGO)
{
    memset(m_arch, 0, sizeof(m_arch));
    memset(m_name, 0, sizeof(m_name));
}


CudaThread::CudaThread(const nvid_ctx &ctx, int64_t affinity, xmrig::Algo algorithm) :
    m_bfactor(ctx.device_bfactor),
    m_blocks(ctx.device_blocks),
    m_bsleep(ctx.device_bsleep),
    m_clockRate(ctx.device_clockRate),
    m_memoryClockRate(ctx.device_memoryClockRate),
    m_nvmlId(-1),
    m_smx(ctx.device_mpcount),
    m_threads(ctx.device_threads),
    m_affinity(affinity),
    m_index(static_cast<size_t>(ctx.device_id)),
    m_memoryFree(ctx.device_memoryFree),
    m_memoryTotal(ctx.device_memoryTotal),
    m_threadId(0),
    m_pciBusID(ctx.device_pciBusID),
    m_pciDeviceID(ctx.device_pciDeviceID),
    m_pciDomainID(ctx.device_pciDomainID),
    m_syncMode(ctx.syncMode),
    m_algorithm(algorithm)
{
    memcpy(m_arch, ctx.device_arch, sizeof(m_arch));
    strncpy(m_name, ctx.device_name, sizeof(m_name) - 1);
}


CudaThread::CudaThread(const rapidjson::Value &object) :
    m_bfactor(0),
    m_blocks(0),
    m_bsleep(0),
    m_clockRate(0),
    m_memoryClockRate(0),
    m_nvmlId(-1),
    m_smx(0),
    m_threads(0),
    m_affinity(-1),
    m_index(0),
    m_threadId(0),
    m_pciBusID(0),
    m_pciDeviceID(0),
    m_pciDomainID(0),
    m_syncMode(3),
    m_algorithm(xmrig::INVALID_ALGO)
{
    memset(m_arch, 0, sizeof(m_arch));
    memset(m_name, 0, sizeof(m_name));

    setIndex(object["index"].GetUint());
    setThreads(object["threads"].GetInt());
    setBlocks(object["blocks"].GetInt());
    setBFactor(object["bfactor"].GetInt());
    setBSleep(object["bsleep"].GetInt());

    const rapidjson::Value &syncMode = object["sync_mode"];
    if (syncMode.IsUint()) {
        setSyncMode(syncMode.GetUint());
    }

    const rapidjson::Value &affinity = object["affine_to_cpu"];
    if (affinity.IsInt()) {
        setAffinity(affinity.GetInt());
    }
}


bool CudaThread::init(xmrig::Algo algorithm)
{
    if (m_blocks < -1 || m_threads < -1 || m_bfactor < 0 || m_bsleep < 0) {
        return false;
    }

    if (cuda_get_devicecount() == 0) {
        return false;
    }

    nvid_ctx ctx;
    ctx.device_id      = static_cast<int>(m_index);
    ctx.device_blocks  = m_blocks;
    ctx.device_threads = m_threads;
    ctx.device_bfactor = m_bfactor;
    ctx.device_bsleep  = m_bsleep;
    ctx.syncMode       = m_syncMode;

    if (cuda_get_deviceinfo(&ctx, algorithm, false) != 0) {
        return false;
    }

    memcpy(m_arch, ctx.device_arch, sizeof(m_arch));
    strncpy(m_name, ctx.device_name, sizeof(m_name) - 1);

    m_threads = ctx.device_threads;
    m_blocks  = ctx.device_blocks;
    m_smx     = ctx.device_mpcount;

    m_clockRate       = ctx.device_clockRate;
    m_memoryClockRate = ctx.device_memoryClockRate;
    m_memoryTotal     = ctx.device_memoryTotal;
    m_memoryFree      = ctx.device_memoryFree;
    m_pciBusID        = ctx.device_pciBusID;
    m_pciDeviceID     = ctx.device_pciDeviceID;
    m_pciDomainID     = ctx.device_pciDomainID;

    m_algorithm = algorithm;
    return true;
}


void CudaThread::limit(int maxUsage, int maxThreads)
{
    if (maxThreads > 0) {
        if (m_threads > maxThreads) {
            m_threads = maxThreads;
        }

        return;
    }

    if (maxUsage < 100) {
        m_threads = static_cast<int>(m_threads / 100.0 * maxUsage);
    }
}


#ifdef APP_DEBUG
void CudaThread::print() const
{
}
#endif


#ifndef XMRIG_NO_API
rapidjson::Value CudaThread::toAPI(rapidjson::Document &doc) const
{
    return toConfig(doc);
}
#endif


rapidjson::Value CudaThread::toConfig(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    Value obj(kObjectType);
    auto &allocator = doc.GetAllocator();

    obj.AddMember("index",     static_cast<uint64_t>(index()), allocator);
    obj.AddMember("threads",   m_threads, allocator);
    obj.AddMember("blocks",    m_blocks, allocator);
    obj.AddMember("bfactor",   m_bfactor, allocator);
    obj.AddMember("bsleep",    m_bsleep, allocator);
    obj.AddMember("sync_mode", m_syncMode, allocator);

    if (affinity() >= 0) {
        obj.AddMember("affine_to_cpu", affinity(), allocator);
    }
    else {
        obj.AddMember("affine_to_cpu", false, allocator);
    }

    return obj;
}
