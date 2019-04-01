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

#ifndef XMRIG_CUDATHREAD_H
#define XMRIG_CUDATHREAD_H

#include <vector>


#include "interfaces/IThread.h"
#include "nvidia/cryptonight.h"


class CudaThread : public xmrig::IThread
{
public:
    CudaThread();
    CudaThread(const nvid_ctx &ctx, int64_t affinity, xmrig::Algo algorithm);
    CudaThread(const rapidjson::Value &object);

    bool init(xmrig::Algo algorithm);
    void limit(int maxUsage, int maxThreads);

    inline const char *name() const       { return m_name; }
    inline const int *arch() const        { return m_arch; }
    inline int bfactor() const            { return m_bfactor; }
    inline int blocks() const             { return m_blocks; }
    inline int bsleep() const             { return m_bsleep; }
    inline int clockRate() const          { return m_clockRate; }
    inline int memoryClockRate() const    { return m_memoryClockRate; }
    inline size_t memoryTotal() const     { return m_memoryTotal; }
    inline size_t memoryFree() const      { return m_memoryFree; }
    inline int nvmlId() const             { return m_nvmlId; }
    inline int smx() const                { return m_smx; }
    inline int threads() const            { return m_threads; }
    inline size_t threadId() const        { return m_threadId; }
    inline uint32_t pciBusID() const      { return m_pciBusID; }
    inline uint32_t pciDeviceID() const   { return m_pciDeviceID; }
    inline uint32_t pciDomainID() const   { return m_pciDomainID; }
    inline uint32_t syncMode() const      { return m_syncMode; }

    inline xmrig::Algo algorithm() const override { return m_algorithm; }
    inline int priority() const override          { return -1; }
    inline int64_t affinity() const override      { return m_affinity; }
    inline Multiway multiway() const override     { return SingleWay; }
    inline size_t index() const override          { return m_index; }
    inline Type type() const override             { return CUDA; }

    inline void setAffinity(int affinity)      { m_affinity = affinity; }
    inline void setBFactor(int bfactor)        { if (bfactor >= 0 && bfactor <= 12) { m_bfactor = bfactor; } }
    inline void setBlocks(int blocks)          { m_blocks = blocks; }
    inline void setBSleep(int bsleep)          { m_bsleep = bsleep; }
    inline void setIndex(size_t index)         { m_index = index; }
    inline void setNvmlId(int id)              { m_nvmlId = id; }
    inline void setThreadId(size_t threadId)   { m_threadId = threadId; }
    inline void setThreads(int threads)        { m_threads = threads; }
    inline void setSyncMode(uint32_t syncMode) { m_syncMode = syncMode > 3 ? 3 : syncMode; }

protected:
#   ifdef APP_DEBUG
    void print() const override;
#   endif

#   ifndef XMRIG_NO_API
    rapidjson::Value toAPI(rapidjson::Document &doc) const override;
#   endif

    rapidjson::Value toConfig(rapidjson::Document &doc) const override;

private:
    char m_name[256];
    int m_arch[2];
    int m_bfactor;
    int m_blocks;
    int m_bsleep;
    int m_clockRate;
    int m_memoryClockRate;
    int m_nvmlId;
    int m_smx;
    int m_threads;
    int64_t m_affinity;
    size_t m_index;
    size_t m_memoryFree;
    size_t m_memoryTotal;
    size_t m_threadId;
    uint32_t m_pciBusID;
    uint32_t m_pciDeviceID;
    uint32_t m_pciDomainID;
    uint32_t m_syncMode;
    xmrig::Algo m_algorithm;
};


#endif /* XMRIG_CUDATHREAD_H */
