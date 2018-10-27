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


#include <thread>


#include "common/log/Log.h"
#include "common/Platform.h"
#include "crypto/CryptoNight.h"
#include "workers/CudaThread.h"
#include "workers/CudaWorker.h"
#include "workers/Handle.h"
#include "workers/Workers.h"


CudaWorker::CudaWorker(Handle *handle) :
    m_id(handle->threadId()),
    m_threads(handle->totalWays()),
    m_algorithm(handle->config()->algorithm()),
    m_hashCount(0),
    m_timestamp(0),
    m_count(0),
    m_sequence(0),
    m_blob()
{
    const CudaThread *thread = static_cast<CudaThread *>(handle->config());

    m_ctx.device_id      = static_cast<int>(thread->index());
    m_ctx.device_blocks  = thread->blocks();
    m_ctx.device_threads = thread->threads();
    m_ctx.device_bfactor = thread->bfactor();
    m_ctx.device_bsleep  = thread->bsleep();
    m_ctx.syncMode       = thread->syncMode();

    if (thread->affinity() >= 0) {
        Platform::setThreadAffinity(static_cast<uint64_t>(thread->affinity()));
    }
}


void CudaWorker::start()
{
    if (cuda_get_deviceinfo(&m_ctx, m_algorithm, false) != 0 || cryptonight_gpu_init(&m_ctx, m_algorithm) != 1) {
        LOG_ERR("Setup failed for GPU %zu. Exitting.", m_id);
        return;
    }

    while (Workers::sequence() > 0) {
        if (Workers::isPaused()) {
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            while (Workers::isPaused());

            if (Workers::sequence() == 0) {
                break;
            }

            consumeJob();
        }

        cryptonight_extra_cpu_set_data(&m_ctx, m_blob, m_job.size());

        while (!Workers::isOutdated(m_sequence)) {
            uint32_t foundNonce[10];
            uint32_t foundCount;
  
            cryptonight_extra_cpu_prepare(&m_ctx, m_nonce, m_algorithm, m_job.algorithm().variant());
            cryptonight_gpu_hash(&m_ctx, m_algorithm, m_job.algorithm().variant(), m_nonce);
            cryptonight_extra_cpu_final(&m_ctx, m_nonce, m_job.target(), &foundCount, foundNonce, m_algorithm);

            for (size_t i = 0; i < foundCount; i++) {
                *m_job.nonce() = foundNonce[i];
                Workers::submit(m_job);
            }

            m_count += m_ctx.device_blocks * m_ctx.device_threads;
            m_nonce += m_ctx.device_blocks * m_ctx.device_threads;

            storeStats();
            std::this_thread::yield();
        }

        consumeJob();
    }

    cryptonight_extra_cpu_free(&m_ctx, m_algorithm);
}


bool CudaWorker::resume(const Job &job)
{
    if (m_job.poolId() == -1 && job.poolId() >= 0 && job.id() == m_pausedJob.id()) {
        m_job   = m_pausedJob;
        m_nonce = m_pausedNonce;
        return true;
    }

    return false;
}


void CudaWorker::consumeJob()
{
    Job job = Workers::job();
    m_sequence = Workers::sequence();
    if (m_job == job) {
        return;
    }

    save(job);

    if (resume(job)) {
        setJob();
        return;
    }

    m_job = std::move(job);
    m_job.setThreadId(m_id);

    if (m_job.isNicehash()) {
        m_nonce = (*m_job.nonce() & 0xff000000U) + (0xffffffU / m_threads * m_id);
    }
    else {
        m_nonce = 0xffffffffU / m_threads * m_id;
    }

    setJob();
}


void CudaWorker::save(const Job &job)
{
    if (job.poolId() == -1 && m_job.poolId() >= 0) {
        m_pausedJob   = m_job;
        m_pausedNonce = m_nonce;
    }
}


void CudaWorker::setJob()
{
    memcpy(m_blob, m_job.blob(), sizeof(m_blob));
}


void CudaWorker::storeStats()
{
    using namespace std::chrono;

    const uint64_t timestamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
    m_hashCount.store(m_count, std::memory_order_relaxed);
    m_timestamp.store(timestamp, std::memory_order_relaxed);
}
