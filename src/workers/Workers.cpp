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

#include <cmath>
#include <thread>


#include "api/Api.h"
#include "common/log/Log.h"
#include "common/cpu/Cpu.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "crypto/CryptoNight.h"
#include "crypto/randomx/allocator.hpp"
#include "crypto/randomx/configuration.h"
#include "interfaces/IJobResultListener.h"
#include "interfaces/IThread.h"
#include "nvidia/NvmlApi.h"
#include "rapidjson/document.h"
#include "workers/CudaThread.h"
#include "workers/CudaWorker.h"
#include "workers/Handle.h"
#include "workers/Hashrate.h"
#include "workers/Workers.h"
#include "Mem.h"


bool Workers::m_active = false;
bool Workers::m_enabled = true;
Hashrate *Workers::m_hashrate = nullptr;
size_t Workers::m_threadsCount = 0;
std::atomic<int> Workers::m_paused;
std::atomic<uint64_t> Workers::m_sequence;
std::list<xmrig::Job> Workers::m_queue;
std::vector<Handle*> Workers::m_workers;
uint64_t Workers::m_ticks = 0;
uv_async_t Workers::m_async;
uv_mutex_t Workers::m_mutex;
uv_rwlock_t Workers::m_rwlock;
uv_timer_t Workers::m_reportTimer;
uv_timer_t Workers::m_timer;
xmrig::Controller *Workers::m_controller = nullptr;
xmrig::IJobResultListener *Workers::m_listener = nullptr;
xmrig::Job Workers::m_job;

#ifdef XMRIG_ALGO_RANDOMX
uv_rwlock_t Workers::m_rx_dataset_lock;
randomx_cache *Workers::m_rx_cache = nullptr;
randomx_dataset *Workers::m_rx_dataset = nullptr;
uint8_t *Workers::m_rx_scratchpad = nullptr;
randomx_vm *Workers::m_rx_vm = nullptr;
uint8_t Workers::m_rx_seed_hash[32] = {};
xmrig::Variant Workers::m_rx_variant = xmrig::VARIANT_MAX;
std::atomic<uint32_t> Workers::m_rx_dataset_init_thread_counter = {};
#endif


struct JobBaton
{
    uv_work_t request;
    std::vector<xmrig::Job> jobs;
    std::vector<xmrig::JobResult> results;
    int errors = 0;

    JobBaton() {
        request.data = this;
    }
};


xmrig::Job Workers::job()
{
    uv_rwlock_rdlock(&m_rwlock);
    xmrig::Job job = m_job;
    uv_rwlock_rdunlock(&m_rwlock);

    return job;
}


size_t Workers::hugePages()
{
    return 0;
}


size_t Workers::threads()
{
    return m_threadsCount;
}


void Workers::printHashrate(bool detail)
{
    assert(m_controller != nullptr);
    if (!m_controller) {
        return;
    }

    if (detail) {
        char num1[8] = { 0 };
        char num2[8] = { 0 };
        char num3[8] = { 0 };

        Log::i()->text(WHITE_BOLD_S "| THREAD | GPU | 10s H/s | 60s H/s | 15m H/s | NAME");

        size_t i = 0;
        for (const xmrig::IThread *t : m_controller->config()->threads()) {
            auto thread = static_cast<const CudaThread *>(t);
             Log::i()->text("| %6zu | %3zu | %7s | %7s | %7s | " BLACK_BOLD_S "%s",
                            i, thread->index(),
                            Hashrate::format(m_hashrate->calc(i, Hashrate::ShortInterval), num1, sizeof num1),
                            Hashrate::format(m_hashrate->calc(i, Hashrate::MediumInterval), num2, sizeof num2),
                            Hashrate::format(m_hashrate->calc(i, Hashrate::LargeInterval), num3, sizeof num3),
                            thread->name()
                            );

             i++;
        }
    }

    m_hashrate->print();
}

const std::string _spf(const char * const fmt, ...)
{
    va_list args = nullptr, copy = nullptr;
    va_start(args, fmt);
    va_copy(copy, args);
    const auto len = static_cast<const unsigned __int64>(std::vsnprintf(nullptr, 0, fmt, copy));
    va_end(copy);
    std::vector<char> str(len + 1);
    std::vsnprintf(str.data(), str.size(), fmt, args);
    va_end(args);
    return std::string(str.data(), len);
}

void Workers::printHealth()
{
    if (!NvmlApi::isAvailable()) {
        LOG_ERR("NVML GPU monitoring is not available");
        return;
    }

    Health health;
    for (const xmrig::IThread *t : m_controller->config()->threads()) {
        auto thread = static_cast<const CudaThread *>(t);
        if (!NvmlApi::health(thread->nvmlId(), health)) {
            continue;
        }

        const uint32_t temp = health.temperature;

        if (health.clock && health.memClock) {
            LOG_INFO(" * " MAGENTA("GPU #%d:") " "
                     MAGENTA_BOLD("%u") MAGENTA("/") MAGENTA_BOLD("%u MHz") " "
                     MAGENTA_BOLD("%uW") " "
                     "%s%uC%s "
                     MAGENTA("FAN") " " MAGENTA_BOLD_S "%u%%",
                thread->index(),
                health.clock, health.memClock,
                health.power / 1000,
                (temp < 45 ? GREEN_BOLD_S : (temp > 65 ? RED_BOLD_S : YELLOW_BOLD_S)), temp, CLEAR,
                health.fanSpeed);

            continue;
        }

        LOG_INFO(" * " MAGENTA("GPU #%d:") " "
                 "%s%uC%s "
                 MAGENTA("FAN") " " MAGENTA_BOLD_S "%u%%",
            thread->index(),
            (temp < 45 ? GREEN_BOLD_S : (temp > 65 ? RED_BOLD_S : YELLOW_BOLD_S)), temp, CLEAR,
            health.fanSpeed);
    }
}


void Workers::setEnabled(bool enabled)
{
    if (m_enabled == enabled) {
        return;
    }

    m_enabled = enabled;
    if (!m_active) {
        return;
    }

    m_paused = enabled ? 0 : 1;
    m_sequence++;
}


void Workers::setJob(const xmrig::Job &job, bool donate)
{
    uv_rwlock_wrlock(&m_rwlock);
    m_job = job;

    if (donate) {
        m_job.setPoolId(-1);
    }
    uv_rwlock_wrunlock(&m_rwlock);

    m_active = true;
    if (!m_enabled) {
        return;
    }

    m_sequence++;
    m_paused = 0;
}


bool Workers::start(xmrig::Controller *controller)
{
    m_controller = controller;
    const std::vector<xmrig::IThread *> &threads = controller->config()->threads();
    size_t ways = 0;

    for (const xmrig::IThread *thread : threads) {
       ways += static_cast<size_t>(thread->multiway());
    }

    m_threadsCount = threads.size();
    m_hashrate     = new Hashrate(m_threadsCount, controller);
    m_sequence     = 1;
    m_paused       = 1;

#   ifdef XMRIG_ALGO_RANDOMX
    uv_rwlock_init(&m_rx_dataset_lock);
#   endif

    uv_mutex_init(&m_mutex);
    uv_rwlock_init(&m_rwlock);
    uv_async_init(uv_default_loop(), &m_async, Workers::onResult);
    uv_timer_init(uv_default_loop(), &m_timer);
    uv_timer_start(&m_timer, Workers::onTick, 500, 500);

    uint32_t offset = 0;

    size_t i = 0;
    for (xmrig::IThread *thread : threads) {
        Handle *handle = new Handle(i, thread, offset, ways);
        offset += static_cast<size_t>(thread->multiway());
        i++;

        m_workers.push_back(handle);
        handle->start(Workers::onReady);
    }

    controller->save();

    const uint64_t printTime = static_cast<uint64_t>(m_controller->config()->printTime());
    if (printTime > 0) {
        uv_timer_init(uv_default_loop(), &m_reportTimer);
        uv_timer_start(&m_reportTimer, Workers::onReport, (printTime + 4) * 1000, printTime * 1000);
    }

    return true;
}


void Workers::stop()
{
    if (m_controller->config()->printTime() > 0) {
        uv_timer_stop(&m_reportTimer);
    }

    uv_timer_stop(&m_timer);
    m_hashrate->stop();

    uv_close(reinterpret_cast<uv_handle_t*>(&m_async), nullptr);
    m_paused   = 0;
    m_sequence = 0;

    for (size_t i = 0; i < m_workers.size(); ++i) {
        m_workers[i]->join();
    }
}


void Workers::submit(const xmrig::Job &result)
{
    uv_mutex_lock(&m_mutex);
    m_queue.push_back(result);
    uv_mutex_unlock(&m_mutex);

    uv_async_send(&m_async);
}


#ifndef XMRIG_NO_API
void Workers::threadsSummary(rapidjson::Document &)
{
}
#endif


void Workers::onReady(void *arg)
{
    auto handle = static_cast<Handle*>(arg);

    IWorker *worker = new CudaWorker(handle);
    handle->setWorker(worker);

    start(worker);
}


void Workers::onReport(uv_timer_t *)
{
    m_hashrate->print();

    if (NvmlApi::isAvailable()) {
        printHealth();
    }
}


void Workers::onResult(uv_async_t *)
{
    JobBaton *baton = new JobBaton();

    uv_mutex_lock(&m_mutex);
    while (!m_queue.empty()) {
        baton->jobs.push_back(std::move(m_queue.front()));
        m_queue.pop_front();
    }
    uv_mutex_unlock(&m_mutex);

    uv_queue_work(uv_default_loop(), &baton->request,
        [](uv_work_t* req) {
            JobBaton *baton = static_cast<JobBaton*>(req->data);
            if (baton->jobs.empty()) {
                return;
            }

            cryptonight_ctx *ctx = nullptr;
            MemInfo info;

            for (const xmrig::Job &job : baton->jobs) {
                xmrig::JobResult result(job);

                bool ok;

#               ifdef XMRIG_ALGO_RANDOMX
                if (job.algorithm().algo() == xmrig::RANDOM_X) {
                    uv_rwlock_wrlock(&m_rx_dataset_lock);

                    if (m_rx_variant != job.algorithm().variant()) {
                        m_rx_variant = job.algorithm().variant();

                        switch (job.algorithm().variant()) {
                        case xmrig::VARIANT_RX_WOW:
                            randomx_apply_config(RandomX_WowneroConfig);
                            break;
                        case xmrig::VARIANT_RX_LOKI:
                            randomx_apply_config(RandomX_LokiConfig);
                            break;
                        default:
                            randomx_apply_config(RandomX_MoneroConfig);
                            break;
                        }
                    }

                    if (!m_rx_vm) {
                        int flags = RANDOMX_FLAG_LARGE_PAGES | RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT;
                        if (!xmrig::Cpu::info()->hasAES()) {
                            flags |= RANDOMX_FLAG_HARD_AES;
                        }

                        m_rx_vm = randomx_create_vm(static_cast<randomx_flags>(flags), nullptr, m_rx_dataset, m_rx_scratchpad);
                        if (!m_rx_vm) {
                            m_rx_vm = randomx_create_vm(static_cast<randomx_flags>(flags - RANDOMX_FLAG_LARGE_PAGES), nullptr, m_rx_dataset, m_rx_scratchpad);
                        }
                    }
                    randomx_calculate_hash(m_rx_vm, job.blob(), job.size(), result.result);

                    uv_rwlock_wrunlock(&m_rx_dataset_lock);

                    ok = *reinterpret_cast<uint64_t*>(result.result + 24) < job.target();
                }
                else
#               endif
                {
                    if (!ctx) {
                        info = Mem::create(&ctx, baton->jobs[0].algorithm().algo(), 1);
                    }
                    ok = CryptoNight::hash(job, result, ctx);
                }

                if (ok) {
                    baton->results.push_back(result);
                }
                else {
                    baton->errors++;
                }
            }

            if (ctx) {
                Mem::release(&ctx, 1, info);
            }
        },
        [](uv_work_t* req, int) {
            JobBaton *baton = static_cast<JobBaton*>(req->data);

            for (const xmrig::JobResult &result : baton->results) {
                m_listener->onJobResult(result);
            }

            if (baton->errors > 0 && !baton->jobs.empty()) {
                LOG_ERR("THREAD #%d COMPUTE ERROR", baton->jobs[0].threadId());
            }

            delete baton;
        }
    );
}


void Workers::onTick(uv_timer_t *)
{
    for (Handle *handle : m_workers) {
        if (!handle->worker()) {
            return;
        }

        m_hashrate->add(handle->threadId(), handle->worker()->hashCount(), handle->worker()->timestamp());
    }

    if ((m_ticks++ & 0xF) == 0)  {
        m_hashrate->updateHighest();
    }
}


void Workers::start(IWorker *worker)
{
    worker->start();
}

#ifdef XMRIG_ALGO_RANDOMX
randomx_dataset* Workers::getDataset(const uint8_t* seed_hash, xmrig::Variant variant)
{
    uv_rwlock_wrlock(&m_rx_dataset_lock);

    // Check if we need to allocate the scratchpad
    if (!m_rx_scratchpad) {
        try {
            m_rx_scratchpad = (uint8_t*) randomx::LargePageAllocator::allocMemory(RANDOMX_SCRATCHPAD_L3_MAX_SIZE);
        }
        catch (...) {
            m_rx_scratchpad = (uint8_t*)randomx::AlignedAllocator<64>::allocMemory(RANDOMX_SCRATCHPAD_L3_MAX_SIZE);
        }
    }

    // Check if we need to update cache and dataset
    if (m_rx_dataset && ((memcmp(m_rx_seed_hash, seed_hash, sizeof(m_rx_seed_hash)) == 0) && (m_rx_variant == variant))) {
        uv_rwlock_wrunlock(&m_rx_dataset_lock);
        return m_rx_dataset;
    }

    if (!m_rx_dataset) {
        randomx_dataset* dataset = randomx_alloc_dataset(RANDOMX_FLAG_LARGE_PAGES);
        if (!dataset) {
            dataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
        }
        m_rx_cache = randomx_alloc_cache(static_cast<randomx_flags>(RANDOMX_FLAG_JIT | RANDOMX_FLAG_LARGE_PAGES));
        if (!m_rx_cache) {
            m_rx_cache = randomx_alloc_cache(RANDOMX_FLAG_JIT);
        }
        m_rx_dataset = dataset;
    }

    const uint32_t num_threads = std::thread::hardware_concurrency();
    LOG_INFO("Started updating RandomX dataset (%u threads)", num_threads);

    if (m_rx_variant != variant) {
        switch (variant) {
            case xmrig::VARIANT_RX_WOW:
                randomx_apply_config(RandomX_WowneroConfig);
                break;
            case xmrig::VARIANT_RX_LOKI:
                randomx_apply_config(RandomX_LokiConfig);
                break;
            default:
                randomx_apply_config(RandomX_MoneroConfig);
                break;
        }
        m_rx_variant = variant;
    }

    if (memcmp(m_rx_seed_hash, seed_hash, sizeof(m_rx_seed_hash)) != 0) {
        memcpy(m_rx_seed_hash, seed_hash, sizeof(m_rx_seed_hash));
        randomx_init_cache(m_rx_cache, m_rx_seed_hash, sizeof(m_rx_seed_hash));
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i) {
        const uint32_t a = (randomx_dataset_item_count() * i) / num_threads;
        const uint32_t b = (randomx_dataset_item_count() * (i + 1)) / num_threads;
        threads.emplace_back(randomx_init_dataset, m_rx_dataset, m_rx_cache, a, b - a);
    }
    for (uint32_t i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    if (m_rx_vm) {
        randomx_vm_set_dataset(m_rx_vm, m_rx_dataset);
    }

    LOG_INFO("Finished updating RandomX dataset (%u threads)", num_threads);

    uv_rwlock_wrunlock(&m_rx_dataset_lock);

    return m_rx_dataset;
}
#endif
