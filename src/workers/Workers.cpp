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

#ifdef __GNUC__
#   include <mm_malloc.h>
#else
#   include <malloc.h>
#endif


#include "api/Api.h"
#include "crypto/CryptoNight.h"
#include "interfaces/IJobResultListener.h"
#include "log/Log.h"
#include "nvidia/NvmlApi.h"
#include "Options.h"
#include "workers/CudaWorker.h"
#include "workers/GpuThread.h"
#include "workers/Handle.h"
#include "workers/Hashrate.h"
#include "workers/Workers.h"


bool Workers::m_active = false;
bool Workers::m_enabled = true;
Hashrate *Workers::m_hashrate = nullptr;
IJobResultListener *Workers::m_listener = nullptr;
Job Workers::m_job;
std::atomic<int> Workers::m_paused;
std::atomic<uint64_t> Workers::m_sequence;
std::list<Job> Workers::m_queue;
std::vector<Handle*> Workers::m_workers;
uint64_t Workers::m_ticks = 0;
uv_async_t Workers::m_async;
uv_mutex_t Workers::m_mutex;
uv_rwlock_t Workers::m_rwlock;
uv_timer_t Workers::m_reportTimer;
uv_timer_t Workers::m_timer;


struct JobBaton
{
    uv_work_t request;
    std::vector<Job> jobs;
    std::vector<JobResult> results;
    int errors = 0;

    JobBaton() {
        request.data = this;
    }
};


Job Workers::job()
{
    uv_rwlock_rdlock(&m_rwlock);
    Job job = m_job;
    uv_rwlock_rdunlock(&m_rwlock);

    return job;
}


void Workers::printHashrate(bool detail)
{
    if (detail) {
       for (const GpuThread *thread : Options::i()->threads()) {
            m_hashrate->print(thread->threadId(), thread->index());
        }
    }

    m_hashrate->print();
}


void Workers::printHealth()
{
    if (!NvmlApi::isAvailable()) {
        LOG_ERR("NVML GPU monitoring is not available");
        return;
    }

    Health health;
    for (const GpuThread *thread : Options::i()->threads()) {
        if (!NvmlApi::health(thread->nvmlId(), health)) {
            continue;
        }

        const uint32_t temp = health.temperature;

        if (health.clock && health.power) {
            if (Options::i()->colors()) {
                LOG_INFO("\x1B[00;35mGPU #%d: \x1B[01m%u\x1B[00;35m/\x1B[01m%u MHz\x1B[00;35m \x1B[01m%uW\x1B[00;35m %s%uC\x1B[00;35m FAN \x1B[01m%u%%",
                    thread->index(), health.clock, health.memClock, health.power / 1000, (temp < 45 ? "\x1B[01;32m" : (temp > 65 ? "\x1B[01;31m" : "\x1B[01;33m")), temp, health.fanSpeed);
            }
            else {
                LOG_INFO(" * GPU #%d: %u/%u MHz %uW %uC FAN %u%%", thread->index(), health.clock, health.memClock, health.power / 1000, health.temperature, health.fanSpeed);
            }

            continue;
        }

        if (Options::i()->colors()) {
            LOG_INFO("\x1B[00;35mGPU #%d: %s%uC\x1B[00;35m FAN \x1B[01m%u%%",
                thread->index(), (temp < 45 ? "\x1B[01;32m" : (temp > 65 ? "\x1B[01;31m" : "\x1B[01;33m")), temp, health.fanSpeed);
        }
        else {
            LOG_INFO(" * GPU #%d: %uC FAN %u%%", thread->index(), health.temperature, health.fanSpeed);
        }
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


void Workers::setJob(const Job &job, bool donate)
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


void Workers::start(const std::vector<GpuThread*> &threads)
{
    const size_t count = threads.size();
    m_hashrate = new Hashrate((int) count);

    uv_mutex_init(&m_mutex);
    uv_rwlock_init(&m_rwlock);

    m_sequence = 1;
    m_paused   = 1;

    uv_async_init(uv_default_loop(), &m_async, Workers::onResult);
    uv_timer_init(uv_default_loop(), &m_timer);
    uv_timer_start(&m_timer, Workers::onTick, 500, 500);

    for (size_t i = 0; i < count; ++i) {
        Handle *handle = new Handle((int) i, threads[i], (int) count, Options::i()->algorithm());
        m_workers.push_back(handle);
        handle->start(Workers::onReady);
    }

    const int printTime = Options::i()->printTime();
    if (printTime > 0) {
        uv_timer_init(uv_default_loop(), &m_reportTimer);
        uv_timer_start(&m_reportTimer, Workers::onReport, (printTime + 4) * 1000, printTime * 1000);
    }
}


void Workers::stop()
{
    if (Options::i()->printTime() > 0) {
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


void Workers::submit(const Job &result)
{
    uv_mutex_lock(&m_mutex);
    m_queue.push_back(result);
    uv_mutex_unlock(&m_mutex);

    uv_async_send(&m_async);
}


void Workers::onReady(void *arg)
{
    auto handle = static_cast<Handle*>(arg);
    handle->setWorker(new CudaWorker(handle));

    handle->worker()->start();
}


void Workers::onResult(uv_async_t *handle)
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
            cryptonight_ctx *ctx = static_cast<cryptonight_ctx*>(_mm_malloc(sizeof(cryptonight_ctx), 16));

            for (const Job &job : baton->jobs) {
                JobResult result(job);

                if (CryptoNight::hash(job, result, ctx)) {
                    baton->results.push_back(result);
                }
                else {
                    baton->errors++;
                }
            }

            _mm_free(ctx);
        },
        [](uv_work_t* req, int status) {
            JobBaton *baton = static_cast<JobBaton*>(req->data);

            for (const JobResult &result : baton->results) {
                m_listener->onJobResult(result);
            }

            if (baton->errors > 0 && !baton->jobs.empty()) {
                LOG_ERR("GPU #%d COMPUTE ERROR", baton->jobs[0].threadId());
            }

            delete baton;
        }
    );
}


void Workers::onReport(uv_timer_t *handle)
{
    m_hashrate->print();

    if (NvmlApi::isAvailable()) {
        printHealth();
    }
}


void Workers::onTick(uv_timer_t *handle)
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

#   ifndef XMRIG_NO_API
    Api::tick(m_hashrate);

    if ((m_ticks++ & 0x4) == 0) {
        std::vector<Health> records;
        Health health;
        for (const GpuThread *thread : Options::i()->threads()) {
            NvmlApi::health(thread->nvmlId(), health);
            records.push_back(health);
        }

        Api::setHealth(records);
    }
#   endif
}
