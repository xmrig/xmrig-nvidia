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

#include <string.h>
#include <uv.h>
#include <inttypes.h>


#include "common/config/ConfigLoader.h"
#include "common/log/Log.h"
#include "core/Config.h"
#include "core/ConfigCreator.h"
#include "crypto/CryptoNight_constants.h"
#include "nvidia/NvmlApi.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "workers/CudaThread.h"


xmrig::Config::Config() : xmrig::CommonConfig(),
    m_autoConf(false),
    m_shouldSave(false),
    m_maxGpuThreads(64),
    m_maxGpuUsage(100)
{
}


bool xmrig::Config::isCNv2() const
{
    if (algorithm().algo() == CRYPTONIGHT_PICO) {
        return true;
    }

    if (algorithm().algo() != CRYPTONIGHT) {
        return false;
    }

    for (const Pool &pool : m_pools.data()) {
        const Variant variant = pool.algorithm().variant();

        if (variant == VARIANT_2 || variant == VARIANT_AUTO || variant == VARIANT_HALF || variant == VARIANT_WOW) {
            return true;
        }
    }

    return false;
}


bool xmrig::Config::reload(const char *json)
{
    return xmrig::ConfigLoader::reload(this, json);
}


void xmrig::Config::getJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    doc.SetObject();

    auto &allocator = doc.GetAllocator();

    doc.AddMember("algo", StringRef(algorithm().name()), allocator);

    Value api(kObjectType);
    api.AddMember("port",             apiPort(), allocator);
    api.AddMember("access-token",     apiToken() ? Value(StringRef(apiToken())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("id",               apiId() ? Value(StringRef(apiId())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("worker-id",        apiWorkerId() ? Value(StringRef(apiWorkerId())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("ipv6",             isApiIPv6(), allocator);
    api.AddMember("restricted",       isApiRestricted(), allocator);
    doc.AddMember("api",              api, allocator);

    doc.AddMember("background",       isBackground(), allocator);
    doc.AddMember("colors",           isColors(), allocator);
    doc.AddMember("cuda-bfactor",     m_cudaCLI.bfactor(), allocator);
    doc.AddMember("cuda-bsleep",      m_cudaCLI.bsleep(), allocator);
    doc.AddMember("cuda-max-threads", m_maxGpuThreads, allocator);
    doc.AddMember("donate-level",     donateLevel(), allocator);
    doc.AddMember("log-file",         logFile() ? Value(StringRef(logFile())).Move() : Value(kNullType).Move(), allocator);
    doc.AddMember("pools",            m_pools.toJSON(doc), allocator);
    doc.AddMember("print-time",       printTime(), allocator);
    doc.AddMember("retries",          m_pools.retries(), allocator);
    doc.AddMember("retry-pause",      m_pools.retryPause(), allocator);

    Value threads(kArrayType);
    for (const IThread *thread : m_threads) {
        threads.PushBack(thread->toConfig(doc), allocator);
    }
    doc.AddMember("threads", threads, allocator);

    doc.AddMember("user-agent", userAgent() ? Value(StringRef(userAgent())).Move() : Value(kNullType).Move(), allocator);
    doc.AddMember("syslog",     isSyslog(), allocator);
    doc.AddMember("watch",      m_watch, allocator);
}


xmrig::Config *xmrig::Config::load(Process *process, IConfigListener *listener)
{
    return static_cast<Config*>(ConfigLoader::load(process, new ConfigCreator(), listener));
}


bool xmrig::Config::finalize()
{
    if (m_state != NoneState) {
        return CommonConfig::finalize();
    }

    if (!CommonConfig::finalize()) {
        return false;
    }

    if (m_threads.empty() && !m_cudaCLI.setup(m_threads, algorithm().algo(), isCNv2())) {
        m_autoConf   = true;
        m_shouldSave = true;
        m_cudaCLI.autoConf(m_threads, algorithm().algo(), isCNv2());

        for (IThread *thread : m_threads) {
            static_cast<CudaThread *>(thread)->limit(m_maxGpuUsage, m_maxGpuThreads);
        }
    }

    NvmlApi::init();
    NvmlApi::bind(m_threads);
    return true;
}


bool xmrig::Config::parseString(int key, const char *arg)
{
    if (!CommonConfig::parseString(key, arg)) {
        return false;
    }

    switch (key) {
    case CudaBFactorKey: /* --cuda-bfactor */
        m_cudaCLI.parseBFactor(arg);
        break;

    case CudaBSleepKey: /* --cuda-bsleep */
        m_cudaCLI.parseBSleep(arg);
        break;

    case CudaDevicesKey: /* --cuda-devices */
        m_cudaCLI.parseDevices(arg);
        break;

    case CudaLaunchKey: /* --cuda-launch */
        m_cudaCLI.parseLaunch(arg);
        break;

    case CudaAffinityKey: /* --cuda-affinity */
        m_cudaCLI.parseAffinity(arg);
        break;

    case CudaMaxThreadsKey:
    case CudaMaxUsageKey:
        return parseUint64(key, strtoul(arg, nullptr, 10));

    default:
        break;
    }

    return true;
}


bool xmrig::Config::parseUint64(int key, uint64_t arg)
{
    if (!CommonConfig::parseUint64(key, arg)) {
        return false;
    }

    switch (key) {
    case CudaMaxThreadsKey: /* --cuda-max-threads */
        m_maxGpuThreads = static_cast<int>(arg);
        break;

    case CudaBFactorKey: /* --cuda-bfactor */
        m_cudaCLI.addBFactor(static_cast<int>(arg));
        break;

    case CudaBSleepKey: /* --cuda-bsleep */
        m_cudaCLI.addBSleep(static_cast<int>(arg));
        break;

    case CudaMaxUsageKey: /* --max-gpu-usage */
        m_maxGpuUsage = static_cast<int>(arg);
        break;

    default:
        break;
    }

    return true;
}


void xmrig::Config::parseJSON(const rapidjson::Document &doc)
{
    CommonConfig::parseJSON(doc);

    const rapidjson::Value &threads = doc["threads"];

    if (threads.IsArray()) {
        for (const rapidjson::Value &value : threads.GetArray()) {
            if (!value.IsObject()) {
                continue;
            }

            if (value.HasMember("threads")) {
                parseThread(value);
            }
        }
    }
}


void xmrig::Config::parseThread(const rapidjson::Value &object)
{
    CudaThread *thread = new CudaThread(object);
    if (thread->init(algorithm().algo())) {
        m_threads.push_back(thread);
        return;
    }

    delete thread;
}
