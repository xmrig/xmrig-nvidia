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
#include "Cpu.h"
#include "crypto/CryptoNight_constants.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"


xmrig::Config::Config() : xmrig::CommonConfig(),
    m_autoConf(false),
    m_cache(true),
    m_shouldSave(false),
    m_platformIndex(0),
#   ifdef _WIN32
    m_loader("OpenCL.dll")
#   else
    m_loader("libOpenCL.so")
#   endif
{
}


xmrig::Config::~Config()
{
}


bool xmrig::Config::oclInit()
{
    return true;
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
    api.AddMember("port",         apiPort(), allocator);
    api.AddMember("access-token", apiToken() ? Value(StringRef(apiToken())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("worker-id",    apiWorkerId() ? Value(StringRef(apiWorkerId())).Move() : Value(kNullType).Move(), allocator);
    api.AddMember("ipv6",         isApiIPv6(), allocator);
    api.AddMember("restricted",   isApiRestricted(), allocator);
    doc.AddMember("api",          api, allocator);

    doc.AddMember("background",   isBackground(), allocator);
    doc.AddMember("cache",        isOclCache(), allocator);
    doc.AddMember("colors",       isColors(), allocator);
    doc.AddMember("donate-level", donateLevel(), allocator);
    doc.AddMember("log-file",     logFile() ? Value(StringRef(logFile())).Move() : Value(kNullType).Move(), allocator);
    doc.AddMember("opencl-platform", platformIndex(), allocator);
    doc.AddMember("opencl-loader",   StringRef(loader()), allocator);

    Value pools(kArrayType);

    for (const Pool &pool : m_activePools) {
        pools.PushBack(pool.toJSON(doc), allocator);
    }

    doc.AddMember("pools",         pools, allocator);
    doc.AddMember("print-time",    printTime(), allocator);
    doc.AddMember("retries",       retries(), allocator);
    doc.AddMember("retry-pause",   retryPause(), allocator);

    Value threads(kArrayType);
    //for (const IThread *thread : m_threads) {
    //    threads.PushBack(thread->toConfig(doc), allocator);
    //}
    doc.AddMember("threads", threads, allocator);

    doc.AddMember("user-agent", userAgent() ? Value(StringRef(userAgent())).Move() : Value(kNullType).Move(), allocator);
    doc.AddMember("syslog",     isSyslog(), allocator);
    doc.AddMember("watch",      m_watch, allocator);
}


xmrig::Config *xmrig::Config::load(int argc, char **argv, IWatcherListener *listener)
{
    return static_cast<Config*>(ConfigLoader::load(argc, argv, new ConfigCreator(), listener));
}


bool xmrig::Config::finalize()
{
    if (m_state != NoneState) {
        return CommonConfig::finalize();
    }

    if (!CommonConfig::finalize()) {
        return false;
    }

    return true;
}


bool xmrig::Config::parseBoolean(int key, bool enable)
{
    if (!CommonConfig::parseBoolean(key, enable)) {
        return false;
    }

    switch (key) {
    case OclCache: /* cache */
        m_cache = enable;
        break;

    default:
        break;
    }

    return true;
}


bool xmrig::Config::parseString(int key, const char *arg)
{
    if (!CommonConfig::parseString(key, arg)) {
        return false;
    }

    switch (key) {
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
    default:
        break;
    }

    return true;
}


void xmrig::Config::parseJSON(const rapidjson::Document &doc)
{
    const rapidjson::Value &threads = doc["threads"];

    if (threads.IsArray()) {
        for (const rapidjson::Value &value : threads.GetArray()) {
            if (!value.IsObject()) {
                continue;
            }

            if (value.HasMember("intensity")) {
                parseThread(value);
            }
        }
    }
}


void xmrig::Config::parseThread(const rapidjson::Value &object)
{
    //m_threads.push_back(new OclThread(object));
}
