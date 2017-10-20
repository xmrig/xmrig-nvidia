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


#include <string.h>
#include <uv.h>


#ifdef _MSC_VER
#   include "getopt/getopt.h"
#else
#   include <getopt.h>
#endif


#include "Cpu.h"
#include "donate.h"
#include "net/Url.h"
#include "nvidia/cryptonight.h"
#include "nvidia/NvmlApi.h"
#include "Options.h"
#include "Platform.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "version.h"
#include "workers/GpuThread.h"


#ifndef ARRAY_SIZE
#   define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif


Options *Options::m_self = nullptr;


static char const usage[] = "\
Usage: " APP_ID " [OPTIONS]\n\
\n\
Options:\n\
  -a, --algo=ALGO           cryptonight (default) or cryptonight-lite\n\
  -o, --url=URL             URL of mining server\n\
  -O, --userpass=U:P        username:password pair for mining server\n\
  -u, --user=USERNAME       username for mining server\n\
  -p, --pass=PASSWORD       password for mining server\n\
  -k, --keepalive           send keepalived for prevent timeout (need pool support)\n\
  -r, --retries=N           number of times to retry before switch to backup server (default: 5)\n\
  -R, --retry-pause=N       time to pause between retries (default: 5)\n\
      --cuda-devices=N      List of CUDA devices to use.\n\
      --cuda-launch=TxB     List of launch config for the CryptoNight kernel\n\
      --cuda-max-threads=N  limit maximum count of GPU threads in automatic mode\n\
      --cuda-bfactor=[0-12] run CryptoNight core kernel in smaller pieces\n\
      --cuda-bsleep=N       insert a delay of N microseconds between kernel launches\n\
      --cuda-affinity=N     affine GPU threads to a CPU\n\
      --no-color            disable colored output\n\
      --donate-level=N      donate level, default 5%% (5 minutes in 100 minutes)\n\
      --user-agent          set custom user-agent string for pool\n\
  -B, --background          run the miner in the background\n\
  -c, --config=FILE         load a JSON-format configuration file\n\
  -l, --log-file=FILE       log all output to a file\n"
# ifdef HAVE_SYSLOG_H
"\
  -S, --syslog              use system log for output messages\n"
# endif
"\
      --nicehash            enable nicehash support\n\
      --print-time=N        print hashrate report every N seconds\n\
      --api-port=N          port for the miner API\n\
      --api-access-token=T  access token for API\n\
      --api-worker-id=ID    custom worker-id for API\n\
  -h, --help                display this help and exit\n\
  -V, --version             output version information and exit\n\
";


static char const short_options[] = "a:c:khBp:Px:r:R:s:T:o:u:O:Vl:S";


static struct option const options[] = {
    { "algo",             1, nullptr, 'a'  },
    { "api-access-token", 1, nullptr, 4001 },
    { "api-port",         1, nullptr, 4000 },
    { "api-worker-id",    1, nullptr, 4002 },
    { "background",       0, nullptr, 'B'  },
    { "bfactor",          1, nullptr, 1201 },
    { "bsleep",           1, nullptr, 1202 },
    { "config",           1, nullptr, 'c'  },
    { "cuda-affinity",    1, nullptr, 1205 },
    { "cuda-bfactor",     1, nullptr, 1201 }, // deprecated, use --cuda-bfactor instead.
    { "cuda-bsleep",      1, nullptr, 1202 }, // deprecated, use --cuda-bsleep instead.
    { "cuda-devices",     1, nullptr, 1203 },
    { "cuda-launch",      1, nullptr, 1204 },
    { "cuda-max-threads", 1, nullptr, 1200 },
    { "donate-level",     1, nullptr, 1003 },
    { "help",             0, nullptr, 'h'  },
    { "keepalive",        0, nullptr ,'k'  },
    { "log-file",         1, nullptr, 'l'  },
    { "max-gpu-threads",  1, nullptr, 1200 }, // deprecated, use --cuda-max-threads instead.
    { "max-gpu-usage",    1, nullptr, 1004 }, // deprecated.
    { "nicehash",         0, nullptr, 1006 },
    { "no-color",         0, nullptr, 1002 },
    { "pass",             1, nullptr, 'p'  },
    { "print-time",       1, nullptr, 1007 },
    { "retries",          1, nullptr, 'r'  },
    { "retry-pause",      1, nullptr, 'R'  },
    { "syslog",           0, nullptr, 'S'  },
    { "url",              1, nullptr, 'o'  },
    { "user",             1, nullptr, 'u'  },
    { "user-agent",       1, nullptr, 1008 },
    { "userpass",         1, nullptr, 'O'  },
    { "version",          0, nullptr, 'V'  },
    { 0, 0, 0, 0 }
};


static struct option const config_options[] = {
    { "algo",             1, nullptr, 'a'  },
    { "background",       0, nullptr, 'B'  },
    { "bfactor",          1, nullptr, 1201 }, // deprecated, use --cuda-bfactor instead.
    { "bsleep",           1, nullptr, 1202 }, // deprecated, use --cuda-bsleep instead.
    { "colors",           0, nullptr, 2000 },
    { "cuda-bfactor",     1, nullptr, 1201 },
    { "cuda-bsleep",      1, nullptr, 1202 },
    { "cuda-max-threads", 1, nullptr, 1200 },
    { "donate-level",     1, nullptr, 1003 },
    { "log-file",         1, nullptr, 'l'  },
    { "max-gpu-threads",  1, nullptr, 1200 }, // deprecated, use --cuda-max-threads instead.
    { "max-gpu-usage",    1, nullptr, 1004 }, // deprecated.
    { "print-time",       1, nullptr, 1007 },
    { "retries",          1, nullptr, 'r'  },
    { "retry-pause",      1, nullptr, 'R'  },
    { "syslog",           0, nullptr, 'S'  },
    { "user-agent",       1, nullptr, 1008 },
    { 0, 0, 0, 0 }
};


static struct option const pool_options[] = {
    { "url",           1, nullptr, 'o'  },
    { "pass",          1, nullptr, 'p'  },
    { "user",          1, nullptr, 'u'  },
    { "userpass",      1, nullptr, 'O'  },
    { "keepalive",     0, nullptr ,'k'  },
    { "nicehash",      0, nullptr, 1006 },
    { 0, 0, 0, 0 }
};


static struct option const api_options[] = {
    { "port",          1, nullptr, 4000 },
    { "access-token",  1, nullptr, 4001 },
    { "worker-id",     1, nullptr, 4002 },
    { 0, 0, 0, 0 }
};


static const char *algo_names[] = {
    "cryptonight",
#   ifndef XMRIG_NO_AEON
    "cryptonight-lite"
#   endif
};


Options *Options::parse(int argc, char **argv)
{
    Options *options = new Options(argc, argv);
    if (options->isReady()) {
        m_self = options;
        return m_self;
    }

    delete options;
    return nullptr;
}


bool Options::save()
{
    if (m_configName == nullptr) {
        return false;
    }

    uv_fs_t req;
    const int fd = uv_fs_open(uv_default_loop(), &req, m_configName, O_WRONLY | O_CREAT | O_TRUNC, 0644, nullptr);
    if (fd < 0) {
        return false;
    }

    uv_fs_req_cleanup(&req);

    rapidjson::Document doc;
    doc.SetObject();

    auto &allocator = doc.GetAllocator();

    doc.AddMember("algo",         rapidjson::StringRef(algoName()), allocator);
    doc.AddMember("background",   m_background, allocator);
    doc.AddMember("colors",       m_colors, allocator);
    doc.AddMember("donate-level", m_donateLevel, allocator);
    doc.AddMember("log-file",     m_logFile ? rapidjson::Value(rapidjson::StringRef(algoName())).Move() : rapidjson::Value(rapidjson::kNullType).Move(), allocator);
    doc.AddMember("print-time",   m_printTime, allocator);
    doc.AddMember("retries",      m_retries, allocator);
    doc.AddMember("retry-pause",  m_retryPause, allocator);

#   ifdef HAVE_SYSLOG_H
    doc.AddMember("syslog", m_syslog, allocator);
#   endif

    rapidjson::Value threads(rapidjson::kArrayType);
    for (const GpuThread *thread : m_threads) {
        rapidjson::Value obj(rapidjson::kObjectType);

        obj.AddMember("index",   thread->index(), allocator);
        obj.AddMember("threads", thread->threads(), allocator);
        obj.AddMember("blocks",  thread->blocks(), allocator);
        obj.AddMember("bfactor", thread->bfactor(), allocator);
        obj.AddMember("bsleep",  thread->bsleep(), allocator);

        if (thread->affinity() >= 0) {
            obj.AddMember("affine_to_cpu", thread->affinity(), allocator);
        }
        else {
            obj.AddMember("affine_to_cpu", false, allocator);
        }

        threads.PushBack(obj, allocator);
    }

    rapidjson::Value pools(rapidjson::kArrayType);
    char tmp[256];

    for (const Url *url : m_pools) {
        rapidjson::Value obj(rapidjson::kObjectType);
        snprintf(tmp, sizeof(tmp) - 1, "%s:%d", url->host(), url->port());

        obj.AddMember("url",       rapidjson::StringRef(tmp), allocator);
        obj.AddMember("user",      rapidjson::StringRef(url->user()), allocator);
        obj.AddMember("pass",      rapidjson::StringRef(url->password()), allocator);
        obj.AddMember("keepalive", url->isKeepAlive(), allocator);
        obj.AddMember("nicehash",  url->isNicehash(), allocator);

        pools.PushBack(obj, allocator);
    }

    rapidjson::Value api(rapidjson::kObjectType);
    api.AddMember("port",         m_apiPort, allocator);
    api.AddMember("access-token", m_apiToken ? rapidjson::Value(rapidjson::StringRef(m_apiToken)).Move() : rapidjson::Value(rapidjson::kNullType).Move(), allocator);
    api.AddMember("worker-id",    m_apiWorkerId ? rapidjson::Value(rapidjson::StringRef(m_apiWorkerId)).Move() : rapidjson::Value(rapidjson::kNullType).Move(), allocator);

    doc.AddMember("threads", threads, allocator);
    doc.AddMember("pools",   pools, allocator);
    doc.AddMember("api",     api, allocator);

    FILE *fp = fdopen(fd, "w");

    char buf[4096];
    rapidjson::FileWriteStream os(fp, buf, sizeof(buf));
    rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(os);
    doc.Accept(writer);

    fclose(fp);

    uv_fs_close(uv_default_loop(), &req, fd, nullptr);
    uv_fs_req_cleanup(&req);

    return true;
}


const char *Options::algoName() const
{
    return algo_names[m_algo];
}


Options::Options(int argc, char **argv) :
    m_autoConf(false),
    m_background(false),
    m_colors(true),
    m_ready(false),
    m_syslog(false),
    m_apiToken(nullptr),
    m_apiWorkerId(nullptr),
    m_configName(nullptr),
    m_logFile(nullptr),
    m_userAgent(nullptr),
    m_algo(0),
    m_algoVariant(0),
    m_apiPort(0),
    m_donateLevel(kDonateLevel),
    m_maxGpuThreads(64),
    m_maxGpuUsage(100),
    m_printTime(60),
    m_retries(5),
    m_retryPause(5),
    m_threads(0)
{
    NvmlApi::init();

    m_pools.push_back(new Url());

    int key;

    while (1) {
        key = getopt_long(argc, argv, short_options, options, NULL);
        if (key < 0) {
            break;
        }

        if (!parseArg(key, optarg)) {
            return;
        }
    }

    if (optind < argc) {
        fprintf(stderr, "%s: unsupported non-option argument '%s'\n", argv[0], argv[optind]);
        return;
    }

    if (!m_pools[0]->isValid()) {
        parseConfig(Platform::defaultConfigName());
    }

    if (!m_pools[0]->isValid()) {
        fprintf(stderr, "No pool URL supplied. Exiting.\n");
        return;
    }

    m_algoVariant = Cpu::hasAES() ? AV1_AESNI : AV3_SOFT_AES;

    if (m_threads.empty() && !m_cudaCLI.setup(m_threads)) {
        m_autoConf = true;
        m_cudaCLI.autoConf(m_threads);

        for (GpuThread *thread : m_threads) {
            thread->limit(m_maxGpuUsage, m_maxGpuThreads);
        }
    }

    for (Url *url : m_pools) {
        url->applyExceptions();
    }

    NvmlApi::bind(m_threads);
    m_ready = true;
}


Options::~Options()
{
    NvmlApi::release();
}


bool Options::getJSON(const char *fileName, rapidjson::Document &doc)
{
    uv_fs_t req;
    const int fd = uv_fs_open(uv_default_loop(), &req, fileName, O_RDONLY, 0644, nullptr);
    if (fd < 0) {
        fprintf(stderr, "unable to open %s: %s\n", fileName, uv_strerror(fd));
        return false;
    }

    uv_fs_req_cleanup(&req);

    FILE *fp = fdopen(fd, "rb");
    char buf[8192];
    rapidjson::FileReadStream is(fp, buf, sizeof(buf));

    doc.ParseStream(is);

    uv_fs_close(uv_default_loop(), &req, fd, nullptr);
    uv_fs_req_cleanup(&req);

    if (doc.HasParseError()) {
        printf("%s:%d: %s\n", fileName, (int)doc.GetErrorOffset(), rapidjson::GetParseError_En(doc.GetParseError()));
        return false;
    }

    return doc.IsObject();
}


bool Options::parseArg(int key, const char *arg)
{
    switch (key) {
    case 'a': /* --algo */
        if (!setAlgo(arg)) {
            return false;
        }
        break;

    case 'o': /* --url */
        if (m_pools.size() > 1 || m_pools[0]->isValid()) {
            Url *url = new Url(arg);
            if (url->isValid()) {
                m_pools.push_back(url);
            }
            else {
                delete url;
            }
        }
        else {
            m_pools[0]->parse(arg);
        }

        if (!m_pools.back()->isValid()) {
            return false;
        }
        break;

    case 'O': /* --userpass */
        if (!m_pools.back()->setUserpass(arg)) {
            return false;
        }
        break;

    case 'u': /* --user */
        m_pools.back()->setUser(arg);
        break;

    case 'p': /* --pass */
        m_pools.back()->setPassword(arg);
        break;

    case 'l': /* --log-file */
        free(m_logFile);
        m_logFile = strdup(arg);
        m_colors = false;
        break;

    case 4001: /* --access-token */
        free(m_apiToken);
        m_apiToken = strdup(arg);
        break;

    case 4002: /* --worker-id */
        free(m_apiWorkerId);
        m_apiWorkerId = strdup(arg);
        break;

    case 1201: /* --bfactor */
        m_cudaCLI.parseBFactor(arg);
        break;

    case 1202: /* --bsleep */
        m_cudaCLI.parseBSleep(arg);
        break;

    case 1203: /* --cuda-devices */
        m_cudaCLI.parseDevices(arg);
        break;

    case 1204: /* --cuda-launch */
        m_cudaCLI.parseLaunch(arg);
        break;

    case 1205: /* --cuda-affinity */
        m_cudaCLI.parseAffinity(arg);
        break;

    case 'r':  /* --retries */
    case 'R':  /* --retry-pause */
    case 't':  /* --threads */
    case 'v':  /* --av */
    case 1003: /* --donate-level */
    case 1004: /* --max-gpu-usage */
    case 1007: /* --print-time */
    case 1200: /* --max-gpu-threads */

    case 4000: /* --api-port */
        return parseArg(key, strtol(arg, nullptr, 10));

    case 'B':  /* --background */
    case 'k':  /* --keepalive */
    case 'S':  /* --syslog */
    case 1005: /* --safe */
    case 1006: /* --nicehash */
        return parseBoolean(key, true);

    case 1002: /* --no-color */
        return parseBoolean(key, false);

    case 'V': /* --version */
        showVersion();
        return false;

    case 'h': /* --help */
        showUsage(0);
        return false;

    case 'c': /* --config */
        parseConfig(arg);
        break;

    case 1008: /* --user-agent */
        free(m_userAgent);
        m_userAgent = strdup(arg);
        break;

    default:
        showUsage(1);
        return false;
    }

    return true;
}


bool Options::parseArg(int key, uint64_t arg)
{
    switch (key) {
        case 'r': /* --retries */
        if (arg < 1 || arg > 1000) {
            showUsage(1);
            return false;
        }

        m_retries = (int) arg;
        break;

    case 'R': /* --retry-pause */
        if (arg < 1 || arg > 3600) {
            showUsage(1);
            return false;
        }

        m_retryPause = (int) arg;
        break;

    case 't': /* --threads */
        if (arg < 1 || arg > 1024) {
            showUsage(1);
            return false;
        }

        //m_threads = arg;
        break;

    case 1003: /* --donate-level */
        if (arg < 1 || arg > 99) {
            return true;
        }

        m_donateLevel = (int) arg;
        break;

    case 1004: /* --max-gpu-usage */
        if (arg < 1 || arg > 100) {
            showUsage(1);
            return false;
        }

        m_maxGpuUsage = (int) arg;
        break;

    case 1007: /* --print-time */
        if (arg > 1000) {
            showUsage(1);
            return false;
        }

        m_printTime = (int) arg;
        break;

    case 1200: /* --max-gpu-threads */
        m_maxGpuThreads = (int) arg;
        break;

    case 4000: /* --api-port */
        if (arg <= 65536) {
            m_apiPort = (int)arg;
        }
        break;

    default:
        break;
    }

    return true;
}


bool Options::parseBoolean(int key, bool enable)
{
    switch (key) {
    case 'k': /* --keepalive */
        m_pools.back()->setKeepAlive(enable);
        break;

    case 'B': /* --background */
        m_background = enable;
        m_colors = enable ? false : m_colors;
        break;

    case 'S': /* --syslog */
        m_syslog = enable;
        m_colors = enable ? false : m_colors;
        break;

    case 1002: /* --no-color */
        m_colors = enable;
        break;

    case 1006: /* --nicehash */
        m_pools.back()->setNicehash(enable);
        break;

    case 2000: /* colors */
        m_colors = enable;
        break;

    default:
        break;
    }

    return true;
}


Url *Options::parseUrl(const char *arg) const
{
    auto url = new Url(arg);
    if (!url->isValid()) {
        delete url;
        return nullptr;
    }

    return url;
}


void Options::parseConfig(const char *fileName)
{
    rapidjson::Document doc;
    if (!getJSON(fileName, doc)) {
        return;
    }

    m_configName = strdup(fileName);

    for (size_t i = 0; i < ARRAY_SIZE(config_options); i++) {
        parseJSON(&config_options[i], doc);
    }

    const rapidjson::Value &pools = doc["pools"];
    if (pools.IsArray()) {
        for (const rapidjson::Value &value : pools.GetArray()) {
            if (!value.IsObject()) {
                continue;
            }

            for (size_t i = 0; i < ARRAY_SIZE(pool_options); i++) {
                parseJSON(&pool_options[i], value);
            }
        }
    }

    const rapidjson::Value &threads = doc["threads"];
    if (pools.IsArray()) {
        for (const rapidjson::Value &value : threads.GetArray()) {
            if (!value.IsObject()) {
                continue;
            }

            parseThread(value);
        }
    }

    const rapidjson::Value &api = doc["api"];
    if (api.IsObject()) {
        for (size_t i = 0; i < ARRAY_SIZE(api_options); i++) {
            parseJSON(&api_options[i], api);
        }
    }
}


void Options::parseJSON(const struct option *option, const rapidjson::Value &object)
{
    if (!option->name || !object.HasMember(option->name)) {
        return;
    }

    const rapidjson::Value &value = object[option->name];

    if (option->has_arg && value.IsString()) {
        parseArg(option->val, value.GetString());
    }
    else if (option->has_arg && value.IsUint64()) {
        parseArg(option->val, value.GetUint64());
    }
    else if (!option->has_arg && value.IsBool()) {
        parseBoolean(option->val, value.IsTrue());
    }
}


void Options::parseThread(const rapidjson::Value &object)
{
    GpuThread *thread = new GpuThread();
    thread->setIndex(object["index"].GetInt());
    thread->setThreads(object["threads"].GetInt());
    thread->setBlocks(object["blocks"].GetInt());
    thread->setBFactor(object["bfactor"].GetInt());
    thread->setBSleep(object["bsleep"].GetInt());

    const rapidjson::Value &affinity = object["affine_to_cpu"];
    if (affinity.IsInt()) {
        thread->setAffinity(affinity.GetInt());
    }

    if (thread->init()) {
        m_threads.push_back(thread);
        return;
    }

    delete thread;
}



void Options::showUsage(int status) const
{
    if (status) {
        fprintf(stderr, "Try \"" APP_ID "\" --help' for more information.\n");
    }
    else {
        printf(usage);
    }
}


void Options::showVersion()
{
    printf(APP_NAME " " APP_VERSION "\n built on " __DATE__

#   if defined(__clang__)
    " with clang " __clang_version__);
#   elif defined(__GNUC__)
    " with GCC");
    printf(" %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
    " with MSVC");
    printf(" %d", MSVC_VERSION);
#   else
    );
#   endif

    printf("\n features:"
#   if defined(__i386__) || defined(_M_IX86)
    " i386"
#   elif defined(__x86_64__) || defined(_M_AMD64)
    " x86_64"
#   endif

#   if defined(__AES__) || defined(_MSC_VER)
    " AES-NI"
#   endif
    "\n");

    printf("\nlibuv/%s\n", uv_version_string());

    const int cudaVersion = cuda_get_runtime_version();
    printf("CUDA/%d.%d\n", cudaVersion / 1000, cudaVersion % 100);
}


bool Options::setAlgo(const char *algo)
{
    for (size_t i = 0; i < ARRAY_SIZE(algo_names); i++) {
        if (algo_names[i] && !strcmp(algo, algo_names[i])) {
            m_algo = (int) i;
            break;
        }

#       ifndef XMRIG_NO_AEON
        if (i == ARRAY_SIZE(algo_names) - 1 && !strcmp(algo, "cryptonight-light")) {
            m_algo = ALGO_CRYPTONIGHT_LITE;
            break;
        }
#       endif

        if (i == ARRAY_SIZE(algo_names) - 1) {
            showUsage(1);
            return false;
        }
    }

    return true;
}
