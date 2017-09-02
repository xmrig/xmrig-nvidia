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


#include <inttypes.h>
#include <uv.h>


#include "Cpu.h"
#include "log/Log.h"
#include "net/Url.h"
#include "nvidia/cryptonight.h"
#include "Options.h"
#include "Summary.h"
#include "version.h"
#include "workers/GpuThread.h"


static void print_versions()
{
    char buf[16];

#   if defined(__clang__)
    snprintf(buf, 16, " clang/%d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#   elif defined(__GNUC__)
    snprintf(buf, 16, " gcc/%d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#   elif defined(_MSC_VER)
    snprintf(buf, 16, " MSVC/%d", MSVC_VERSION);
#   else
    buf[0] = '\0';
#   endif

    const int cudaVersion = cuda_get_runtime_version();
    Log::i()->text(Options::i()->colors() ? "\x1B[01;32m * \x1B[01;37mVERSIONS:     \x1B[01;36mXMRig/%s\x1B[01;37m libuv/%s CUDA/%d.%d%s" : " * VERSIONS:     XMRig/%s libuv/%s CUDA/%d.%d%s",
                   APP_VERSION, uv_version_string(), cudaVersion / 1000, cudaVersion % 100, buf);
}


static void print_cpu()
{
    if (Options::i()->colors()) {
        Log::i()->text("\x1B[01;32m * \x1B[01;37mCPU:          %s %sx64 %sAES-NI",
                       Cpu::brand(),
                       Cpu::isX64() ? "\x1B[01;32m" : "\x1B[01;31m-",
                       Cpu::hasAES() ? "\x1B[01;32m" : "\x1B[01;31m-");
    }
    else {
        Log::i()->text(" * CPU:          %s (%d) %sx64 %sAES-NI", Cpu::brand(), Cpu::sockets(), Cpu::isX64() ? "" : "-", Cpu::hasAES() ? "" : "-");
    }
}


static void print_algo()
{
    Log::i()->text(Options::i()->colors() ? "\x1B[01;32m * \x1B[01;37mALGO:         %s, %sdonate=%d%%" : " * ALGO:         %s, %sdonate=%d%%",
                   Options::i()->algoName(),
                   Options::i()->colors() && Options::i()->donateLevel() == 0 ? "\x1B[01;31m" : "",
                   Options::i()->donateLevel()
    );
}


static void print_pools()
{
    const std::vector<Url*> &pools = Options::i()->pools();

    for (size_t i = 0; i < pools.size(); ++i) {
        Log::i()->text(Options::i()->colors() ? "\x1B[01;32m * \x1B[01;37mPOOL #%d:      \x1B[01;36m%s:%d" : " * POOL #%d:      %s:%d",
                       i + 1,
                       pools[i]->host(),
                       pools[i]->port());
    }

#   ifdef APP_DEBUG
    for (size_t i = 0; i < pools.size(); ++i) {
        Log::i()->text("%s:%d, user: %s, pass: %s, ka: %d, nicehash: %d", pools[i]->host(), pools[i]->port(), pools[i]->user(), pools[i]->password(), pools[i]->isKeepAlive(), pools[i]->isNicehash());
    }
#   endif
}


static void print_gpu()
{
    for (const GpuThread *thread : Options::i()->threads()) {
        Log::i()->text(Options::i()->colors() ? "\x1B[01;32m * \x1B[01;37mGPU #%d:       \x1B[22;32m%s @ %d/%d MHz \x1B[01;30m%dx%d %dx%d arch:%d%d SMX:%d" : " * GPU #%d:       %s @ %d/%d MHz %dx%d %dx%d arch:%d%d SMX:%d",
            thread->index(),
            thread->name(),
            thread->clockRate() / 1000,
            thread->memoryClockRate() / 1000,
            thread->threads(),
            thread->blocks(),
            thread->bfactor(),
            thread->bsleep(),
            thread->arch()[0],
            thread->arch()[1],
            thread->smx()
        );
    }
}


#ifndef XMRIG_NO_API
static void print_api()
{
    if (Options::i()->apiPort() == 0) {
        return;
    }

    Log::i()->text(Options::i()->colors() ? "\x1B[01;32m * \x1B[01;37mAPI PORT:     \x1B[01;36m%d" : " * API PORT:     %d", Options::i()->apiPort());
}
#endif


static void print_commands()
{
    if (Options::i()->colors()) {
        Log::i()->text("\x1B[01;32m * \x1B[01;37mCOMMANDS:     \x1B[01;35mh\x1B[01;37mashrate, h\x1B[01;35me\x1B[01;37malth, \x1B[01;35mp\x1B[01;37mause, \x1B[01;35mr\x1B[01;37mesume");
    }
    else {
        Log::i()->text(" * COMMANDS:     'h' hashrate, 'e' health, 'p' pause, 'r' resume");
    }
}


static bool print_extra()
{
    const std::vector<GpuThread*> &threads = Options::i()->threads();
    if (threads.empty()) {
        LOG_ERR("No CUDA device found!");
        return false;
    }

    if (!Options::i()->isAutoConf()) {
        return true;
    }

    if (Options::i()->save()) {
        Log::i()->text("Initial configuration saved to: %s", Options::i()->configName());
    }

    return true;
}


bool Summary::print()
{
    print_versions();
    print_cpu();
    print_gpu();
    print_algo();
    print_pools();

#   ifndef XMRIG_NO_API
    print_api();
#   endif

    print_commands();
    
    return print_extra();
}



