/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      SChernykh   <https://github.com/SChernykh>
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



#include <stdlib.h>
#include <uv.h>
#include <cuda.h>


#include "api/Api.h"
#include "App.h"
#include "base/kernel/Signals.h"
#include "common/Console.h"
#include "common/log/Log.h"
#include "common/Platform.h"
#include "core/Config.h"
#include "core/Controller.h"
#include "crypto/CryptoNight.h"
#include "Mem.h"
#include "net/Network.h"
#include "Summary.h"
#include "version.h"
#include "workers/Workers.h"


#ifndef XMRIG_NO_HTTPD
#   include "common/api/Httpd.h"
#endif


xmrig::App::App(Process *process) :
    m_console(nullptr),
    m_httpd(nullptr),
    m_signals(nullptr)
{
    m_controller = new xmrig::Controller(process);
    if (m_controller->init() != 0) {
        return;
    }

    if (!m_controller->config()->isBackground()) {
        m_console = new Console(this);
    }

    cuInit(0);
}


xmrig::App::~App()
{
    uv_tty_reset_mode();

    delete m_signals;
    delete m_console;
    delete m_controller;

#   ifndef XMRIG_NO_HTTPD
    delete m_httpd;
#   endif
}


int xmrig::App::exec()
{
    if (!m_controller->isReady()) {
        return 2;
    }

    m_signals = new Signals(this);

    background();

    if (!CryptoNight::init(m_controller->config()->algorithm().algo())) {
        LOG_ERR(RED_BOLD("\"%s\" hash self-test failed."), m_controller->config()->algorithm().name());
        return 1;
    }

    Summary::print(m_controller);

    if (m_controller->config()->threads().empty()) {
        LOG_ERR(RED_BOLD("No CUDA device found!"));
        return 1;
    }

    if (m_controller->config()->isDryRun()) {
        LOG_NOTICE(GREEN_BOLD("OK"));
        return 0;
    }

#   ifndef XMRIG_NO_API
    Api::start(m_controller);
#   endif

#   ifndef XMRIG_NO_HTTPD
    m_httpd = new Httpd(
                m_controller->config()->apiPort(),
                m_controller->config()->apiToken(),
                m_controller->config()->isApiIPv6(),
                m_controller->config()->isApiRestricted()
                );

    m_httpd->start();
#   endif

    if (!Workers::start(m_controller)) {
        LOG_ERR("Failed to start threads");
        return 1;
    }

    m_controller->network()->connect();

    const int r = uv_run(uv_default_loop(), UV_RUN_DEFAULT);
    uv_loop_close(uv_default_loop());

    return r;
}


void xmrig::App::onConsoleCommand(char command)
{
    switch (command) {
    case 'h':
    case 'H':
        Workers::printHashrate(true);
        break;

    case 'p':
    case 'P':
        if (Workers::isEnabled()) {
            LOG_INFO(YELLOW("paused") ", press '" MAGENTA_BOLD("r") "' to resume");
            Workers::setEnabled(false);
        }
        break;

    case 'r':
    case 'R':
        if (!Workers::isEnabled()) {
            LOG_INFO(GREEN_BOLD("resumed"));
            Workers::setEnabled(true);
        }
        break;

    case 'E':
    case 'e':
        Workers::printHealth();
        break;

    case 3:
        LOG_WARN("Ctrl+C received, exiting" CLEAR);
        close();
        break;

    default:
        break;
    }
}


void xmrig::App::onSignal(int signum)
{
    switch (signum)
    {
    case SIGHUP:
        LOG_WARN("SIGHUP received, exiting" CLEAR);
        break;

    case SIGTERM:
        LOG_WARN("SIGTERM received, exiting" CLEAR);
        break;

    case SIGINT:
        LOG_WARN("SIGINT received, exiting" CLEAR);
        break;

    default:
        break;
    }

    close();
}


void xmrig::App::close()
{
    m_controller->network()->stop();
    Workers::stop();

    uv_stop(uv_default_loop());
}
