/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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


#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include "common/interfaces/ILogBackend.h"
#include "common/log/BasicLog.h"
#include "common/log/Log.h"


Log *Log::m_self = nullptr;
bool Log::colors = true;


static const char *color[5] = {
    RED_S,         /* ERR     */
    YELLOW_S,      /* WARNING */
    WHITE_BOLD_S,  /* NOTICE  */
    "",            /* INFO    */
#   ifdef WIN32
    BLACK_BOLD_S   /* DEBUG   */
#   else
    BRIGHT_BLACK_S /* DEBUG   */
#   endif
};


void Log::message(ILogBackend::Level level, const char* fmt, ...)
{
    uv_mutex_lock(&m_mutex);

    va_list args;
    va_list copy;
    va_start(args, fmt);

    for (ILogBackend *backend : m_backends) {
        va_copy(copy, args);
        backend->message(level, fmt, copy);
        va_end(copy);
    }

    va_end(args);

    uv_mutex_unlock(&m_mutex);
}


void Log::text(const char* fmt, ...)
{
    uv_mutex_lock(&m_mutex);

    va_list args;
    va_list copy;
    va_start(args, fmt);

    for (ILogBackend *backend : m_backends) {
        va_copy(copy, args);
        backend->text(fmt, copy);
        va_end(copy);
    }

    va_end(args);

    uv_mutex_unlock(&m_mutex);
}


const char *Log::colorByLevel(ILogBackend::Level level)
{
    return color[level];
}


const char *Log::endl()
{
    return CLEAR ENDL;
}


void Log::defaultInit()
{
    m_self = new Log();

    add(new BasicLog());
}


Log::~Log()
{
    for (auto backend : m_backends) {
        delete backend;
    }
}
