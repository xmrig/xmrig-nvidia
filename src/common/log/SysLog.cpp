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


#include <syslog.h>


#include "common/log/SysLog.h"
#include "common/log/Log.h"
#include "version.h"


SysLog::SysLog()
{
    openlog(APP_ID, LOG_PID, LOG_USER);
}


void SysLog::stripColor()
{
    // strip ANSI CSI sequences
    std::string txt(m_fmt);
    std::string::size_type i, j;
    while ((i = txt.find(CSI)) != std::string::npos) {
        j = txt.find('m', i);
        txt.erase(i, j-i+1);
    }
    snprintf(m_fmt, sizeof(m_fmt) - 1, "%s", txt.c_str());
}


void SysLog::message(Level level, const char *fmt, va_list args)
{
    char buf[kBufferSize];
    vsnprintf(buf, sizeof(buf) - 1, m_fmt, args);
    snprintf(m_fmt, sizeof(m_fmt) - 1, "%s", buf);
    SysLog::stripColor();
    vsyslog(static_cast<int>(level), m_fmt, args);
}


void SysLog::text(const char *fmt, va_list args)
{
    vsyslog(LOG_INFO, fmt, args);
}
