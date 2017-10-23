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

#ifndef __CUDACLI_H__
#define __CUDACLI_H__


#include <vector>


class GpuThread;


class CudaCLI
{
public:
    CudaCLI();

    bool setup(std::vector<GpuThread*> &threads);
    void autoConf(std::vector<GpuThread*> &threads);
    void parseDevices(const char *arg);
    void parseLaunch(const char *arg);

    inline void parseAffinity(const char *arg) { parse(m_affinity, arg); }
    inline void parseBFactor(const char *arg)  { parse(m_bfactors, arg); }
    inline void parseBSleep(const char *arg)   { parse(m_bsleeps, arg); }

private:
    inline int affinity(int index) const { return get(m_affinity, index, -1); }
    inline int blocks(int index) const   { return get(m_blocks, index, -1); }
    inline int threads(int index) const  { return get(m_threads, index, -1); }
    inline bool isEmpty() const          { return m_devices.empty() && m_threads.empty(); };

    inline int bfactor(int index = 0) const {
#       ifdef _WIN32
        return get(m_bfactors, index, 6);
#       else
        return get(m_bfactors, index, 0);
#       endif
    }

    inline int bsleep(int index = 0) const {
#       ifdef _WIN32
        return get(m_bsleeps, index, 25);
#       else
        return get(m_bsleeps, index, 0);
#       endif
    }

    int get(const std::vector<int> &vector, int index, int defaultValue) const;
    void parse(std::vector<int> &vector, const char *arg) const;

    const int m_count;
    std::vector<int> m_affinity;
    std::vector<int> m_bfactors;
    std::vector<int> m_blocks;
    std::vector<int> m_bsleeps;
    std::vector<int> m_devices;
    std::vector<int> m_threads;
};


#endif /* __CUDACLI_H__ */
