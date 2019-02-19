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

#ifndef XMRIG_CONFIG_H
#define XMRIG_CONFIG_H


#include <stdint.h>
#include <vector>


#include "common/config/CommonConfig.h"
#include "common/xmrig.h"
#include "nvidia/CudaCLI.h"
#include "rapidjson/fwd.h"


namespace xmrig {


class ConfigLoader;
class IThread;
class IConfigListener;
class Process;


class Config : public CommonConfig
{
public:
    Config();

    bool isCNv2() const;
    bool reload(const char *json);
    void getJSON(rapidjson::Document &doc) const override;

    inline bool isShouldSave() const                     { return m_shouldSave; }
    inline const std::vector<IThread *> &threads() const { return m_threads; }
    inline int maxGpuThreads() const                     { return m_maxGpuThreads; }

    static Config *load(Process *process, IConfigListener *listener);

protected:
    bool finalize() override;
    bool parseString(int key, const char *arg) override;
    bool parseUint64(int key, uint64_t arg) override;
    void parseJSON(const rapidjson::Document &doc) override;

private:
    void parseThread(const rapidjson::Value &object);

    bool m_autoConf;
    bool m_shouldSave;
    CudaCLI m_cudaCLI;
    int m_maxGpuThreads;
    int m_maxGpuUsage;
    std::vector<IThread *> m_threads;
};


} /* namespace xmrig */

#endif /* XMRIG_CONFIG_H */
