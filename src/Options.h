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

#ifndef __OPTIONS_H__
#define __OPTIONS_H__


#include <stdint.h>
#include <vector>


#include "nvidia/CudaCLI.h"
#include "rapidjson/fwd.h"


class GpuThread;
class Url;
struct option;


class Options
{
public:
    enum Algo {
        ALGO_CRYPTONIGHT,      /* CryptoNight (Monero) */
        ALGO_CRYPTONIGHT_LITE, /* CryptoNight-Lite (AEON) */
    };

    enum AlgoVariant {
        AV0_AUTO,
        AV1_AESNI,
        AV2_UNUSED,
        AV3_SOFT_AES,
        AV4_UNUSED,
        AV_MAX
    };

    static inline Options* i() { return m_self; }
    static Options *parse(int argc, char **argv);

    inline bool background() const                        { return m_background; }
    inline bool colors() const                            { return m_colors; }
    inline bool isAutoConf() const                        { return m_autoConf; }
    inline bool syslog() const                            { return m_syslog; }
    inline const char *apiToken() const                   { return m_apiToken; }
    inline const char *apiWorkerId() const                { return m_apiWorkerId; }
    inline const char *configName() const                 { return m_configName; }
    inline const char *logFile() const                    { return m_logFile; }
    inline const char *userAgent() const                  { return m_userAgent; }
    inline const std::vector<GpuThread*> &threads() const { return m_threads; }
    inline const std::vector<Url*> &pools() const         { return m_pools; }
    inline int algo() const                               { return m_algo; }
    inline int algoVariant() const                        { return m_algoVariant; }
    inline int apiPort() const                            { return m_apiPort; }
    inline int donateLevel() const                        { return m_donateLevel; }
    inline int maxGpuThreads() const                      { return m_maxGpuThreads; }
    inline int printTime() const                          { return m_printTime; }
    inline int retries() const                            { return m_retries; }
    inline int retryPause() const                         { return m_retryPause; }

    inline static void release()                          { delete m_self; }

    bool save();
    const char *algoName() const;

private:
    Options(int argc, char **argv);
    ~Options();

    inline bool isReady() const { return m_ready; }

    static Options *m_self;

    bool getJSON(const char *fileName, rapidjson::Document &doc);
    bool parseArg(int key, const char *arg);
    bool parseArg(int key, uint64_t arg);
    bool parseBoolean(int key, bool enable);
    Url *parseUrl(const char *arg) const;
    void parseConfig(const char *fileName);
    void parseJSON(const struct option *option, const rapidjson::Value &object);
    void parseThread(const rapidjson::Value &object);
    void showUsage(int status) const;
    void showVersion(void);

    bool setAlgo(const char *algo);

    bool m_autoConf;
    bool m_background;
    bool m_colors;
    bool m_ready;
    bool m_syslog;
    char *m_apiToken;
    char *m_apiWorkerId;
    char *m_configName;
    char *m_logFile;
    char *m_userAgent;
    CudaCLI m_cudaCLI;
    int m_algo;
    int m_algoVariant;
    int m_apiPort;
    int m_donateLevel;
    int m_maxGpuThreads;
    int m_maxGpuUsage;
    int m_printTime;
    int m_retries;
    int m_retryPause;
    std::vector<GpuThread*> m_threads;
    std::vector<Url*> m_pools;
};

#endif /* __OPTIONS_H__ */
