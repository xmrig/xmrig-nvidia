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

#ifndef XMRIG_ICONFIG_H
#define XMRIG_ICONFIG_H


#include "common/crypto/Algorithm.h"
#include "rapidjson/fwd.h"


namespace xmrig {


class String;


class IConfig
{
public:
    enum Keys {
        // common
        AlgorithmKey      = 'a',
        ApiAccessTokenKey = 4001,
        ApiIPv6Key        = 4003,
        ApiPort           = 4000,
        ApiRestrictedKey  = 4004,
        ApiWorkerIdKey    = 4002,
        ApiIdKey          = 4005,
        BackgroundKey     = 'B',
        ColorKey          = 1002,
        ConfigKey         = 'c',
        DonateLevelKey    = 1003,
        KeepAliveKey      = 'k',
        LogFileKey        = 'l',
        PasswordKey       = 'p',
        RetriesKey        = 'r',
        RetryPauseKey     = 'R',
        RigIdKey          = 1012,
        SyslogKey         = 'S',
        UrlKey            = 'o',
        UserAgentKey      = 1008,
        UserKey           = 'u',
        UserpassKey       = 'O',
        VariantKey        = 1010,
        VerboseKey        = 1100,
        WatchKey          = 1105,
        TlsKey            = 1013,
        FingerprintKey    = 1014,
        AutoSaveKey       = 1016,

        // xmrig common
        CPUPriorityKey    = 1021,
        NicehashKey       = 1006,
        PrintTimeKey      = 1007,

        // xmrig cpu
        AVKey             = 'v',
        CPUAffinityKey    = 1020,
        DryRunKey         = 5000,
        HugePagesKey      = 1009,
        MaxCPUUsageKey    = 1004,
        SafeKey           = 1005,
        ThreadsKey        = 't',
        HardwareAESKey    = 1011,
        AssemblyKey       = 1015,

        // xmrig amd
        OclPlatformKey    = 1400,
        OclAffinityKey    = 1401,
        OclDevicesKey     = 1402,
        OclLaunchKey      = 1403,
        OclCacheKey       = 1404,
        OclPrintKey       = 1405,
        OclLoaderKey      = 1406,
        OclSridedIndexKey = 1407,
        OclMemChunkKey    = 1408,
        OclUnrollKey      = 1409,
        OclCompModeKey    = 1410,

        // xmrig-proxy
        AccessLogFileKey   = 'A',
        BindKey            = 'b',
        CoinKey            = 1104,
        CustomDiffKey      = 1102,
        DebugKey           = 1101,
        ModeKey            = 'm',
        PoolCoinKey        = 'C',
        ReuseTimeoutKey    = 1106,
        WorkersKey         = 1103,
        WorkersAdvKey      = 1107,
        TlsBindKey         = 1108,
        TlsCertKey         = 1109,
        TlsCertKeyKey      = 1110,
        TlsDHparamKey      = 1111,
        TlsCiphersKey      = 1112,
        TlsCipherSuitesKey = 1113,
        TlsProtocolsKey    = 1114,

        // xmrig nvidia
        CudaMaxThreadsKey = 1200,
        CudaBFactorKey    = 1201,
        CudaBSleepKey     = 1202,
        CudaDevicesKey    = 1203,
        CudaLaunchKey     = 1204,
        CudaAffinityKey   = 1205,
        CudaMaxUsageKey   = 1206,
        NvmlTempL         = 1207,
        NvmlTempH         = 1208,
    };

    virtual ~IConfig() = default;

    virtual bool finalize()                                = 0;
    virtual bool isWatch() const                           = 0;
    virtual bool parseBoolean(int key, bool enable)        = 0;
    virtual bool parseString(int key, const char *arg)     = 0;
    virtual bool parseUint64(int key, uint64_t arg)        = 0;
    virtual bool save()                                    = 0;
    virtual const Algorithm &algorithm() const             = 0;
    virtual const String &fileName() const                 = 0;
    virtual void getJSON(rapidjson::Document &doc) const   = 0;
    virtual void parseJSON(const rapidjson::Document &doc) = 0;
    virtual void setFileName(const char *fileName)         = 0;
};


} /* namespace xmrig */


#endif // XMRIG_ICONFIG_H
