/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
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


#include <nvml.h>
#include <uv.h>


#include "nvidia/NvmlApi.h"
#include "workers/CudaThread.h"


static uv_lib_t nvmlLib;
static char nvmlVersion[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE] = { 0 };


bool NvmlApi::m_available = false;

static nvmlReturn_t(*pNvmlInit)(void) = nullptr;
static nvmlReturn_t(*pNvmlShutdown)(void) = nullptr;
static nvmlReturn_t(*pNvmlDeviceGetHandleByIndex)(unsigned int index, nvmlDevice_t *device) = nullptr;
static nvmlReturn_t(*pNvmlDeviceGetTemperature)(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp) = nullptr;
static nvmlReturn_t(*pNvmlDeviceGetPowerUsage)(nvmlDevice_t device, unsigned int* power) = nullptr;
static nvmlReturn_t(*pNvmlDeviceGetFanSpeed)(nvmlDevice_t device, unsigned int* speed) = nullptr;
static nvmlReturn_t(*pNvmlDeviceGetClockInfo)(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) = nullptr;
static nvmlReturn_t(*pNvmlSystemGetNVMLVersion)(char *version, unsigned int length) = nullptr;
static nvmlReturn_t(*pNvmlDeviceGetCount)(unsigned int *deviceCount) = nullptr;
static nvmlReturn_t(*pNvmlDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t *pci) = nullptr;


bool NvmlApi::init()
{
#   ifdef _WIN32
    char tmp[261]; //LoadLibrary calls are still "260 char" limited
    ExpandEnvironmentStringsA(R"(%ProgramFiles%\NVIDIA Corporation\NVSMI\nvml.dll)", tmp, sizeof(tmp));
    if (uv_dlopen(tmp, &nvmlLib) == -1 && uv_dlopen("nvml.dll", &nvmlLib) == -1) {
        return false;
    }
#   else
    if (uv_dlopen("libnvidia-ml.so", &nvmlLib) == -1) {
        return false;
    }
#   endif

    if (uv_dlsym(&nvmlLib, "nvmlInit_v2", reinterpret_cast<void**>(&pNvmlInit)) == -1) {
        return false;
    }

    uv_dlsym(&nvmlLib, "nvmlShutdown", reinterpret_cast<void**>(&pNvmlShutdown));
    uv_dlsym(&nvmlLib, "nvmlDeviceGetHandleByIndex_v2", reinterpret_cast<void**>(&pNvmlDeviceGetHandleByIndex));
    uv_dlsym(&nvmlLib, "nvmlDeviceGetTemperature", reinterpret_cast<void**>(&pNvmlDeviceGetTemperature));
    uv_dlsym(&nvmlLib, "nvmlDeviceGetPowerUsage", reinterpret_cast<void**>(&pNvmlDeviceGetPowerUsage));
    uv_dlsym(&nvmlLib, "nvmlDeviceGetFanSpeed", reinterpret_cast<void**>(&pNvmlDeviceGetFanSpeed));
    uv_dlsym(&nvmlLib, "nvmlDeviceGetClockInfo", reinterpret_cast<void**>(&pNvmlDeviceGetClockInfo));
    uv_dlsym(&nvmlLib, "nvmlSystemGetNVMLVersion", reinterpret_cast<void**>(&pNvmlSystemGetNVMLVersion));
    uv_dlsym(&nvmlLib, "nvmlDeviceGetCount_v2", reinterpret_cast<void**>(&pNvmlDeviceGetCount));
    uv_dlsym(&nvmlLib, "nvmlDeviceGetPciInfo_v2", reinterpret_cast<void**>(&pNvmlDeviceGetPciInfo));

    m_available = pNvmlInit() == NVML_SUCCESS;

    if (pNvmlSystemGetNVMLVersion) {
        pNvmlSystemGetNVMLVersion(nvmlVersion, sizeof(nvmlVersion));
    }

    return m_available;
}


void NvmlApi::release()
{
    if (!isAvailable() && !pNvmlShutdown) {
        return;
    }

    pNvmlShutdown();
}


bool NvmlApi::health(int i, Health &health)
{
    const auto id = static_cast<unsigned int>(i);
    nvmlDevice_t device;

    if (i == -1 || !isAvailable()
        ||
        (pNvmlDeviceGetHandleByIndex && pNvmlDeviceGetHandleByIndex(id, &device) != NVML_SUCCESS)
    ) {
        return false;
    }

    // cache items previously pegged as unavailable via function call failure
    // this has to happen before the reset or we don't see the previous value
    const bool hasPowerUsage = PROBED_UNSUPPORTED != health.power;
    const bool hasFanSpeed   = PROBED_UNSUPPORTED != health.fanSpeed;
    const bool hasClockInfo  = PROBED_UNSUPPORTED != health.clock;

    health.reset();

    if (pNvmlDeviceGetTemperature) {
        pNvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &health.temperature);
    }

    if (pNvmlDeviceGetPowerUsage) {
        if (!hasPowerUsage || pNvmlDeviceGetPowerUsage(device, &health.power) != NVML_SUCCESS){
            health.power = PROBED_UNSUPPORTED;
        }
    }

    if (pNvmlDeviceGetFanSpeed) {
        if (!hasFanSpeed || pNvmlDeviceGetFanSpeed(device, &health.fanSpeed) != NVML_SUCCESS){
            health.fanSpeed = PROBED_UNSUPPORTED;
        }
    }

    if (pNvmlDeviceGetClockInfo) {
        if (!hasClockInfo
            ||
            pNvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &health.clock) != NVML_SUCCESS
            ||
            pNvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &health.memClock) != NVML_SUCCESS
        ) {
            health.clock = PROBED_UNSUPPORTED;
            health.memClock = PROBED_UNSUPPORTED;
        }
    }

    return true;
}


const char *NvmlApi::version()
{
    return nvmlVersion;
}


void NvmlApi::bind(const std::vector<xmrig::IThread*> &threads)
{
    if (!isAvailable() || !pNvmlDeviceGetCount || !pNvmlDeviceGetHandleByIndex || !pNvmlDeviceGetPciInfo) {
        return;
    }

    uint32_t count = 0;
    if (pNvmlDeviceGetCount(&count) != NVML_SUCCESS) {
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        nvmlDevice_t device;
        if (pNvmlDeviceGetHandleByIndex(i, &device) != NVML_SUCCESS) {
            continue;
        }

        nvmlPciInfo_t pci;
        if (pNvmlDeviceGetPciInfo(device, &pci) != NVML_SUCCESS) {
            continue;
        }

        for (xmrig::IThread *t : threads) {
            auto thread = dynamic_cast<CudaThread *>(t);
            if (thread->pciBusID() == pci.bus && thread->pciDeviceID() == pci.device && thread->pciDomainID() == pci.domain) {
                thread->setNvmlId(i);
                break;
            }
        }
    }
}
