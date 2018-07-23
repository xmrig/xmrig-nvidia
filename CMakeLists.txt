cmake_minimum_required(VERSION 2.8)
project(xmrig-nvidia)

option(WITH_AEON     "CryptoNight-Lite support" ON)
option(WITH_SUMO     "CryptoNight-Heavy support" ON)
option(WITH_HTTPD    "HTTP REST API" ON)
option(BUILD_STATIC  "Build static binary" OFF)

include (CheckIncludeFile)
include (cmake/cpu.cmake)


set(HEADERS
    src/api/NetworkState.h
    src/App.h
    src/common/config/CommonConfig.h
    src/common/config/ConfigLoader.h
    src/common/config/ConfigWatcher.h
    src/common/Console.h
    src/common/crypto/Algorithm.h
    src/common/crypto/keccak.h
    src/common/interfaces/IClientListener.h
    src/common/interfaces/IConfig.h
    src/common/interfaces/IConfigCreator.h
    src/common/interfaces/IConsoleListener.h
    src/common/interfaces/IControllerListener.h
    src/common/interfaces/ILogBackend.h
    src/common/interfaces/IStrategy.h
    src/common/interfaces/IStrategyListener.h
    src/common/interfaces/IWatcherListener.h
    src/common/log/BasicLog.h
    src/common/log/ConsoleLog.h
    src/common/log/FileLog.h
    src/common/log/Log.h
    src/common/net/Client.h
    src/common/net/Id.h
    src/common/net/Job.h
    src/common/net/Pool.h
    src/common/net/Storage.h
    src/common/net/strategies/FailoverStrategy.h
    src/common/net/strategies/SinglePoolStrategy.h
    src/common/net/SubmitResult.h
    src/common/Platform.h
    src/common/utils/c_str.h
    src/common/utils/mm_malloc.h
    src/common/utils/timestamp.h
    src/common/xmrig.h
    src/core/ConfigLoader_platform.h
    src/core/Controller.h
    src/Cpu.h
    src/interfaces/IJobResultListener.h
    src/interfaces/IThread.h
    src/interfaces/IWorker.h
    src/net/JobResult.h
    src/net/Network.h
    src/net/strategies/DonateStrategy.h
    src/nvidia/CudaCLI.h
    src/nvidia/Health.h
    src/nvidia/NvmlApi.h
    src/Summary.h
    src/version.h
    src/workers/CudaWorker.h
    src/workers/CudaThread.h
    src/workers/Handle.h
    src/workers/Hashrate.h
    src/workers/Workers.h
   )

set(HEADERS_CRYPTO
    src/crypto/c_blake256.h
    src/crypto/c_groestl.h
    src/crypto/c_jh.h
    src/crypto/c_skein.h
    src/crypto/CryptoNight.h
    src/crypto/CryptoNight_constants.h
    src/crypto/CryptoNight_monero.h
    src/crypto/CryptoNight_test.h
    src/crypto/groestl_tables.h
    src/crypto/hash.h
    src/crypto/skein_port.h
    src/crypto/soft_aes.h
   )

if (XMRIG_ARM)
    set(HEADERS_CRYPTO "${HEADERS_CRYPTO}" src/crypto/CryptoNight_arm.h)
else()
    set(HEADERS_CRYPTO "${HEADERS_CRYPTO}" src/crypto/CryptoNight_x86.h)
endif()

set(SOURCES
    src/api/NetworkState.cpp
    src/App.cpp
    src/common/config/CommonConfig.cpp
    src/common/config/ConfigLoader.cpp
    src/common/config/ConfigWatcher.cpp
    src/common/Console.cpp
    src/common/crypto/Algorithm.cpp
    src/common/crypto/keccak.cpp
    src/common/log/BasicLog.cpp
    src/common/log/ConsoleLog.cpp
    src/common/log/FileLog.cpp
    src/common/log/Log.cpp
    src/common/net/Client.cpp
    src/common/net/Job.cpp
    src/common/net/Pool.cpp
    src/common/net/strategies/FailoverStrategy.cpp
    src/common/net/strategies/SinglePoolStrategy.cpp
    src/common/net/SubmitResult.cpp
    src/common/Platform.cpp
    src/core/Config.cpp
    src/core/Controller.cpp
    src/net/Network.cpp
    src/net/strategies/DonateStrategy.cpp
    src/nvidia/CudaCLI.cpp
    src/Summary.cpp
    src/workers/CudaWorker.cpp
    src/workers/CudaThread.cpp
    src/workers/Handle.cpp
    src/workers/Hashrate.cpp
    src/workers/Workers.cpp
    src/xmrig.cpp
   )

set(SOURCES_CRYPTO
    src/crypto/c_groestl.c
    src/crypto/c_blake256.c
    src/crypto/c_jh.c
    src/crypto/c_skein.c
    src/crypto/CryptoNight.cpp
   )

if (WIN32)
    set(SOURCES_OS
        res/app.rc
        src/App_win.cpp
        src/common/Platform_win.cpp
        src/Cpu_win.cpp
        )

    add_definitions(/DWIN32)
    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv)
elseif (APPLE)
    set(SOURCES_OS
        src/App_unix.cpp
        src/common/Platform_mac.cpp
        src/Cpu_mac.cpp
        )
else()
    set(SOURCES_OS
        src/App_unix.cpp
        src/common/Platform_unix.cpp
        src/Cpu_unix.cpp
        )

    set(EXTRA_LIBS pthread rt dl)
endif()

add_definitions(/D__STDC_FORMAT_MACROS)
add_definitions(/DUNICODE)
#add_definitions(/DAPP_DEBUG)
add_definitions(/DXMRIG_NVIDIA_PROJECT)
add_definitions(/DXMRIG_NO_LIBCPUID)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(UV REQUIRED)

if (NOT WITH_AEON)
    add_definitions(/DXMRIG_NO_AEON)
endif()

if (NOT WITH_SUMO)
    add_definitions(/DXMRIG_NO_SUMO)
endif()

include_directories(src)

include(cmake/flags.cmake)
include(cmake/CUDA.cmake)

find_package(NVML)
if (NVML_FOUND)
    include_directories(${NVML_INCLUDE_DIR})

    set(SOURCES_NVML src/nvidia/NvmlApi.cpp)
else()
    add_definitions(/DXMRIG_NO_NVML)

    set(SOURCES_NVML src/nvidia/NvmlApi_stub.cpp)
endif()

if (XMRIG_ARM)
    set(SOURCES_CPUID src/Cpu_arm.cpp)
else()
    set(SOURCES_CPUID src/Cpu_stub.cpp)
endif()

CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
if (HAVE_SYSLOG_H)
    add_definitions(/DHAVE_SYSLOG_H)
    set(SOURCES_SYSLOG src/common/log/SysLog.h src/common/log/SysLog.cpp)
endif()

if (WITH_HTTPD)
    find_package(MHD)

    if (MHD_FOUND)
        include_directories(${MHD_INCLUDE_DIRS})
        set(HTTPD_SOURCES
            src/api/Api.h
            src/api/ApiRouter.h
            src/common/api/HttpBody.h
            src/common/api/Httpd.h
            src/common/api/HttpReply.h
            src/common/api/HttpRequest.h
            src/api/Api.cpp
            src/api/ApiRouter.cpp
            src/common/api/Httpd.cpp
            src/common/api/HttpRequest.cpp
            )
    else()
        message(FATAL_ERROR "microhttpd NOT found: use `-DWITH_HTTPD=OFF` to build without http deamon support")
    endif()
else()
    set(MHD_LIBRARY "")
    add_definitions(/DXMRIG_NO_HTTPD)
    add_definitions(/DXMRIG_NO_API)
endif()

include_directories(src/3rdparty)
include_directories(${UV_INCLUDE_DIR})

if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS " -static")
endif()

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} ${SOURCES_OS} ${SOURCES_CPUID} ${SOURCES_NVML} ${HEADERS_CRYPTO} ${SOURCES_CRYPTO} ${SOURCES_SYSLOG} ${HTTPD_SOURCES})
target_link_libraries(${PROJECT_NAME} xmrig-cuda ${UV_LIBRARIES} ${MHD_LIBRARY} ${LIBS} ${EXTRA_LIBS} ${CPUID_LIB})
