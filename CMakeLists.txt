cmake_minimum_required(VERSION 2.8)
project(xmrig-nvidia)

option(WITH_AEON  "CryptoNight-Lite support" ON)
option(WITH_HTTPD "HTTP REST API" ON)

include (CheckIncludeFile)


set(HEADERS
    src/api/Api.h
    src/api/ApiState.h
    src/api/NetworkState.h
    src/App.h
    src/Console.h
    src/Cpu.h
    src/interfaces/IClientListener.h
    src/interfaces/IConsoleListener.h
    src/interfaces/IJobResultListener.h
    src/interfaces/ILogBackend.h
    src/interfaces/IStrategy.h
    src/interfaces/IStrategyListener.h
    src/interfaces/IWorker.h
    src/log/ConsoleLog.h
    src/log/FileLog.h
    src/log/Log.h
    src/net/Client.h
    src/net/Id.h
    src/net/Job.h
    src/net/JobResult.h
    src/net/Network.h
    src/net/strategies/DonateStrategy.h
    src/net/strategies/FailoverStrategy.h
    src/net/strategies/SinglePoolStrategy.h
    src/net/SubmitResult.h
    src/net/Url.h
    src/nvidia/CudaCLI.h
    src/nvidia/Health.h
    src/nvidia/NvmlApi.h
    src/Options.h
    src/Platform.h
    src/Summary.h
    src/version.h
    src/workers/CudaWorker.h
    src/workers/GpuThread.h
    src/workers/Handle.h
    src/workers/Hashrate.h
    src/workers/Workers.h
    src/xmrig.h
   )

set(HEADERS_CRYPTO
    src/crypto/c_blake256.h
    src/crypto/c_groestl.h
    src/crypto/c_jh.h
    src/crypto/c_keccak.h
    src/crypto/c_skein.h
    src/crypto/CryptoNight.h
    src/crypto/CryptoNight_monero.h
    src/crypto/CryptoNight_test.h
    src/crypto/CryptoNight_x86.h
    src/crypto/groestl_tables.h
    src/crypto/hash.h
    src/crypto/skein_port.h
    src/crypto/soft_aes.h
   )

set(SOURCES
    src/api/Api.cpp
    src/api/ApiState.cpp
    src/api/NetworkState.cpp
    src/App.cpp
    src/Console.cpp
    src/Cpu_stub.cpp
    src/log/ConsoleLog.cpp
    src/log/FileLog.cpp
    src/log/Log.cpp
    src/net/Client.cpp
    src/net/Job.cpp
    src/net/Network.cpp
    src/net/strategies/DonateStrategy.cpp
    src/net/strategies/FailoverStrategy.cpp
    src/net/strategies/SinglePoolStrategy.cpp
    src/net/SubmitResult.cpp
    src/net/Url.cpp
    src/nvidia/CudaCLI.cpp
    src/Options.cpp
    src/Platform.cpp
    src/Summary.cpp
    src/workers/CudaWorker.cpp
    src/workers/GpuThread.cpp
    src/workers/Handle.cpp
    src/workers/Hashrate.cpp
    src/workers/Workers.cpp
    src/xmrig.cpp
   )

set(SOURCES_CRYPTO
    src/crypto/c_keccak.c
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
        src/Cpu_win.cpp
        src/Platform_win.cpp
        )

    add_definitions(/DWIN32)
    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv)
elseif (APPLE)
    set(SOURCES_OS
        src/App_unix.cpp
        src/Cpu_mac.cpp
        src/Platform_mac.cpp
        )
else()
    set(SOURCES_OS
        src/App_unix.cpp
        src/Cpu_unix.cpp
        src/Platform_unix.cpp
        )

    set(EXTRA_LIBS pthread)
endif()

add_definitions(/DXMRIG_NVIDIA_PROJECT)
add_definitions(/DUNICODE)
add_definitions(/DXMRIG_NO_LIBCPUID)
add_definitions(/D__STDC_FORMAT_MACROS)
add_definitions(/DRAPIDJSON_SSE2)
#add_definitions(/DAPP_DEBUG)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(UV REQUIRED)

if (NOT WITH_AEON)
    add_definitions(/DXMRIG_NO_AEON)
endif()

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

CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
if (HAVE_SYSLOG_H)
    add_definitions(/DHAVE_SYSLOG_H)
    set(SOURCES_SYSLOG src/log/SysLog.h src/log/SysLog.cpp)
endif()

if (WITH_HTTPD)
    find_package(MHD)

    if (MHD_FOUND)
        include_directories(${MHD_INCLUDE_DIRS})
        set(HTTPD_SOURCES src/api/Httpd.h src/api/Httpd.cpp)
    else()
        message(FATAL_ERROR "microhttpd NOT found: use `-DWITH_HTTPD=OFF` to build without http deamon support")
    endif()
else()
    add_definitions(/DXMRIG_NO_HTTPD)
    add_definitions(/DXMRIG_NO_API)

    set(MHD_LIBRARY "")
endif()

include_directories(src)
include_directories(src/3rdparty)
include_directories(${UV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} ${SOURCES_OS} ${SOURCES_NVML} ${HEADERS_CRYPTO} ${SOURCES_CRYPTO} ${SOURCES_SYSLOG} ${HTTPD_SOURCES})
target_link_libraries(${PROJECT_NAME} xmrig-cuda ${UV_LIBRARIES} ${MHD_LIBRARY} ${LIBS} ${EXTRA_LIBS} ${CPUID_LIB})
