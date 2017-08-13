cmake_minimum_required(VERSION 3.0)
project(xmrig)

option(WITH_AEON "CryptoNight-Lite support" ON)


include (CheckIncludeFile)


set(HEADERS
    src/3rdparty/align.h
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
    src/net/Job.h
    src/net/JobResult.h
    src/net/Network.h
    src/net/SubmitResult.h
    src/net/Url.h
    src/net/strategies/DonateStrategy.h
    src/net/strategies/FailoverStrategy.h
    src/net/strategies/SinglePoolStrategy.h
    src/Options.h
    src/Summary.h
    src/version.h
    src/workers/CudaWorker.h
    src/workers/Handle.h
    src/workers/Hashrate.h
    src/workers/GpuThread.h
    src/workers/Workers.h
   )

set(HEADERS_CRYPTO
    src/crypto/c_blake256.h
    src/crypto/c_groestl.h
    src/crypto/c_jh.h
    src/crypto/c_keccak.h
    src/crypto/c_skein.h
    src/crypto/CryptoNight.h
    src/crypto/CryptoNight_p.h
    src/crypto/CryptoNight_test.h
    src/crypto/groestl_tables.h
    src/crypto/hash.h
    src/crypto/skein_port.h
   )

set(SOURCES
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
    src/net/Url.cpp
    src/Options.cpp
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
    src/crypto/soft_aes.c
    src/crypto/soft_aes.c
    src/crypto/CryptoNight.cpp
   )

if (WIN32)
    set(SOURCES_OS
        res/app.rc
        src/App_win.cpp
        src/Cpu_win.cpp
        src/net/Network_win.cpp
        )

    add_definitions(/DWIN32)
    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv)
elseif (APPLE)
    set(SOURCES_OS
        src/App_unix.cpp
        src/Cpu_mac.cpp
        src/net/Network_mac.cpp
        )
else()
    set(SOURCES_OS
        src/App_unix.cpp
        src/Cpu_unix.cpp
        src/net/Network_unix.cpp
        )

    set(EXTRA_LIBS pthread)
endif()

add_definitions(/DUNICODE)
add_definitions(/DXMRIG_NO_LIBCPUID)
#add_definitions(/DAPP_DEBUG)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(UV REQUIRED)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()


include(cmake/flags.cmake)
include(cmake/CUDA.cmake)


CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
if (HAVE_SYSLOG_H)
    add_definitions(/DHAVE_SYSLOG_H)
    set(SOURCES_SYSLOG src/log/SysLog.h src/log/SysLog.cpp)
endif()

include_directories(src)
include_directories(src/3rdparty)
include_directories(src/3rdparty/jansson)
include_directories(${UV_INCLUDE_DIR})

add_subdirectory(src/3rdparty/jansson)

add_executable(xmrig ${HEADERS} ${SOURCES} ${SOURCES_OS} ${SOURCES_CPUID} ${HEADERS_CRYPTO} ${SOURCES_CRYPTO} ${SOURCES_SYSLOG})
target_link_libraries(xmrig xmrig-cuda jansson ${UV_LIBRARIES} ${LIBS} ${EXTRA_LIBS} ${CPUID_LIB})
