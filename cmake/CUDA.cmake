if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

option(XMR-STAK_LARGEGRID "Support large CUDA block count > 128" ON)
if(XMR-STAK_LARGEGRID)
    add_definitions("-DXMR_STAK_LARGEGRID=${XMR-STAK_LARGEGRID}")
endif()

set(DEVICE_COMPILER "nvcc")
set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND DEVICE_COMPILER "clang")
endif()

set_property(CACHE CUDA_COMPILER PROPERTY STRINGS "${DEVICE_COMPILER}")

list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

set(CUDA_STATIC ON)
find_package(CUDA 7.5 REQUIRED)

set(LIBS ${LIBS} ${CUDA_LIBRARIES})

set(DEFAULT_CUDA_ARCH "30;50")

# Fermi GPUs are only supported with CUDA < 9.0
if (CUDA_VERSION VERSION_LESS 9.0)
    list(APPEND DEFAULT_CUDA_ARCH "20")
endif()

# add Pascal support for CUDA >= 8.0
if (NOT CUDA_VERSION VERSION_LESS 8.0)
    list(APPEND DEFAULT_CUDA_ARCH "60")
endif()

# add Volta support for CUDA >= 9.0
if (NOT CUDA_VERSION VERSION_LESS 9.0)
    list(APPEND DEFAULT_CUDA_ARCH "70")
endif()

set(CUDA_ARCH "${DEFAULT_CUDA_ARCH}" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")

# validate architectures (only numbers are allowed)
foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
    string(REGEX MATCH "^[0-9]+$" IS_NUMBER ${CUDA_ARCH})
    if(NOT IS_NUMBER)
        message(FATAL_ERROR "Defined compute architecture '${CUDA_ARCH_ELEM}' in "
                            "'${CUDA_ARCH}' is not an integral number, use e.g. '30' (for compute architecture 3.0).")
    endif()
    unset(IS_NUMBER)

    if(${CUDA_ARCH_ELEM} LESS 20)
        message(FATAL_ERROR "Unsupported CUDA architecture '${CUDA_ARCH_ELEM}' specified. "
                            "Use '20' (for compute architecture 2.0) or higher.")
    endif()
endforeach()

option(CUDA_SHOW_REGISTER "Show registers used for each kernel and compute architecture" OFF)
option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps" OFF)

if("${CUDA_COMPILER}" STREQUAL "clang")
    set(LIBS ${LIBS} cudart_static)
    set(CLANG_BUILD_FLAGS "-O3 -x cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
    # activation usage of FMA
    set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -ffp-contract=fast")

    if (CUDA_SHOW_REGISTER)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -Xcuda-ptxas -v")
    endif(CUDA_SHOW_REGISTER)

    if (CUDA_KEEP_FILES)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -save-temps=${PROJECT_BINARY_DIR}")
    endif(CUDA_KEEP_FILES)

    foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architectures
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} --cuda-gpu-arch=sm_${CUDA_ARCH_ELEM}")
    endforeach()

elseif("${CUDA_COMPILER}" STREQUAL "nvcc")
    # avoid that nvcc in CUDA < 8 tries to use libc `memcpy` within the kernel
    if (CUDA_VERSION VERSION_LESS 8.0)
        add_definitions(-D_FORCE_INLINES)
        add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
    endif()
    foreach(CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architecture
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            "-Wno-deprecated-gpu-targets --generate-code arch=compute_${CUDA_ARCH_ELEM},code=sm_${CUDA_ARCH_ELEM} --generate-code arch=compute_${CUDA_ARCH_ELEM},code=compute_${CUDA_ARCH_ELEM}")
    endforeach()

    # give each thread an independent default stream
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} static")

    option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

    if (CUDA_SHOW_CODELINES)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -lineinfo)
        set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif(CUDA_SHOW_CODELINES)

    if (CUDA_SHOW_REGISTER)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
    endif(CUDA_SHOW_REGISTER)

    if (CUDA_KEEP_FILES)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}")
    endif(CUDA_KEEP_FILES)

else()
    message(FATAL_ERROR "selected CUDA compiler '${CUDA_COMPILER}' is not supported")
endif()


set(CUDA_SOURCES
    src/nvidia/cryptonight.h
    src/nvidia/cuda_extra.h
    src/nvidia/cuda_aes.hpp
    src/nvidia/cuda_blake.hpp
    src/nvidia/cuda_device.hpp
    src/nvidia/cuda_groestl.hpp
    src/nvidia/cuda_jh.hpp
    src/nvidia/cuda_keccak.hpp
    src/nvidia/cuda_skein.hpp
    src/nvidia/cuda_core.cu
    src/nvidia/cuda_extra.cu
)

if("${CUDA_COMPILER}" STREQUAL "clang")
    add_library(xmrig-cuda STATIC ${CUDA_SOURCES})
    
    set_target_properties(xmrig-cuda PROPERTIES COMPILE_FLAGS ${CLANG_BUILD_FLAGS})
    set_target_properties(xmrig-cuda PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
else()
    cuda_add_library(xmrig-cuda STATIC ${CUDA_SOURCES})
endif()
