if (CMAKE_CXX_COMPILER_ID MATCHES GNU)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -Wall -Wno-strict-aliasing")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -funroll-loops -fvariable-expansion-in-unroller -ftree-loop-if-convert-stores -fmerge-all-constants -fbranch-target-load-optimize2")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes -Wall -std=c++14 -fno-exceptions -fno-rtti")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -s -funroll-loops -fvariable-expansion-in-unroller -ftree-loop-if-convert-stores -fmerge-all-constants -fbranch-target-load-optimize2")

    if (WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
        add_definitions(/D__STDC_FORMAT_MACROS)
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    endif()

    add_definitions(/D_GNU_SOURCE)

    #set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -gdwarf-2")

elseif (CMAKE_CXX_COMPILER_ID MATCHES MSVC)

    message(${CMAKE_C_FLAGS_RELEASE})
    message(${CMAKE_CXX_FLAGS_RELEASE})

    set(CMAKE_C_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")

elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -Wall")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -funroll-loops -fmerge-all-constants")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes -Wall -std=c++14 -fno-exceptions -fno-rtti")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -funroll-loops -fmerge-all-constants")

endif()