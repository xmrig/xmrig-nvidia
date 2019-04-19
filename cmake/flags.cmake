if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(/DNDEBUG)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)

    if (XMRIG_ARM)
         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    else()
         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -Wall")
    endif()
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wno-strict-aliasing")

    if (XMRIG_ARM)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -flax-vector-conversions")
    else()
	 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -maes -Wall")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s -Wno-sign-compare")

    if (WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    endif()

    add_definitions(/D_GNU_SOURCE)

    #set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -gdwarf-2")

elseif (CMAKE_CXX_COMPILER_ID MATCHES MSVC)

    set(CMAKE_C_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/D_CRT_NONSTDC_NO_WARNINGS)
    add_definitions(/DNOMINMAX)

elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang)

    if (XMRIG_ARM)
         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    else()
         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -Wall")
    endif()

    if (XMRIG_ARM)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-exceptions -fno-rtti")
    else()
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes -Wall -fno-exceptions -fno-rtti")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

endif()
