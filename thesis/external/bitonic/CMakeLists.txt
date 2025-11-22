# This file defines the bitonic_rt library component
# Based on code from https://zenodo.org/records/15169905

# ------------------------------------------------------------------------------
# liboblivious (constant-time primitives needed by bitonic.c)
# ------------------------------------------------------------------------------
add_library(liboblivious STATIC
    third_party/liboblivious/algorithms.c
    third_party/liboblivious/opagedmem.c
    third_party/liboblivious/oram.c)

target_include_directories(liboblivious PUBLIC
    third_party/liboblivious/include)
set_target_properties(liboblivious PROPERTIES C_STANDARD 11)

# Apply aggressive optimisations to liboblivious (only in Release builds)
target_compile_options(liboblivious PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG -mno-avx512f>
    $<$<CONFIG:RelWithDebInfo>:-O3 -march=native -DNDEBUG -mno-avx512f -g>)

# ------------------------------------------------------------------------------
# Bitonic sorter + tiny task runtime
# ------------------------------------------------------------------------------
add_library(bitonic_rt STATIC
    bitonic.c
    threading.c
    synch.c)

target_include_directories(bitonic_rt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                           # bitonic.h, elem_t.h, threading.h …
    third_party/liboblivious/include
    ${CMAKE_SOURCE_DIR})                                  # for inputs.h and other project headers

# link against liboblivious and pthreads
target_link_libraries(bitonic_rt PUBLIC liboblivious Threads::Threads)
set_target_properties(bitonic_rt PROPERTIES C_STANDARD 11)

target_compile_options(bitonic_rt PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG -mno-avx512f>
    $<$<CONFIG:RelWithDebInfo>:-O3 -march=native -DNDEBUG -mno-avx512f -g>) 
