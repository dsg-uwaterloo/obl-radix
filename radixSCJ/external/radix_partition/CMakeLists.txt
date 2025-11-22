# This file defines the radix_partition library component
# Based on code from https://github.com/agora-ecosystem/tee-bench

set(CMAKE_C_STANDARD 11)

# Build the radix_partition static library
add_library(radix_partition STATIC
    radix_join_counts.c
    radix_join_idx.c
    task_queue.c
    util.c)

# Set include directories for the library
target_include_directories(radix_partition PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(radix_partition PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG -mno-avx512f>
    $<$<CONFIG:RelWithDebInfo>:-O3 -march=native -DNDEBUG -mno-avx512f -g>)

# Link against pthreads
target_link_libraries(radix_partition PUBLIC Threads::Threads)
