cmake_minimum_required(VERSION 3.16)
project(UnifiedPrototype)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -mno-avx512f")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -DNDEBUG -mno-avx512f -g")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -mno-avx512f")

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

# ------------------------------------------------------------------------------
# Tunables (compile-time constants)
# ------------------------------------------------------------------------------
set(BINS_PER_PART 32 CACHE STRING "Number of bins per partition")
set(NUM_RADIX_BITS 10 CACHE STRING "Radix bits per pass")
set(NUM_PASSES 2 CACHE STRING "Number of radix passes")
option(ENABLE_AVX2_GATHER_JOIN "Enable AVX2 gather path radix_partition" ON)

# ------------------------------------------------------------------------------
# External implementations
# ------------------------------------------------------------------------------
# Add bitonic sort implementation
add_subdirectory(external/bitonic)

# Add radix partition implementation  
add_subdirectory(external/radix_partition)

# ------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------
add_executable(OblRadix
    main.cpp
    oblivious_ops.cpp)

target_include_directories(OblRadix PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    external/bitonic
    external/radix_partition
    external/flexway_oshuffle)

target_sources(OblRadix PRIVATE
    external/flexway_oshuffle/common/encutils.cpp
    external/flexway_oshuffle/external_memory/server/serverBackend.cpp)

target_compile_options(OblRadix PRIVATE -pthread)
target_compile_definitions(OblRadix PRIVATE BINS_PER_PART=${BINS_PER_PART})

# Set radix fan-out
target_compile_definitions(radix_partition PUBLIC 
    NUM_RADIX_BITS=${NUM_RADIX_BITS}
    NUM_PASSES=${NUM_PASSES})

if (ENABLE_AVX2_GATHER_JOIN)
  target_compile_definitions(radix_partition PUBLIC ENABLE_AVX2_GATHER_JOIN=1)
endif()

# Link against all required libraries
target_link_libraries(OblRadix PRIVATE 
    bitonic_rt 
    radix_partition 
    Threads::Threads
    OpenSSL::Crypto
    OpenSSL::SSL 
    TBB::tbb
    OpenMP::OpenMP_CXX)
