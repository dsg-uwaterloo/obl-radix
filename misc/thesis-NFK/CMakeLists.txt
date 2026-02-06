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
    main_serial.cpp
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

# Set radix fan-out
target_compile_definitions(radix_partition PUBLIC 
    NUM_RADIX_BITS=10
    NUM_PASSES=2)

# Link against all required libraries
target_link_libraries(OblRadix PRIVATE 
    bitonic_rt 
    radix_partition 
    Threads::Threads
    OpenSSL::Crypto
    TBB::tbb)
