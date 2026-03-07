cmake_minimum_required(VERSION 3.16)
project(gam C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CORE_SOURCES csrc/gam.c csrc/wl_api.c)

# Shared library
add_library(gam SHARED ${CORE_SOURCES})
target_include_directories(gam PUBLIC csrc)
target_compile_definitions(gam PRIVATE _POSIX_C_SOURCE=200809L)
target_compile_options(gam PRIVATE -Wall -Wextra -Wpedantic -O2)
target_link_libraries(gam m)
set_target_properties(gam PROPERTIES OUTPUT_NAME gam)

# Test executable
if(BUILD_TESTING)
    add_executable(test_gam test/test_c.c)
    target_link_libraries(test_gam gam)
    target_include_directories(test_gam PRIVATE csrc)
    target_link_libraries(test_gam m)
endif()
