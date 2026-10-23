# Standalone build of the CPU inference engine: the NAM loaders and kernels, their
# unit tests, and the golden render tool. It needs no Pulp SDK and no GPU, so it
# builds on any CI runner. Used by .github/workflows/cpu-engine.yml.
#
#   cmake -S tools/cpu-engine -B build-cpu -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cpu -j4 && ctest --test-dir build-cpu --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(gpu_nam_cpu_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
# Pinned to the versions the Pulp SDK uses.
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
FetchContent_Declare(choc
    GIT_REPOSITORY https://github.com/danielraffel/choc.git   # the fork tag Pulp pins (pulp-webview-dnd-poc1)
    GIT_TAG df148a41a6bc9cbd67727532c4d5c9d8aa6d5d60)
FetchContent_MakeAvailable(Catch2)
FetchContent_GetProperties(choc)
if(NOT choc_POPULATED)
    FetchContent_Populate(choc)
endif()

set(GPU_NAM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
set(GPU_NAM_ACCELERATE 0)
if(APPLE)
    set(GPU_NAM_ACCELERATE 1)
endif()

function(gpu_nam_cpu_target name)
    target_include_directories(${name} PRIVATE "${GPU_NAM_SRC}" "${choc_SOURCE_DIR}")
    target_compile_definitions(${name} PRIVATE
        GPU_NAM_MODELS_DIR="${GPU_NAM_SRC}/models"
        GPU_NAM_HAVE_ACCELERATE=${GPU_NAM_ACCELERATE})
    if(APPLE)
        target_link_libraries(${name} PRIVATE "-framework Accelerate")
    endif()
endfunction()

enable_testing()
foreach(suite model lstm a2 convnet linear)
    add_executable(cpu-nam-${suite}-test "${GPU_NAM_SRC}/test_nam_${suite}.cpp")
    gpu_nam_cpu_target(cpu-nam-${suite}-test)
    target_link_libraries(cpu-nam-${suite}-test PRIVATE Catch2::Catch2WithMain)
    add_test(NAME cpu-nam-${suite}-test COMMAND cpu-nam-${suite}-test)
endforeach()

add_executable(gpu-nam-golden-render "${GPU_NAM_SRC}/nam_golden_render.cpp")
gpu_nam_cpu_target(gpu-nam-golden-render)
