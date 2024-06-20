cmake_minimum_required(VERSION 3.5)
project(liteqwen LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CUDA_VERBOSE_BUILD ON)

# cuda flags from xformers
set(CUDA_FLAGS
  -DHAS_PYTORCH
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  --expt-extended-lambda
  --expt-relaxed-constexpr
  --use_fast_math
  -D_ENABLE_EXTENDED_ALIGNED_STORAGE
  --ptxas-options=-v
  --threads
  4
  --ptxas-options=-O2 # for cuda > 11.6
  --ptxas-options=-allow-expensive-optimizations=true # for cuda > 11.6
)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


# lookup for pytorch and include its library
execute_process(
    COMMAND
      python -c
      "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_PATH)
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
message(TORCH_PATH= ${TORCH_PATH})

set(TORCH_INCLUDE ${TORCH_PATH}/include)
include_directories("${TORCH_INCLUDE}")
set(TORCH_LINKED_LIB ${TORCH_PATH}/lib)

set(CMAKE_BUILD_TYPE "Release")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -O2 /std:c++17 /arch:AVX /source-charset:utf-8")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()

message(STATUS "CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS})
message(STATUS "CUDA_FLAGS" ${CUDA_FLAGS})


# =============== pack implementing codes ============
set(LITEQWEN_CXX_SOURCES src/core_cpu.cpp src/pool.cpp src/kv_cache.cpp src/generate.cpp src/liteqwen.cpp src/entities.cpp src/json11.cpp)

include_directories(src/include)
include_directories(src/include/device)
include_directories(src/cutlass_fmha)
include_directories(src/exllama/cuda_func)
include_directories(src/exllama)
# message(STATUS "CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES " ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set(LITEQWEN_CUDA_SOURCES src/core_gpu.cu src/forward_gpu.cu src/sampling.cu src/sgemm_lora.cu)
set(LITEQWEN_LINKED_LIBS ${LITEQWEN_LINKED_LIBS} cublas)

message(STATUS "LITEQWEN_LINKED_LIBS " ${LITEQWEN_LINKED_LIBS})
set(CMAKE_CUDA_ARCHITECTURES "native")

# only compile xformer forward attention kernels.
file(GLOB XFORMER_AUTOGEN_IMPL
     "src/cutlass_fmha/autogen/impl/cutlassF_f*.cu" "src/cutlass_fmha/autogen/impl/cutlassF_b*.cu"
)


# file(GLOB XFORMER_FLASH_REQUIRE
#      "src/c10/util/*.cpp" "src/c10/cuda/*.cpp" "src/c10/core/impl/*.cpp" "src/aten/ATen/*.cpp"
# )
# file(GLOB XFORMER_FLASH_EXCLUDE src/c10/util/signal_handler.cpp src/aten/ATen/FunctionalInverses.cpp src/aten/ATen/FunctionalTensorWrapper.cpp src/aten/ATen/ParallelCommon.cpp src/aten/ATen/ParallelOpenMP.cpp src/aten/ATen/Version.cpp)
# list(REMOVE_ITEM XFORMER_FLASH_REQUIRE ${XFORMER_FLASH_EXCLUDE}) # ${XFORMER_FLASH_REQUIRE}

message(STATUS "XFORMER_AUTOGEN_IMPL" ${XFORMER_AUTOGEN_IMPL})
set(LITEQWEN_ATTN_SOURCES src/cutlass_fmha/xformer_attention.cu ${XFORMER_AUTOGEN_IMPL}) # src/cutlass_fmha/xformer_flash_decode.cu

# compile xllama for 4bit quantization
file(GLOB EXLLAMA_IMPL "src/exllama/cuda_func/q4_matrix.cu" "src/exllama/cuda_func/q4_matmul.cu" "src/exllama/cuda_func/column_remap.cu" "src/exllama/cuda_buffers.cu")
set(LITEQWEN_QUANT_SOURCES src/exllama/exllama_liteqwen_ext.cpp ${EXLLAMA_IMPL})

# ====== python library and connector ==================
add_library(liteqwen_conn SHARED
            ${LITEQWEN_CXX_SOURCES}
            ${LITEQWEN_CUDA_SOURCES}
            ${LITEQWEN_ATTN_SOURCES}
            ${LITEQWEN_QUANT_SOURCES}
            src/liteqwen_py.cpp)
target_compile_features(liteqwen_conn PUBLIC cuda_std_17)
target_link_libraries(liteqwen_conn PUBLIC ${LITEQWEN_LINKED_LIBS} ${TORCH_LINKED_LIB})
target_include_directories(liteqwen_conn PUBLIC third-party/cutlass/include third-party/cutlass/examples)
target_include_directories(liteqwen_conn PUBLIC src/exllama)

target_compile_options(liteqwen_conn PUBLIC
    # $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

# ======== postprocess commands =======
if (${CMAKE_HOST_WIN32})
    add_custom_command(
            TARGET liteqwen_conn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory package
            COMMAND ${CMAKE_COMMAND} -E make_directory package/liteqwen_py
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/scripts ${CMAKE_BINARY_DIR}/package/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/liteqwen_conn.dll ${CMAKE_BINARY_DIR}/package/liteqwen_py/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/liteqwen_conn.dll
    )
else()
    add_custom_command(
            TARGET liteqwen_conn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory package
            COMMAND ${CMAKE_COMMAND} -E make_directory package/liteqwen_py
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/scripts ${CMAKE_BINARY_DIR}/package/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libliteqwen_conn.* ${CMAKE_BINARY_DIR}/package/liteqwen_py/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/libliteqwen_conn.*
    )
endif()