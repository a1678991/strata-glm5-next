# cmake/glm_kernels.cmake - GLM-5.3-Flash (llama.cpp arch glm5-next) decode kernels and their parity tests.
#
# Kept out of CMakeLists.txt (one include line there) so the GLM port's kernel half does not collide with the
# geometry/loader changes to the main build file.  The kernels join strata_kernels; nothing calls them yet.
#
# The parity tests that compare against ggml-cpu need the oracle of the GLM reference, llama.cpp f1cee99, built as
# CPU-only shared libraries by tools/glm/build_oracle.sh:
#   -DSTRATA_GLM_ORACLE_DIR=<llama.cpp f1cee99 checkout> -DSTRATA_GLM_ORACLE_BUILD_DIR=<its build dir>
# Those executables link that ggml by path and nothing else that carries a ggml (not strata_kernels_cpu, whose static
# ggml is Strata's pinned 3cf0325).  Without the two paths they are not registered, and configure says so.
if(NOT (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
  return()
endif()

target_sources(strata_kernels PRIVATE
  src/kernels/cuda/kda_decode.cu
  src/kernels/cuda/mhc.cu
  src/kernels/cuda/router_sigmoid.cu)

if(NOT STRATA_BUILD_TESTS)
  return()
endif()
enable_testing()

set(STRATA_GLM_ORACLE_DIR "" CACHE PATH "llama.cpp f1cee99 checkout: the headers of the GLM parity tests' oracle")
set(STRATA_GLM_ORACLE_BUILD_DIR "" CACHE PATH "its CPU-only shared ggml build (tools/glm/build_oracle.sh)")
set(_glm_oracle_tests kda_parity mhc_parity router_sigmoid_parity)
if(STRATA_GLM_ORACLE_DIR AND STRATA_GLM_ORACLE_BUILD_DIR)
  set(_glm_oracle_libs)
  foreach(_c IN ITEMS ggml ggml-base ggml-cpu)
    string(REPLACE "-" "_" _v "STRATA_GLM_ORACLE_LIB_${_c}")
    find_library(${_v} NAMES ${_c} PATHS "${STRATA_GLM_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    list(APPEND _glm_oracle_libs "${${_v}}")
  endforeach()
  foreach(_t IN LISTS _glm_oracle_tests)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/glm/${_t}.cpp")
      continue()
    endif()
    add_executable(glm_${_t} tests/glm/${_t}.cpp)
    target_include_directories(glm_${_t} PRIVATE "${STRATA_GLM_ORACLE_DIR}/ggml/include"
                               ${_strata_gpu_include_directories})
    target_link_libraries(glm_${_t} PRIVATE strata_kernels ${_strata_gpu_runtime_target} ${_glm_oracle_libs})
    set_target_properties(glm_${_t} PROPERTIES BUILD_RPATH "${STRATA_GLM_ORACLE_BUILD_DIR}/bin")
    add_test(NAME glm_${_t} COMMAND glm_${_t})
    set_tests_properties(glm_${_t} PROPERTIES TIMEOUT 600)
  endforeach()
else()
  message(STATUS "strata: STRATA_GLM_ORACLE_DIR / STRATA_GLM_ORACLE_BUILD_DIR not set - the GLM kernel parity tests "
                 "against ggml-cpu f1cee99 (${_glm_oracle_tests}) are NOT registered (tools/glm/build_oracle.sh)")
endif()

# The routed-expert changes (clamp epilogue, Q6_K down rows) on the GPU and the CPU pool: ggml here is Strata's own
# pinned one (strata_kernels_cpu), whose Q6_K arithmetic is f1cee99's.  expert_dump is the Qwen bit-identity tool: its
# output, built once against the base tree and once against this one, must be byte-identical (see its header).
if(STRATA_NATIVE_EXPERTS)
  foreach(_t IN ITEMS expert_parity expert_dump)
    add_executable(glm_${_t} tests/glm/${_t}.cpp)
    target_include_directories(glm_${_t} PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(glm_${_t} PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  endforeach()
  add_test(NAME glm_expert_parity COMMAND glm_expert_parity)
  set_tests_properties(glm_expert_parity PROPERTIES TIMEOUT 900)
endif()
