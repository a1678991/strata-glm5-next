# cmake/glm_prefill.cmake - GLM-5.3-Flash (llama.cpp arch glm5-next) prompt path: port plan Stage 6, worker E
# (~/dev/llm/docs/strata-glm-port.md).  Kept out of CMakeLists.txt (one include line there).
#
# strata_glm_prefill_kernels: the prompt path's own kernels (src/prefill/glm_prefill_kernels.cu) and its FP16 GEMM; a
# library of its own that carries no ggml, so the parity tests can link it beside ggml-cpu at f1cee99.  The
# orchestration (src/prefill/glm_prefill.cpp) joins strata_prefill, which the `strata` program links; nothing the Qwen
# prompt path compiles changes.
if(NOT (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))
  return()
endif()

add_library(strata_glm_prefill_kernels STATIC src/prefill/glm_prefill_kernels.cu src/prefill/glm_gemm16.cu)
target_include_directories(strata_glm_prefill_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           ${_strata_gpu_include_directories})
target_link_libraries(strata_glm_prefill_kernels PUBLIC strata_dsa_kernels ${_strata_gpu_blas_target}
                      ${_strata_gpu_runtime_target})
if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
  # the FP16 GEMM through hipBLASLt's heuristic algorithm (Gemm16): ~6-10x hipblasGemmEx on gfx1201
  target_compile_definitions(strata_glm_prefill_kernels PRIVATE STRATA_HIPBLASLT_AVAILABLE=1)
  target_link_libraries(strata_glm_prefill_kernels PUBLIC roc::hipblaslt)
endif()

if(TARGET strata_prefill)
  target_sources(strata_prefill PRIVATE src/prefill/glm_prefill.cpp)
  target_link_libraries(strata_prefill PUBLIC strata_glm_prefill_kernels)
endif()

# Parity tests against ggml-cpu at llama.cpp f1cee99 (port plan Stage 6), registered only with the oracles' paths:
#   prefill_kda_parity, prefill_moe_parity - the GLM oracle (STRATA_GLM_ORACLE_DIR / _BUILD_DIR, shared ggml)
#   dsa_prompt_parity                     - the DSA oracle (STRATA_DSA_ORACLE_DIR / _BUILD_DIR, strata_dsa_oracle)
if(NOT STRATA_BUILD_TESTS)
  return()
endif()
if(STRATA_GLM_ORACLE_DIR AND STRATA_GLM_ORACLE_BUILD_DIR)
  set(_glm_pf_oracle_libs)
  foreach(_c IN ITEMS ggml ggml-base ggml-cpu)
    string(REPLACE "-" "_" _v "STRATA_GLM_ORACLE_LIB_${_c}")
    find_library(${_v} NAMES ${_c} PATHS "${STRATA_GLM_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    list(APPEND _glm_pf_oracle_libs "${${_v}}")
  endforeach()
  foreach(_t IN ITEMS prefill_kda_parity prefill_moe_parity)
    add_executable(glm_${_t} tests/glm/${_t}.cpp)
    target_include_directories(glm_${_t} PRIVATE "${STRATA_GLM_ORACLE_DIR}/ggml/include" ${_strata_gpu_include_directories})
    target_link_libraries(glm_${_t} PRIVATE strata_glm_prefill_kernels strata_kernels ${_strata_gpu_runtime_target}
                          ${_glm_pf_oracle_libs})
    set_target_properties(glm_${_t} PROPERTIES BUILD_RPATH "${STRATA_GLM_ORACLE_BUILD_DIR}/bin")
    add_test(NAME glm_${_t} COMMAND glm_${_t})
    set_tests_properties(glm_${_t} PROPERTIES TIMEOUT 1200)
  endforeach()
else()
  message(STATUS "strata: the GLM oracle paths are unset - glm_prefill_kda_parity and glm_prefill_moe_parity are NOT "
                 "registered")
endif()
if(TARGET strata_dsa_oracle)
  add_executable(dsa_prompt_parity tests/dsa/dsa_prompt_parity.cpp)
  target_include_directories(dsa_prompt_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dsa_prompt_parity PRIVATE strata_glm_prefill_kernels strata_dsa_kernels strata_dsa_oracle
                        ${_strata_gpu_runtime_target})
  add_test(NAME dsa_prompt_parity COMMAND dsa_prompt_parity)
  set_tests_properties(dsa_prompt_parity PROPERTIES TIMEOUT 1800)
else()
  message(STATUS "strata: the DSA oracle is not built - dsa_prompt_parity is NOT registered")
endif()
