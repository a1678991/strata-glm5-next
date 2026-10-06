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

add_library(strata_glm_prefill_kernels STATIC src/prefill/glm_prefill_kernels.cu)
target_include_directories(strata_glm_prefill_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           ${_strata_gpu_include_directories})
target_link_libraries(strata_glm_prefill_kernels PUBLIC strata_dsa_kernels ${_strata_gpu_blas_target}
                      ${_strata_gpu_runtime_target})

