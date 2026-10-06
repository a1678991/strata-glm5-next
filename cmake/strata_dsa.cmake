# GLM-5.3-Flash (glm5-next) DSA layer kernels: port plan Stage 4, worker D (~/dev/llm/docs/strata-glm-port.md).
# A library of its own, so nothing Qwen links changes: `strata_kernels` and every target that links it build exactly
# as before; only these targets link `strata_dsa_kernels`.
add_library(strata_dsa_kernels STATIC
  src/kernels/cuda/mla_kv.cu
  src/kernels/cuda/dsa_indexer.cu
  src/kernels/cuda/mla_decode.cu)
target_link_libraries(strata_dsa_kernels PUBLIC strata_kernels)

if(STRATA_BUILD_TESTS)
  # The cache layout: int8 / f16 append bitwise against a host reference, the page read, the pooling through a
  # streamed cache, and streamed vs resident attention bitwise through kv_stream_resolve (GPU, synthetic, no model).
  add_executable(dsa_kv_parity tests/dsa/dsa_kv_parity.cpp)
  target_include_directories(dsa_kv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dsa_kv_parity PRIVATE strata_dsa_kernels ${_strata_gpu_runtime_target})
  add_test(NAME dsa_kv_parity COMMAND dsa_kv_parity)

endif()
