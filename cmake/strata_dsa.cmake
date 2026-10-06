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

  # The oracle: ggml at llama.cpp f1cee99, CPU backend, static, built separately (its ggml is not Strata's pinned
  # 3cf0325, so it cannot be the add_subdirectory above):
  #   cmake -S <llama.cpp-f1cee99> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
  #         -DGGML_STATIC=ON -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=OFF
  #         -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
  #         -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  #   cmake --build <build> --target ggml ggml-cpu ggml-base
  set(STRATA_DSA_ORACLE_DIR "" CACHE PATH "llama.cpp f1cee99 checkout: the DSA parity tests' ggml oracle headers")
  set(STRATA_DSA_ORACLE_BUILD_DIR "" CACHE PATH "its CPU-only static ggml build (libggml.a, libggml-cpu.a, libggml-base.a)")
  if(STRATA_DSA_ORACLE_DIR AND STRATA_DSA_ORACLE_BUILD_DIR)
    find_package(Threads REQUIRED)
    set(_dsa_oracle_libs)
    foreach(component IN ITEMS ggml ggml-cpu ggml-base)
      string(REPLACE "-" "_" _var "STRATA_DSA_ORACLE_LIB_${component}")
      find_library(${_var} NAMES ${component} PATHS "${STRATA_DSA_ORACLE_BUILD_DIR}/ggml/src"
                   NO_DEFAULT_PATH NO_CACHE REQUIRED)
      list(APPEND _dsa_oracle_libs "${${_var}}")
    endforeach()
    add_library(strata_dsa_oracle STATIC tests/dsa/dsa_oracle.cpp)
    target_include_directories(strata_dsa_oracle PUBLIC "${STRATA_DSA_ORACLE_DIR}/ggml/include"
                               ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
    target_link_libraries(strata_dsa_oracle PUBLIC ${_dsa_oracle_libs} ${_dsa_oracle_libs} Threads::Threads m
                          ${_strata_gpu_runtime_target})
    foreach(t IN ITEMS dsa_indexer_parity)
      add_executable(${t} tests/dsa/${t}.cpp)
      target_include_directories(${t} PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(${t} PRIVATE strata_dsa_kernels strata_dsa_oracle ${_strata_gpu_runtime_target})
      add_test(NAME ${t} COMMAND ${t})
      set_tests_properties(${t} PROPERTIES TIMEOUT 1800)
    endforeach()
  else()
    message(STATUS "strata: STRATA_DSA_ORACLE_DIR / STRATA_DSA_ORACLE_BUILD_DIR unset - "
                   "dsa_indexer_parity (ggml f1cee99 oracle) is NOT registered")
  endif()
endif()
