# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Adapted for the Triton PyTorch backend from
# triton-inference-server-backend/triton_rocm_hipify.cmake.
# Added .hh glob patterns for PyTorch backend header convention.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(auto_set_source_files_hip_language)
  foreach(f ${ARGN})
    if(f MATCHES ".*\\.cu$")
      set_source_files_properties(${f} PROPERTIES LANGUAGE HIP)
    endif()
  endforeach()
endfunction()

# cuda_dir is informational; globs are rooted at REPO_ROOT/src and REPO_ROOT/include.
function(hipify cuda_dir in_excluded_file_patterns out_generated_cc_files out_generated_cu_files)
  set(hipify_tool ${REPO_ROOT}/amd_hipify.py)

  file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
    "${REPO_ROOT}/src/*.h"
    "${REPO_ROOT}/src/*.hh"
    "${REPO_ROOT}/src/*.cc"
    "${REPO_ROOT}/src/*.cuh"
    "${REPO_ROOT}/src/*.cu"
    "${REPO_ROOT}/include/*.h"
    "${REPO_ROOT}/include/*.hh"
    "${REPO_ROOT}/include/*.cc"
    "${REPO_ROOT}/include/*.cuh"
    "${REPO_ROOT}/include/*.cu"
  )

  # Exclude non-source files that happen to match the patterns
  list(FILTER srcs EXCLUDE REGEX "libtriton_pytorch\\.ldscript$")
  list(FILTER srcs EXCLUDE REGEX "\\.py$")

  set(excluded_file_patterns ${${in_excluded_file_patterns}})

  message(STATUS "Hipify file list: ${srcs}")
  message(STATUS "Project root: ${REPO_ROOT}")

  foreach(f ${srcs})
    message(STATUS "Hipifying ${f}")
    file(RELATIVE_PATH cuda_f_rel "${REPO_ROOT}" ${f})
    string(REPLACE "cuda" "rocm" rocm_f_rel ${cuda_f_rel})
    set(f_out "${CMAKE_CURRENT_BINARY_DIR}/amdgpu/${rocm_f_rel}")

    add_custom_command(
      OUTPUT ${f_out}
      COMMAND Python3::Interpreter ${hipify_tool}
        --hipify_perl ${TRITON_HIPIFY_PERL}
        ${f} -o ${f_out}
      DEPENDS ${hipify_tool} ${f}
      COMMENT "Hipify: ${cuda_f_rel} -> amdgpu/${rocm_f_rel}"
    )

    if(f MATCHES "kernel\\..*")
      list(APPEND generated_cu_files ${f_out})
      if(f MATCHES ".*\\.h$" OR f MATCHES ".*\\.hh$")
        list(APPEND generated_cc_files ${f_out})
      endif()
    else()
      list(APPEND generated_cc_files ${f_out})
    endif()
  endforeach()

  set_source_files_properties(${generated_cc_files} PROPERTIES GENERATED TRUE)
  set_source_files_properties(${generated_cu_files} PROPERTIES GENERATED TRUE)
  auto_set_source_files_hip_language(${generated_cu_files})
  set(${out_generated_cc_files} ${generated_cc_files} PARENT_SCOPE)
  set(${out_generated_cu_files} ${generated_cu_files} PARENT_SCOPE)
endfunction()
