# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Hipify script for the Triton PyTorch backend.
# Runs hipify-perl first, then applies PyTorch-specific fixups
# so the generated code links correctly against ROCm-built LibTorch
# and the hipified Triton backend utilities.

import argparse
import os
import subprocess


def hipify(hipify_perl_path, src_file_path, dst_file_path):
    dir_name = os.path.dirname(dst_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # ── Step 1: Run hipify-perl -roc ──────────────────────────────────
    # Handles CUDA runtime/driver API → HIP (cudaStream_t → hipStream_t,
    # cudaSetDevice → hipSetDevice, cuda_runtime_api.h → hip/hip_runtime_api.h, …)
    s = subprocess.run(
        [hipify_perl_path, "-roc", src_file_path],
        stdout=subprocess.PIPE, text=True, check=False,
    ).stdout

    # ── Step 2: Preprocessor guards ───────────────────────────────────
    s = s.replace("#ifdef TRITON_ENABLE_GPU", "#ifdef TRITON_ENABLE_ROCM")
    s = s.replace("#ifndef TRITON_ENABLE_GPU", "#ifndef TRITON_ENABLE_ROCM")

    # ── Step 3: Triton backend utility naming ─────────────────────────
    # The hipified backend library renames these (CudaStream→RocmStream, etc.).
    # We must match so that our calls resolve against the hipified backend API.
    s = s.replace("CudaStream", "RocmStream")
    s = s.replace("CudaEvent", "RocmEvent")

    # ── Step 4: Broad identifier replacement ──────────────────────────
    # Catches remaining variable names (cuda_copy → rocm_copy, etc.)
    # and matches the backend library's own hipify conventions.
    s = s.replace("cudaEvent", "hipEvent")
    s = s.replace("cuda", "rocm")
    s = s.replace("CUDA", "ROCM")

    # ── Step 5: Fix PyTorch C++ API references ─────────────────────────
    # PyTorch's ROCm build provides c10/hip/ headers that avoid any
    # <cuda.h> / <cuda_runtime.h> dependency.  Internally those headers
    # still define classes in the c10::cuda / at::cuda namespaces.
    # After the broad cuda→rocm / CUDA→ROCM replacements above we must:
    #   a) redirect include paths to c10/hip/ with HIP-prefixed filenames
    #   b) restore the c10::cuda / at::cuda namespaces and class names

    # (a) Include paths — specific header renames first, then catch-all
    s = s.replace("c10/rocm/ROCMCachingAllocator.h", "c10/hip/HIPCachingAllocator.h")
    s = s.replace("c10/rocm/ROCMAllocatorConfig.h", "c10/hip/HIPAllocatorConfig.h")
    s = s.replace("c10/rocm/ROCMGuard.h", "c10/hip/HIPGuard.h")
    s = s.replace("c10/rocm/ROCMStream.h", "c10/hip/HIPStream.h")
    s = s.replace("c10/rocm/ROCMEvent.h", "c10/hip/HIPEvent.h")
    s = s.replace("c10/rocm/ROCMException.h", "c10/hip/HIPException.h")
    s = s.replace("c10/rocm/ROCMMacros.h", "c10/hip/HIPMacros.h")
    s = s.replace("c10/rocm/ROCMFunctions.h", "c10/hip/HIPFunctions.h")
    s = s.replace("c10/rocm/ROCMMiscFunctions.h", "c10/hip/HIPMiscFunctions.h")
    s = s.replace("c10/rocm/ROCMGraphsC10Utils.h", "c10/hip/HIPGraphsC10Utils.h")
    s = s.replace("c10/rocm/ROCMAlgorithm.h", "c10/hip/HIPAlgorithm.h")
    s = s.replace("c10/rocm/ROCMMathCompat.h", "c10/hip/HIPMathCompat.h")
    s = s.replace("c10/rocm/ROCMDeviceAssertion.h", "c10/hip/HIPDeviceAssertion.h")
    s = s.replace("c10/rocm/ROCMDeviceAssertionHost.h", "c10/hip/HIPDeviceAssertionHost.h")
    # Catch-all for any remaining c10/rocm/ includes
    s = s.replace("c10/rocm/", "c10/hip/")

    # (b) Namespaces — HIP headers still define classes under c10::cuda
    s = s.replace("c10::rocm::", "c10::cuda::")
    s = s.replace("at::rocm::", "at::cuda::")
    s = s.replace("torch::rocm::", "torch::cuda::")

    # (b) Class / function names used in code
    s = s.replace("ROCMCachingAllocator", "CUDACachingAllocator")
    s = s.replace("ROCMGuard", "CUDAGuard")
    s = s.replace("ROCMStream", "CUDAStream")
    s = s.replace("setCurrentROCMStream", "setCurrentCUDAStream")

    # (b) Device type and queries
    s = s.replace("torch::kROCM", "torch::kCUDA")
    s = s.replace(".is_rocm()", ".is_cuda()")

    # PyTorch JIT codegen paths still use "cuda" in the source tree
    s = s.replace("torch/csrc/jit/codegen/rocm/", "torch/csrc/jit/codegen/cuda/")

    with open(dst_file_path, "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipify_perl", required=True)
    parser.add_argument("--output", "-o", help="output file")
    parser.add_argument("src", help="src")
    args = parser.parse_args()
    print("hipifying " + str(args.src))
    hipify(args.hipify_perl, args.src, args.output)
