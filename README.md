<!--
# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->


# PyTorch (LibTorch) Backend - ROCm Edition

This is the ROCm-enabled fork of the
[Triton Inference Server PyTorch Backend](https://github.com/triton-inference-server/pytorch_backend).
It adds support for AMD GPUs via ROCm/HIP while maintaining full compatibility
with the upstream feature set.

## ROCm Hipification

The backend's CUDA C++ source code is automatically converted to HIP at build
time using a two-stage process:

1. **`hipify-perl`** (shipped with ROCm at `/opt/rocm/bin/hipify-perl`) performs
   the bulk conversion of CUDA runtime/driver API calls to their HIP
   equivalents (e.g. `cudaStream_t` to `hipStream_t`, `cudaSetDevice` to
   `hipSetDevice`, `cuda_runtime_api.h` to `hip/hip_runtime_api.h`).

2. **`amd_hipify.py`** applies project-specific post-processing:
   - Switches preprocessor guards from `TRITON_ENABLE_GPU` to
     `TRITON_ENABLE_ROCM`.
   - Renames Triton backend utilities to match the hipified backend library
     (`CudaStream` to `RocmStream`, etc.).
   - Redirects PyTorch include paths from `c10/cuda/` to `c10/hip/` so the
     build does not depend on CUDA compatibility headers.
   - Preserves PyTorch's `c10::cuda` / `at::cuda` C++ namespaces, which
     remain unchanged even in ROCm-built PyTorch.

The CMake module `triton_rocm_hipify.cmake` orchestrates this process,
generating hipified source files under `build/amdgpu/` which are then compiled
in place of the original CUDA sources.

## Building for ROCm (Standalone)

> **Note:** A standalone build of the PyTorch backend is typically not necessary.
> In most cases, the
> [Triton Server build system](https://github.com/ROCm/triton-inference-server-server)
> will build all backends (including PyTorch) into the final server artifacts.
> The standalone build instructions below are provided for development and
> testing purposes only.

### Prerequisites

- AMD GPU with ROCm support (e.g. MI210, MI250, MI300X, MI355X)
- ROCm 7.2 installed (`/opt/rocm`)
- Docker (recommended)

### Step 1: Start a ROCm container

Use an Ubuntu-based container with ROCm installed:

```bash
docker run \
  --name pytorch_container \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -it \
  --net=host \
  rocm/dev-ubuntu-24.04:7.2.2-complete \
  /bin/bash
```

### Step 2: Install dependencies

```bash
apt-get update && apt-get install -y \
  git cmake rapidjson-dev python3-dev python3-pip patchelf libboost-dev libre2-dev

pip3 install --break-system-packages \
  torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

### Step 3: Configure and build

```bash
mkdir /workspace && cd /workspace && git clone https://github.com/ROCm/triton-inference-server-pytorch_backend.git
cd /workspace/triton-inference-server-pytorch_backend
mkdir -p build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX:PATH=$(pwd)/install \
  -DTRITON_ENABLE_GPU=OFF \
  -DTRITON_ENABLE_ROCM=ON \
  -DTRITON_ROCM_HOME=/opt/rocm \
  -DTRITON_HIPIFY_PERL=/opt/rocm/bin/hipify-perl \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)');/opt/rocm" \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DTRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
  -DTRITON_PYTORCH_NVSHMEM=OFF \
  -DTRITON_BACKEND_REPO_TAG=r25.12 \
  -DTRITON_CORE_REPO_TAG=r25.12 \
  -DTRITON_COMMON_REPO_TAG=r25.12

make -j$(nproc) install
```

The built backend will be located at `build/install/backends/pytorch/`.

### CMake Options for ROCm

| Option | Default | Description |
|--------|---------|-------------|
| `TRITON_ENABLE_ROCM` | `ON` | Enable ROCm/HIP support (mutually exclusive with `TRITON_ENABLE_GPU`) |
| `TRITON_ROCM_HOME` | `/opt/rocm` | ROCm installation path |
| `TRITON_HIPIFY_PERL` | *(empty)* | Path to `hipify-perl` executable |
| `TRITON_ROCM_REPO_ORGANIZATION` | `https://github.com/ROCm` | Git organization for ROCm forks of core/backend |
| `TRITON_ROCM_REPO_TAG` | `rocm7.2_r25.12` | Git tag for ROCm fork repos |

---

> The sections below are from the upstream
> [triton-inference-server/pytorch_backend](https://github.com/triton-inference-server/pytorch_backend)
> repository and describe the NVIDIA/CUDA build and general usage of the
> PyTorch backend. They apply equally to the ROCm edition unless noted otherwise.

---

# PyTorch (LibTorch) Backend

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

The Triton backend for
[PyTorch](https://github.com/pytorch/pytorch)
is designed to run
[TorchScript](https://pytorch.org/docs/stable/jit.html)
models using the PyTorch C++ API.
All models created in PyTorch using the python API must be traced/scripted to produce a TorchScript model.

You can learn more about Triton backends in the
[Triton Backend](https://github.com/triton-inference-server/backend)
repository.

Ask questions or report problems using
[Triton Server issues](https://github.com/triton-inference-server/server/issues).

Be sure to read all the information below as well as the
[general Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the [Triton Server](https://github.com/triton-inference-server/server) repository.

## Build the PyTorch Backend

Use a recent cmake to build.
First install the required dependencies.

```bash
apt-get install rapidjson-dev python3-dev python3-pip
pip3 install patchelf==0.17.2
```

An appropriate PyTorch container from [NVIDIA NGC Catalog](https://ngc.nvidia.com) must be used.
For example, to build a backend that uses the 23.04 version of the PyTorch container from NGC:

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_DOCKER_IMAGE="nvcr.io/nvidia/pytorch:23.04-py3" ..
make install
```

The following required Triton repositories will be pulled and used in the build.
By default, the `main` head will be used for each repository but the listed CMake argument can be used to override the value.

* triton-inference-server/backend: `-DTRITON_BACKEND_REPO_TAG=[tag]`
* triton-inference-server/core: `-DTRITON_CORE_REPO_TAG=[tag]`
* triton-inference-server/common: `-DTRITON_COMMON_REPO_TAG=[tag]`

## Build the PyTorch Backend With Custom PyTorch

Currently, Triton requires that a specially patched version of PyTorch be used with the PyTorch backend.
The full source for these PyTorch versions are available as Docker images from
[NGC](https://ngc.nvidia.com).

For example, the PyTorch version compatible with the 25.09 release of Triton is available as `nvcr.io/nvidia/pytorch:25.09-py3` which supports PyTorch version `2.9.0a0`.

> [!NOTE]
> Additional details and version information can be found in the container's
> [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-09.html#rel-25-09).

Copy over the LibTorch and TorchVision headers and libraries from the
[PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
into local directories.
You can see which headers and libraries are needed/copied from the docker.

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_PYTORCH_INCLUDE_PATHS="<PATH_PREFIX>/torch;<PATH_PREFIX>/torch/torch/csrc/api/include;<PATH_PREFIX>/torchvision" -DTRITON_PYTORCH_LIB_PATHS="<LIB_PATH_PREFIX>" ..
make install
```

## Using the PyTorch Backend

### PyTorch 2.0 Models

PyTorch 2.0 features are available.
However, Triton's PyTorch backend requires a serialized representation of the model in the form a `model.pt` file.
The serialized representation of the model can be generated using PyTorch's
[`torch.save()`](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#id1)
function to generate the `model.pt` file.

The model repository should look like:

```bash
model_repository/
`-- model_directory
    |-- 1
    |   `-- model.pt
    `-- config.pbtxt
```

Where `model.pt` is the serialized representation of the model.

### TorchScript Models

The model repository should look like:

```bash
model_repository/
`-- model_directory
    |-- 1
    |   `-- model.pt
    `-- config.pbtxt
```

The `model.pt` is the TorchScript model file.

## Configuration

Triton exposes some flags to control the execution mode of the TorchScript models through the `Parameters` section of the model's `config.pbtxt` file.

### Configuration Options

* `default_model_name`:
  Instructs the Triton PyTorch backend to load the model from a file of the given name.

  The model config specifying the option would look like:

  ```proto
  default_model_name: "another_file_name.pt"
  ```

### Parameters

* `DISABLE_OPTIMIZED_EXECUTION`:
  Boolean flag to disable the optimized execution of TorchScript models.
  By default, the optimized execution is always enabled.

  The initial calls to a loaded TorchScript model take a significant amount of time.
  Due to this longer model warmup
  ([pytorch #57894](https://github.com/pytorch/pytorch/issues/57894)),
  Triton also allows execution of models without these optimizations.
  In some models, optimized execution does not benefit performance
  ([pytorch #19978](https://github.com/pytorch/pytorch/issues/19978))
  and in other cases impacts performance negatively
  ([pytorch #53824](https://github.com/pytorch/pytorch/issues/53824)).

  The section of model config file specifying this parameter will look like:

  ```proto
  parameters: {
    key: "DISABLE_OPTIMIZED_EXECUTION"
    value: { string_value: "true" }
  }
  ```

* `INFERENCE_MODE`:

  Boolean flag to enable the Inference Mode execution of TorchScript models.
  By default, the inference mode is enabled.

  [InferenceMode](https://pytorch.org/cppdocs/notes/inference_mode.html) is a new RAII guard analogous to `NoGradMode` to be used when you are certain your operations will have no interactions with autograd.
  Compared to `NoGradMode`, code run under this mode gets better performance by disabling autograd.

  Please note that in some models, InferenceMode might not benefit performance and in fewer cases might impact performance negatively.

  To enable inference mode, use the configuration example below:

  ```proto
  parameters: {
    key: "INFERENCE_MODE"
    value: { string_value: "true" }
  }
  ```

* `DISABLE_CUDNN`:

  Boolean flag to disable the cuDNN library.
  By default, cuDNN is enabled.

  [cuDNN](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for deep neural networks.
  It provides highly tuned implementations for standard routines.

  Typically, models run with cuDNN enabled execute faster.
  However there are some exceptions where using cuDNN can be slower, cause higher memory usage, or result in errors.

  To disable cuDNN, use the configuration example below:

  ```proto
  parameters: {
    key: "DISABLE_CUDNN"
    value: { string_value: "true" }
  }
  ```

* `ENABLE_WEIGHT_SHARING`:

  Boolean flag to enable model instances on the same device to share weights.
  This optimization should not be used with stateful models.
  If not specified, weight sharing is disabled.

  To enable weight sharing, use the configuration example below:

  ```proto
  parameters: {
    key: "ENABLE_WEIGHT_SHARING"
    value: { string_value: "true" }
  }
  ```

* `ENABLE_CACHE_CLEANING`:

  Boolean flag to enable CUDA cache cleaning after each model execution.
  If not specified, cache cleaning is disabled.
  This flag has no effect if model is on CPU.

  Setting this flag to true will likely negatively impact the performance due to additional CUDA cache cleaning operation after each model execution.
  Therefore, you should only use this flag if you serve multiple models with Triton and encounter CUDA out-of-memory issues during model executions.

  To enable cleaning of the CUDA cache after every execution, use the configuration example below:

  ```proto
  parameters: {
    key: "ENABLE_CACHE_CLEANING"
    value: { string_value: "true" }
  }
  ```

* `INTER_OP_THREAD_COUNT`:

  PyTorch allows using multiple CPU threads during TorchScript model inference.
  One or more inference threads execute a model’s forward pass on the given inputs.
  Each inference thread invokes a JIT interpreter that executes the ops of a model inline, one by one.

  This parameter sets the size of this thread pool.
  The default value of this setting is the number of cpu cores.

  > [!TIP]
  > Refer to
  > [CPU Threading TorchScript](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
  > on how to set this parameter properly.

  To set the inter-op thread count, use the configuration example below:

  ```proto
  parameters: {
    key: "INTER_OP_THREAD_COUNT"
    value: { string_value: "1" }
  }
  ```

> [!NOTE]
> This parameter is set globally for the PyTorch backend.
> The value from the first model config file that specifies this parameter will be used.
> Subsequent values from other model config files, if different, will be ignored.

* `INTRA_OP_THREAD_COUNT`:

  In addition to the inter-op parallelism, PyTorch can also utilize multiple threads within the ops (intra-op parallelism).
  This can be useful in many cases, including element-wise ops on large tensors, convolutions, GEMMs, embedding lookups and others.

  The default value for this setting is the number of CPU cores.

  > [!TIP]
  > Refer to
  > [CPU Threading TorchScript](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
  > on how to set this parameter properly.

  To set the intra-op thread count, use the configuration example below:

  ```proto
  parameters: {
    key: "INTRA_OP_THREAD_COUNT"
    value: { string_value: "1" }
  }
  ```

* **Additional Optimizations**:

  Three additional boolean parameters are available to disable certain Torch optimizations that can sometimes cause latency regressions in models with complex execution modes and dynamic shapes.
  If not specified, all are enabled by default.

    `ENABLE_JIT_EXECUTOR`

    `ENABLE_JIT_PROFILING`

### Model Instance Group Kind

The PyTorch backend supports the following kinds of
[Model Instance Groups](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
where the input tensors are placed as follows:

* `KIND_GPU`:

  Inputs are prepared on the GPU device associated with the model instance.

* `KIND_CPU`:

  Inputs are prepared on the CPU.

* `KIND_MODEL`:

  Inputs are prepared on the CPU.
  When loading the model, the backend does not choose the GPU device for the model;
  instead, it respects the device(s) specified in the model and uses them as they are during inference.

  This is useful when the model internally utilizes multiple GPUs, as demonstrated in
  [this example model](https://github.com/triton-inference-server/server/blob/main/qa/L0_libtorch_instance_group_kind_model/gen_models.py).

  > [!IMPORTANT]
  > If a device is not specified in the model, the backend uses the first available GPU device.

To set the model instance group, use the configuration example below:

```proto
instance_group {
   count: 2
   kind: KIND_GPU
}
```

### Customization

The following PyTorch settings may be customized by setting parameters on the
`config.pbtxt`.

[`torch.set_num_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_threads.html#torch.set_num_threads)

* Key: `NUM_THREADS`
* Value: The number of threads used for intra-op parallelism on CPU.

[`torch.set_num_interop_threads(int)`](https://pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html#torch.set_num_interop_threads)

* Key: `NUM_INTEROP_THREADS`
* Value: The number of threads used for interop parallelism (e.g. in JIT interpreter) on CPU.

[`torch.compile()` parameters](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)

* Key: `TORCH_COMPILE_OPTIONAL_PARAMETERS`
* Value: Any of following parameter(s) encoded as a JSON object.
  * `fullgraph` (`bool`): Whether it is ok to break model into several subgraphs.
  * `dynamic` (`bool`): Use dynamic shape tracing.
  * `backend` (`str`): The backend to be used.
  * `mode` (`str`): Can be either `"default"`, `"reduce-overhead"`, or `"max-autotune"`.
  * `options` (`dict`): A dictionary of options to pass to the backend.
  * `disable` (`bool`): Turn `torch.compile()` into a no-op for testing.

For example:

```proto
parameters: {
  key: "NUM_THREADS"
  value: { string_value: "4" }
}
parameters: {
  key: "TORCH_COMPILE_OPTIONAL_PARAMETERS"
  value: { string_value: "{\"disable\": true}" }
}
```

## Important Notes

* The execution of PyTorch model on GPU is asynchronous in nature.
  See
  [CUDA Asynchronous Execution](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
  for additional details.
  Consequently, an error in PyTorch model execution may be raised during the next few inference requests to the server.
  Setting environment variable `CUDA_LAUNCH_BLOCKING=1` when launching server will help in correctly debugging failing cases by forcing synchronous execution.

  * The PyTorch model in such cases may or may not recover from the failed state and a restart of the server may be required to continue serving successfully.

* PyTorch does not support Tensor of Strings but it does support models that accept a List of Strings as input(s) / produces a List of String as output(s).
  For these models Triton allows users to pass String input(s)/receive String output(s) using the String datatype.
  As a limitation of using List instead of Tensor for String I/O, only for 1-dimensional input(s)/output(s) are supported for I/O of String type.

* In a multi-GPU environment, a potential runtime issue can occur when using
  [Tracing](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)
  to generate a
  [TorchScript](https://pytorch.org/docs/stable/jit.html)
  model.
  This issue arises due to a device mismatch between the model instance and the tensor.

  By default, Triton creates a single execution instance of the model for each available GPU.
  The runtime error occurs when a request is sent to a model instance with a different GPU device from the one used during the TorchScript generation process.

  To address this problem, it is highly recommended to use
  [Scripting](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script)
  instead of Tracing for model generation in a multi-GPU environment.
  Scripting avoids the device mismatch issue and ensures compatibility with different GPUs when used with Triton.

  However, if using Tracing is unavoidable, there is a workaround available.
  You can explicitly specify the GPU device for the model instance in the
  [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
  to ensure that the model instance and the tensors used for inference are assigned to the same GPU device as on which the model was traced.

* When using `KIND_MODEL` as model instance kind, the default device of the first parameter on the model is used.

> [!WARNING]
>
> * Python functions optimizable by `torch.compile` may not be served directly in the `model.py` file, they need to be enclosed by a class extending the
  [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
>
> * Model weights cannot be shared across multiple instances on the same GPU device.
