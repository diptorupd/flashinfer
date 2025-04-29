## Single Prefill Driver
A standalone CUDA driver program for FlashInfer's prefill attention kernel that allows direct testing without Python/PyTorch bindings.

## Overview
The Single Prefill Driver implements a self-contained executable that demonstrates FlashInfer's attention mechanism for prefill operations. It provides a C++ interface for running FlashInfer's single prefill attention kernels without requiring Python integration.

## Prerequisites
CUDA Toolkit (12.4+ recommended)
CMake (3.21+)
C++ compiler with C++20 support
FlashInfer library (installed separately)

## Installing Flashinfer C++ API

The example requires the C++ flashinfer CUDA headers. The included CMakeLists.txt included with the example assumes that a standalone installation of the C++ flashinfer header was previously done. An example of how to install the headers
is shown in the following snippet.

```bash
mkdir build
cd build
cmake .. -GNinja -DFLASHINFER_CUTLASS_DIR=../3rdparty/cutlass -DCMAKE_INSTALL_PREFIX=<your_install_path>
ninja install
```

## Configure and build the single_prefill_driver

```bash
cmake -B build -GNinja -DCMAKE_PREFIX_PATH=<your_install_path>
cd build
ninja single_prefill_driver
```

## Running the driver

```bash
./single_prefill_driver --qo_len 512 --kv_len 8192 --causal 1 --head_dim 128 --pos_encoding rope
```

Several run options are available to run the driver program.

```bash
--qo_len <int>: Query sequence length (default: 512)
--kv_len <int>: Key/value sequence length (default: 512)
--num_qo_heads <int>: Number of query heads (default: 32)
--num_kv_heads <int>: Number of key-value heads (default: 32)
--head_dim <int>: Head dimension (default: 128)
--layout <nhd|hnd>: KV tensor layout (default: nhd)
--pos_encoding <none|rope|alibi>: Position encoding mode (default: none)
--causal <0|1>: Use causal mask (default: 1)
--window_left <int>: Window left size (default: -1)
--rope_scale <float>: RoPE scale factor (default: 1.0)
--rope_theta <float>: RoPE theta (default: 10000.0)
--iterations <int>: Number of iterations for timing (default: 10)
--warmup <int>: Number of warmup iterations (default: 5)
```
