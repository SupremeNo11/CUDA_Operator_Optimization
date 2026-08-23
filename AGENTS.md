# AGENTS.md

## Project Overview

CUDA operator optimization project. Each operator is a self-contained CMake project under a numbered directory. Only `01_GEMM` is implemented; directories `02`–`07` are empty placeholders for future operators.

## Build & Run

Each operator builds independently from its own directory:

```bash
cd 01_GEMM
mkdir build && cd build
cmake ..
make
./gemm_optimization
```

There is no top-level CMakeLists.txt. Build from within the operator directory.

### Profiling

```bash
nvprof ./build/gemm_optimization
```

Profiling results and performance analysis are recorded in `01_GEMM/OptimizationLog.md`.

## Architecture

- **`utils/`** — shared CUDA utilities, globbed into every operator's build via `../utils/src/*.cu` and `../utils/src/*.cpp` (relative path in each CMakeLists.txt).
  - `utils/inc/cuda_common.h` — error macros, `SYS_STATUS` enum, device query helpers
  - `utils/src/cuda_common.cu` — matrix initialization
  - `utils/src/deviceQuery.cpp` — GPU device info printer
- **Per-operator files** follow the pattern:
  - `*_optimization.cu` — entry point (`main()`)
  - `*_gpu.cu` — CUDA kernels + GPU orchestration (`cudaMalloc`, kernel launch, `cudaMemcpy`, timing)
  - `*_cpu.cpp` — CPU reference implementation
  - `*_common.h` — shared function declarations (includes `utils/inc/cuda_common.h`)

## Key Conventions

- **GPU architecture is hardcoded**: `CMAKE_CUDA_ARCHITECTURES` is set to `61` (Pascal, GTX 1050 Ti). Change this in the operator's `CMakeLists.txt` when targeting a different GPU.
- **C++ standard**: gnu++17.
- **Error handling**: two macros with different behavior:
  - `CUDA_CHECK(expr)` — prints error to stderr, **does not exit** (non-fatal)
  - `checkCudaErrors(val)` — prints error and **calls `exit(EXIT_FAILURE)`** (fatal)
- **Return status**: functions return `SYS_STATUS` enum (`SYS_STATUS_SUCCESS` = 0, `SYS_STATUS_CALL_FAIL` = 1).
- **Host memory**: uses `new`/`delete[]` (pageable memory, not pinned). The OptimizationLog notes this causes hidden overhead in `cudaMemcpy`.
- **Timing**: CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`), output in microseconds.
- **CPU path toggle**: `#if _ENABLE_CPU` conditionally compiles the CPU reference path. The definition is commented out in CMakeLists.txt — uncomment `target_compile_definitions` to enable.
- **Matrix layout**: row-major, manual index computation (`row * width + col`).
- **Block size**: defined as macro `BS` (currently 32) in `*_gpu.cu`.
- **Comments and documentation are in Chinese** (中文). Maintain this convention.

## Adding a New Operator

1. Create `NN_OperatorName/` with the file pattern above.
2. Write a `CMakeLists.txt` modeled on `01_GEMM/CMakeLists.txt` — set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability.
3. Include `../utils/src/*.cu` and `../utils/src/*.cpp` in the source list.
4. Include `../utils/inc/cuda_common.h` from your operator's `*_common.h`.
5. Create an `OptimizationLog.md` to record profiling results and analysis.

## Verification

No test framework. Correctness is verified inline by comparing GPU results against CPU reference with a tolerance (`fabs(cpu - gpu) > 0.0001`). Keep this pattern when adding operators.
