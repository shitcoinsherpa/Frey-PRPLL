# Frey-PRPLL

**Native CUDA fork of [PRPLL/gpuowl](https://github.com/preda/gpuowl) for Mersenne prime testing on NVIDIA GPUs.**

PRPLL (by Mihai Preda, with contributions from George Woltman) is the leading GPU program for [GIMPS](https://www.mersenne.org/) Mersenne prime search, but targets AMD GPUs via OpenCL. Frey-PRPLL replaces the OpenCL backend with native CUDA, using the CUDA Driver API and NVRTC for runtime kernel compilation. The existing `.cl` kernel sources are compiled directly by NVRTC through a compatibility layer — no manual kernel rewrites needed.

## Performance

Tested on **NVIDIA GeForce RTX 4090** (Ada Lovelace, sm_89, 24GB VRAM):

| Exponent | FFT Size | bpw | µs/iter | iter/s | Notes |
|----------|----------|------|---------|--------|-------|
| 136M | 4M | 32.49 | 264 | 3,788 | tuned, NTT GF31+GF61 |

The 136M result uses FFT config `3:512:16:256:202` (NTT over GF(2^31-1) x GF(2^61-1)), found via `-tune`. For comparison, the best documented OpenCL PRPLL result on RTX 4090 for this exponent range is ~425 µs/iter ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/integer-ntt-on-rtx-20xx-a100-vs-rtx-30xx-40xx-50xx/350124)).

Other NVIDIA GPUs with CUDA 12.0+ should work but have not been benchmarked yet. Performance will scale with GPU compute capability and memory bandwidth. Use `-tune` to find the optimal FFT configuration for your hardware.

## How It Works

The CUDA backend lives in `src/cuda/` and consists of:

- **`tinycuda.h`** — Maps OpenCL types (`cl_mem`, `cl_kernel`, etc.) to CUDA Driver API equivalents
- **`clwrap_cuda.cpp`** — Implements all `cl*()` API functions via `cu*()` calls
- **`opencl_compat.cuh`** — NVRTC header that maps OpenCL kernel builtins (`get_local_id`, `barrier`, vector types, atomics) to CUDA equivalents
- **`cudawrap.{h,cpp}`** — RAII wrappers for CUDA contexts, modules, streams, buffers, and NVRTC compilation

The OpenCL `.cl` kernels are preprocessed at runtime to handle syntax differences (vector casts, `__global`/`__local` qualifiers, `__attribute__` directives) and then compiled via NVRTC with `opencl_compat.cuh` injected as a virtual header.

### Key CUDA optimizations

- **`NTLOAD` via `__ldg`**: Read-only twiddle factor tables loaded through the texture cache, freeing L1 for working data
- **`CUDA_MIN_BLOCKS=3`** on carryFused and GF61 middle kernels: Reduces register pressure, improves SM occupancy from 33% to 50%
- **`CU_STREAM_NON_BLOCKING`** with stream-ordered async copies: Correct async behavior without implicit synchronization
- **Kernel caching**: Compiled PTX cached to disk, skipping NVRTC on subsequent runs (`-cache` flag)
- **`FFT_VARIANT` auto-fix**: AMD-only FFT variants (using `__builtin_amdgcn_*`) automatically converted to portable variants for NVIDIA

## Build

Requires: CUDA Toolkit 12.0+, CMake 3.18+, C++20 compiler (GCC 11+ or Clang 14+).

```bash
# Linux / WSL2
mkdir build && cd build
cmake .. -DCUDA_BACKEND=ON
make -j$(nproc)

# Binary: build/src/frey-prpll
```

## Usage

```bash
# Create a work directory (checkpoints and proofs are saved here)
mkdir work && cd work

# Add an exponent to test
echo "PRP=N/A,1,2,136279841,-1,99,0,3,4" > worktodo-0.txt

# Run on GPU 0
../build/src/frey-prpll -d 0
```

See `frey-prpll -h` for full options including `-workers`, `-pool`, `-tune`, `-cache`, and FFT selection.

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

Frey-PRPLL is a fork of [PRPLL](https://github.com/preda/gpuowl) by **Mihai Preda**, with FFT and carry code contributions by **George Woltman**. All original copyright notices are preserved in the source files. The CUDA backend (`src/cuda/`) is new code added by this fork.

### References

- [Discrete Weighted Transforms and Large Integer Arithmetic](https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1185244-1/S0025-5718-1994-1185244-1.pdf) — Crandall & Fagin, 1994
- [Rapid Multiplication Modulo the Sum And Difference of Highly Composite Numbers](https://www.daemonology.net/papers/fft.pdf) — Colin Percival, 2002
- [An FFT Extension to the P-1 Factoring Algorithm](https://www.ams.org/journals/mcom/1990-54-190/S0025-5718-1990-1011444-3/S0025-5718-1990-1011444-3.pdf) — Montgomery & Silverman, 1990
