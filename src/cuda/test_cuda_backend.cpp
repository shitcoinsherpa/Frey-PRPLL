// Minimal test for the CUDA backend: compile a kernel via NVRTC, run it, verify.

#include "cudawrap.h"
#include <cstdio>
#include <cstring>
#include <vector>

// A tiny kernel to test: element-wise add of two arrays
static const char* testKernelSource = R"(
extern "C" __global__ void vecAdd(const double* a, const double* b, double* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
)";

// A kernel using M61 arithmetic from opencl_compat.cuh
static const char* m61TestSource = R"(
#include "opencl_compat.cuh"

typedef unsigned long long u64;
typedef unsigned long long Z61;
typedef struct { u64 x; u64 y; } GF61_t;

#define M61 ((((Z61)1) << 61) - 1)

__device__ Z61 addM61(Z61 a, Z61 b) {
  Z61 t = a + b;
  Z61 m = t - M61;
  return (m & 0x8000000000000000ULL) ? t : m;
}

__device__ Z61 mulM61(Z61 a, Z61 b) {
  // Use __umul64hi intrinsic (no __int128 needed)
  u64 lo = a * b;
  u64 hi = __umul64hi(a, b);
  u64 lo61 = lo & M61;
  u64 hi61 = (hi << 3) | (lo >> 61);
  return addM61(lo61, hi61);
}

extern "C" __global__ void m61_test(u64* out, const u64* a, const u64* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = mulM61(a[i], b[i]);
  }
}
)";

int main() {
  printf("=== CUDA Backend Test ===\n\n");

  // Initialize CUDA
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    printf("CUDA init failed: %d\n", (int)err);
    return 1;
  }

  auto devices = getAllDevices();
  if (devices.empty()) {
    printf("No CUDA devices found\n");
    return 1;
  }

  printf("Found %zu device(s):\n", devices.size());
  for (size_t i = 0; i < devices.size(); i++) {
    printf("  [%zu] %s\n", i, getShortInfo(devices[i]).c_str());
  }

  CudaContext ctx(devices[0]);
  printf("\nContext created on device 0\n");
  printf("Driver version: %s\n", getDriverVersion().c_str());

  // --- Test 1: Simple vector add ---
  printf("\n--- Test 1: Vector Add via NVRTC ---\n");
  {
    int computeMajor = 0, computeMinor = 0;
    cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[0]);
    cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[0]);

    char archOpt[32];
    snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=sm_%d%d", computeMajor, computeMinor);

    std::vector<std::string> opts = { archOpt, "-default-device" };
    std::string ptx = NvrtcProgram::compile(testKernelSource, "vecAdd.cu", opts);
    printf("PTX compiled: %zu bytes\n", ptx.size());

    CudaModule mod(ptx, "vecAdd");
    CUfunction func = mod.getFunction("vecAdd");
    printf("Kernel 'vecAdd' loaded\n");

    const int N = 1024;
    std::vector<double> ha(N), hb(N), hc(N);
    for (int i = 0; i < N; i++) { ha[i] = i * 1.0; hb[i] = i * 2.0; }

    CudaBuffer da(N * sizeof(double));
    CudaBuffer db(N * sizeof(double));
    CudaBuffer dc(N * sizeof(double));
    da.writeSync(ha.data(), N * sizeof(double));
    db.writeSync(hb.data(), N * sizeof(double));
    dc.zero();

    CudaStream stream;
    CUdeviceptr daPtrs[4];
    daPtrs[0] = da.get();
    daPtrs[1] = db.get();
    daPtrs[2] = dc.get();
    int n = N;

    void* args[] = { &daPtrs[0], &daPtrs[1], &daPtrs[2], &n };
    CU_CHECK(cuLaunchKernel(func, (N+255)/256, 1, 1, 256, 1, 1, 0, stream.get(), args, nullptr));
    stream.sync();

    dc.readSync(hc.data(), N * sizeof(double));

    bool ok = true;
    for (int i = 0; i < N; i++) {
      if (hc[i] != ha[i] + hb[i]) { printf("MISMATCH at %d: %f != %f\n", i, hc[i], ha[i]+hb[i]); ok = false; break; }
    }
    printf("Vector add: %s\n", ok ? "PASS" : "FAIL");
  }

  // --- Test 2: M61 arithmetic kernel with opencl_compat.cuh ---
  printf("\n--- Test 2: M61 Multiply via NVRTC + opencl_compat.cuh ---\n");
  {
    int computeMajor = 0, computeMinor = 0;
    cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[0]);
    cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[0]);

    char archOpt[32];
    snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=sm_%d%d", computeMajor, computeMinor);

    // Read the compat header to pass to NVRTC
    // For now, inline it as a header
    // Try multiple paths for the compat header
    const char* paths[] = { "src/cuda/opencl_compat.cuh", "../src/cuda/opencl_compat.cuh", nullptr };
    FILE* f = nullptr;
    for (int pi = 0; paths[pi]; pi++) { f = fopen(paths[pi], "r"); if (f) break; }
    if (!f) { printf("Cannot open opencl_compat.cuh\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string compatHeader(len, '\0');
    fread(compatHeader.data(), 1, len, f);
    fclose(f);

    std::vector<std::string> opts = { archOpt, "-default-device", "-std=c++17" };
    std::vector<std::pair<std::string, std::string>> headers = {
      {"opencl_compat.cuh", compatHeader}
    };

    std::string ptx = NvrtcProgram::compile(m61TestSource, "m61_test.cu", opts, headers);
    printf("M61 PTX compiled: %zu bytes\n", ptx.size());

    CudaModule mod(ptx, "m61_test");
    CUfunction func = mod.getFunction("m61_test");
    printf("Kernel 'm61_test' loaded\n");

    const int N = 16;
    const uint64_t M61 = (1ULL << 61) - 1;

    std::vector<uint64_t> ha(N), hb(N), hc(N);
    // Test values: small numbers where we can verify
    for (int i = 0; i < N; i++) {
      ha[i] = (i + 1) % M61;
      hb[i] = (i + 100) % M61;
    }

    CudaBuffer da(N * sizeof(uint64_t));
    CudaBuffer db(N * sizeof(uint64_t));
    CudaBuffer dc(N * sizeof(uint64_t));
    da.writeSync(ha.data(), N * sizeof(uint64_t));
    db.writeSync(hb.data(), N * sizeof(uint64_t));

    CudaStream stream;
    CUdeviceptr daPtrs[3];
    daPtrs[0] = dc.get();
    daPtrs[1] = da.get();
    daPtrs[2] = db.get();
    int n = N;
    void* args[] = { &daPtrs[0], &daPtrs[1], &daPtrs[2], &n };
    CU_CHECK(cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, stream.get(), args, nullptr));
    stream.sync();

    dc.readSync(hc.data(), N * sizeof(uint64_t));

    bool ok = true;
    for (int i = 0; i < N; i++) {
      // Compute expected: (a * b) mod M61
      unsigned __int128 prod = (unsigned __int128)ha[i] * hb[i];
      uint64_t lo = (uint64_t)prod;
      uint64_t hi = (uint64_t)(prod >> 64);
      uint64_t lo61 = lo & M61;
      uint64_t hi61 = (hi << 3) | (lo >> 61);
      uint64_t expected = lo61 + hi61;
      if (expected >= M61) expected -= M61;
      if (expected >= M61) expected -= M61;

      if (hc[i] != expected) {
        printf("MISMATCH at %d: got %llu, expected %llu (a=%llu, b=%llu)\n",
               i, (unsigned long long)hc[i], (unsigned long long)expected,
               (unsigned long long)ha[i], (unsigned long long)hb[i]);
        ok = false;
        break;
      }
    }
    printf("M61 multiply: %s\n", ok ? "PASS" : "FAIL");
  }

  printf("\n=== All tests complete ===\n");
  return 0;
}
