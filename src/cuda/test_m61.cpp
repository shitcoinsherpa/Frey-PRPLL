// Test M61 kernel compilation via NVRTC with opencl_compat.cuh

#include "cudawrap.h"
#include <cstdio>
#include <cstring>

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
  fprintf(stderr, "=== M61 CUDA Test ===\n");

  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) { fprintf(stderr, "CUDA init failed\n"); return 1; }

  auto devices = getAllDevices();
  if (devices.empty()) { fprintf(stderr, "No devices\n"); return 1; }
  fprintf(stderr, "Device: %s\n", getShortInfo(devices[0]).c_str());

  CudaContext ctx(devices[0]);

  int computeMajor = 0, computeMinor = 0;
  cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[0]);
  cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[0]);

  char archOpt[32];
  snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=sm_%d%d", computeMajor, computeMinor);

  // Read compat header
  const char* paths[] = {
    "src/cuda/opencl_compat.cuh",
    "../src/cuda/opencl_compat.cuh",
    nullptr
  };
  FILE* f = nullptr;
  const char* usedPath = nullptr;
  for (int pi = 0; paths[pi]; pi++) {
    f = fopen(paths[pi], "r");
    if (f) { usedPath = paths[pi]; break; }
  }
  if (!f) { fprintf(stderr, "Cannot open opencl_compat.cuh\n"); return 1; }
  fprintf(stderr, "Opened: %s\n", usedPath);

  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string compatHeader(len, '\0');
  size_t rd = fread(compatHeader.data(), 1, len, f);
  fclose(f);
  fprintf(stderr, "Header: %ld bytes read (%zu)\n", len, rd);

  std::vector<std::string> opts = { archOpt, "-default-device", "-std=c++17" };
  std::vector<std::pair<std::string, std::string>> headers = {
    {"opencl_compat.cuh", compatHeader}
  };

  fprintf(stderr, "Preprocessing OpenCL source...\n");
  std::string processedSource = NvrtcProgram::preprocessOpenCL(m61TestSource);
  // Also preprocess the header
  std::vector<std::pair<std::string, std::string>> processedHeaders;
  for (auto& [name, src] : headers) {
    processedHeaders.push_back({name, NvrtcProgram::preprocessOpenCL(src)});
  }

  fprintf(stderr, "Compiling M61 kernel...\n");
  std::string ptx = NvrtcProgram::compile(processedSource, "m61_test.cu", opts, processedHeaders);
  fprintf(stderr, "PTX: %zu bytes\n", ptx.size());

  // Dump full PTX to file
  FILE* ptxFile = fopen("build-wsl/m61_test.ptx", "w");
  if (ptxFile) { fwrite(ptx.data(), 1, ptx.size(), ptxFile); fclose(ptxFile); }
  fprintf(stderr, "PTX written to build-wsl/m61_test.ptx\n");

  // Check if m61_test appears in PTX
  if (ptx.find("m61_test") != std::string::npos) {
    fprintf(stderr, "GOOD: 'm61_test' found in PTX\n");
  } else {
    fprintf(stderr, "BAD: 'm61_test' NOT found in PTX\n");
  }

  CudaModule mod(ptx, "m61_test");
  fprintf(stderr, "Module loaded\n");

  CUfunction func = mod.getFunction("m61_test");
  fprintf(stderr, "Kernel loaded: %p\n", (void*)func);

  // Run test
  const int N = 16;
  const uint64_t M61 = (1ULL << 61) - 1;

  std::vector<uint64_t> ha(N), hb(N), hc(N);
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
  CUdeviceptr daPtrs[3] = { dc.get(), da.get(), db.get() };
  int n = N;
  void* args[] = { &daPtrs[0], &daPtrs[1], &daPtrs[2], &n };
  CU_CHECK(cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, stream.get(), args, nullptr));
  stream.sync();

  dc.readSync(hc.data(), N * sizeof(uint64_t));

  bool ok = true;
  for (int i = 0; i < N; i++) {
    unsigned __int128 prod = (unsigned __int128)ha[i] * hb[i];
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t lo61 = lo & M61;
    uint64_t hi61 = (hi << 3) | (lo >> 61);
    uint64_t expected = lo61 + hi61;
    if (expected >= M61) expected -= M61;
    if (expected >= M61) expected -= M61;

    if (hc[i] != expected) {
      fprintf(stderr, "MISMATCH at %d: got %llu, expected %llu\n",
              i, (unsigned long long)hc[i], (unsigned long long)expected);
      ok = false;
      break;
    }
  }
  fprintf(stderr, "M61 multiply: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
