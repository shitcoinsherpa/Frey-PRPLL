// Minimal CUDA test — just compile a kernel and list exported functions

#include "cudawrap.h"
#include <cstdio>
#include <cstring>

static const char* testKernelSource = R"(
extern "C" __global__ void vecAdd(const double* a, const double* b, double* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
)";

int main() {
  fprintf(stderr, "=== Simple CUDA Test ===\n");
  fflush(stderr);

  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "CUDA init failed: %d\n", (int)err);
    return 1;
  }
  fprintf(stderr, "CUDA init OK\n");

  auto devices = getAllDevices();
  if (devices.empty()) {
    fprintf(stderr, "No CUDA devices\n");
    return 1;
  }
  fprintf(stderr, "Device: %s\n", getShortInfo(devices[0]).c_str());

  CudaContext ctx(devices[0]);
  fprintf(stderr, "Context created\n");

  int computeMajor = 0, computeMinor = 0;
  cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[0]);
  cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[0]);
  fprintf(stderr, "Compute: sm_%d%d\n", computeMajor, computeMinor);

  char archOpt[32];
  snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=sm_%d%d", computeMajor, computeMinor);

  std::vector<std::string> opts = { archOpt };
  fprintf(stderr, "Compiling with: %s\n", archOpt);

  std::string ptx = NvrtcProgram::compile(testKernelSource, "vecAdd.cu", opts);
  fprintf(stderr, "PTX compiled: %zu bytes\n", ptx.size());

  // Print first 500 chars of PTX to see what's in it
  fprintf(stderr, "PTX preview:\n%.500s\n...\n", ptx.c_str());

  CudaModule mod(ptx, "vecAdd");
  fprintf(stderr, "Module loaded\n");

  CUfunction func = mod.getFunction("vecAdd");
  fprintf(stderr, "Kernel loaded: %p\n", (void*)func);

  // Quick test
  const int N = 256;
  double ha[256], hb[256], hc[256];
  for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = i * 2.0; }

  CudaBuffer da(N * sizeof(double));
  CudaBuffer db(N * sizeof(double));
  CudaBuffer dc(N * sizeof(double));
  da.writeSync(ha, N * sizeof(double));
  db.writeSync(hb, N * sizeof(double));

  CudaStream stream;
  CUdeviceptr ptrs[3] = { da.get(), db.get(), dc.get() };
  int n = N;
  void* args[] = { &ptrs[0], &ptrs[1], &ptrs[2], &n };
  CU_CHECK(cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, stream.get(), args, nullptr));
  stream.sync();

  dc.readSync(hc, N * sizeof(double));

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (hc[i] != ha[i] + hb[i]) {
      fprintf(stderr, "MISMATCH at %d: %f != %f\n", i, hc[i], ha[i]+hb[i]);
      ok = false;
      break;
    }
  }
  fprintf(stderr, "Vector add: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
