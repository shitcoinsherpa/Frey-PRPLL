// Test: compile PRPLL's actual .cl kernel files through NVRTC via opencl_compat.cuh
// This validates the compatibility layer works with real production kernel code.

#include "cudawrap.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

namespace fs = std::filesystem;

static std::string readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); return ""; }
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  fprintf(stderr, "=== Frey-PRPLL .cl Kernel Compilation Test ===\n\n");

  // Find the cl/ directory
  std::string clDir;
  for (const char* p : {"src/cl", "../src/cl"}) {
    if (fs::is_directory(p)) { clDir = p; break; }
  }
  if (clDir.empty()) { fprintf(stderr, "Cannot find src/cl/\n"); return 1; }
  fprintf(stderr, "CL dir: %s\n", clDir.c_str());

  // Read all .cl files
  std::map<std::string, std::string> clFiles;
  for (auto& entry : fs::directory_iterator(clDir)) {
    if (entry.path().extension() == ".cl") {
      std::string name = entry.path().filename().string();
      clFiles[name] = readFile(entry.path().string());
      fprintf(stderr, "  Loaded: %s (%zu bytes)\n", name.c_str(), clFiles[name].size());
    }
  }
  fprintf(stderr, "Loaded %zu .cl files\n\n", clFiles.size());

  // Read opencl_compat.cuh
  std::string compatPath;
  for (const char* p : {"src/cuda/opencl_compat.cuh", "../src/cuda/opencl_compat.cuh"}) {
    if (fs::exists(p)) { compatPath = p; break; }
  }
  if (compatPath.empty()) { fprintf(stderr, "Cannot find opencl_compat.cuh\n"); return 1; }
  std::string compatHeader = readFile(compatPath);
  fprintf(stderr, "Compat header: %s (%zu bytes)\n\n", compatPath.c_str(), compatHeader.size());

  // Init CUDA
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) { fprintf(stderr, "CUDA init failed\n"); return 1; }

  auto devices = getAllDevices();
  if (devices.empty()) { fprintf(stderr, "No devices\n"); return 1; }
  fprintf(stderr, "Device: %s\n\n", getShortInfo(devices[0]).c_str());

  CudaContext ctx(devices[0]);

  int computeMajor = 0, computeMinor = 0;
  cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[0]);
  cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[0]);

  char archOpt[32];
  snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=sm_%d%d", computeMajor, computeMinor);

  // Preprocess all .cl sources (but NOT the compat header — it's already CUDA-ready)
  fprintf(stderr, "Preprocessing OpenCL sources...\n");
  std::map<std::string, std::string> processedCl;
  for (auto& [name, src] : clFiles) {
    processedCl[name] = NvrtcProgram::preprocessOpenCL(src);
  }

  // Build NVRTC headers: opencl_compat.cuh (unmodified) + all preprocessed .cl files
  std::vector<std::pair<std::string, std::string>> headers;
  headers.push_back({"opencl_compat.cuh", compatHeader});
  for (auto& [name, src] : processedCl) {
    headers.push_back({name, src});
  }

  // Standard compile options
  std::vector<std::string> opts = {
    archOpt,
    "-default-device",
    "-std=c++17",
    // Required PRPLL macros for base.cl
    "-DEXP=61",
    "-DWIDTH=256",
    "-DSMALL_HEIGHT=256",
    "-DMIDDLE=1",
    "-DCARRY_LEN=8",
    "-DNW=4",
    "-DNH=4",
    "-DNVIDIAGPU=1",
    "-DHAS_PTX=1200",
    "-DAMDGPU=0",
    "-DCARRY64=1",
    "-DFFT_TYPE=0",
    "-DFFT_FP64=1",
    "-DFFT_FP32=0",
    "-DNTT_GF31=0",
    "-DNTT_GF61=0",
    "-DWordSize=8",
    "-DMAXBPW=1800",
    "-DWEIGHT_STEP=1.0",
    "-DIWEIGHT_STEP=1.0",
    "-DTAILT=U2(0.0,0.0)",
    "-DFFT_VARIANT=111",
    "-DFFT_VARIANT_W=1",
    "-DFFT_VARIANT_M=1",
    "-DFFT_VARIANT_H=1",
    "-DFFTP_STRIDE_PHI=1",
    "-DFFTP_NOOP=0",
    "-DFFTP_CARRY=0",
    "-DFFTP_OUT_PHI=0",
    "-DDISTGF31=0",
    "-DDISTGF61=0",
    "-DDISTF64=0",
    "-DDISTWF64=0",
    "-DDISTMF64=0",
    "-DDISTHF64=0",
    "-DDISTF32=0",
    "-DDISTWF32=0",
    "-DDISTMF32=0",
    "-DDISTHF32=0",
    "-DDISTWTRIGGF31=0",
    "-DDISTMTRIGGF31=0",
    "-DDISTHTRIGGF31=0",
    // Trig coefficients (placeholder values for compilation test)
    "-DTRIG_SCALE=1",
    "-DTRIG_SIN={0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}",
    "-DTRIG_COS={1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}",
    // Fractional bits-per-word for carry (placeholder)
    "-DFRAC_BPW_HI=0x3C000000",
    "-DFRAC_BPW_LO=0x00000000",
    "-DBIGLIT=1",
    // Workgroup/wavefront parameters
    "-DWAVEFRONT=32",
    "-DFAST_BARRIER=0",
    "-DG_W=256",
    // Suppress assert redefinition and other harmless warnings
    "-w",
  };

  // Test ALL .cl files
  int passCount = 0, failCount = 0;
  for (auto& [name, src] : clFiles) {
    if (name == "base.cl") continue;  // base.cl is always included as part of the wrapper
    if (name == "carryinc.cl") continue;  // fragment file #include'd by carryutil.cl, cannot compile standalone
    fprintf(stderr, "--- Compiling %s ---\n", name.c_str());

    // Create wrapper: compat header + base.cl + math.cl (for X2/mul_t4/SWAP) + target
    // math.cl is always pre-included to ensure FFT primitives are available
    // before fft4.cl/fft8.cl which need them. #pragma once prevents double inclusion.
    std::string wrapperSource = std::string(
      "#include \"opencl_compat.cuh\"\n"
      "#include \"base.cl\"\n"
      "#include \"math.cl\"\n");
    if (name != "math.cl") {
      wrapperSource += "#include \"" + name + "\"\n";
    }
    wrapperSource +=
      "// Dummy kernel to ensure something is emitted\n"
      "extern \"C\" __global__ void dummy() {}\n";

    // Preprocess the wrapper too
    wrapperSource = NvrtcProgram::preprocessOpenCL(wrapperSource);

    try {
      std::string compileName = "test_" + name;
      std::string ptx = NvrtcProgram::compile(wrapperSource, compileName.c_str(), opts, headers);
      fprintf(stderr, "  PASS: %s compiled to %zu bytes PTX\n\n", name.c_str(), ptx.size());
      passCount++;
    } catch (const std::exception& e) {
      fprintf(stderr, "  FAIL: %s: %s\n\n", name.c_str(), e.what());
      failCount++;
    }
  }

  fprintf(stderr, "=== Results: %d PASS, %d FAIL out of %d files ===\n",
          passCount, failCount, passCount + failCount);
  return failCount > 0 ? 1 : 0;
}
