
add_executable(frey-prpll
  Trig.cpp
  Primes.cpp
  bundle.cpp
  Proof.cpp
  log.cpp md5.cpp sha3.cpp AllocTrac.cpp FFTConfig.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp
  File.cpp
  gpuid.cpp
  version.cpp
  KernelCompiler.cpp
  Kernel.cpp
  Saver.cpp
  Queue.cpp
  TimeInfo.cpp
  Profile.cpp
  Event.cpp
  TrigBufCache.cpp
  CycleFile.cpp
  tune.cpp
  TuneEntry.cpp
  fs.cpp
  version.inc
  )

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Real GCC (not clang), needed for 128-bit floats and std::filesystem::path
  set(CMAKE_CXX_COMPILER g++-14)
endif()

option(USE_CUDA "Use CUDA backend instead of OpenCL" OFF)

if (USE_CUDA)
  # CUDA backend: replace OpenCL with our CUDA Driver API shim
  find_package(CUDAToolkit REQUIRED)
  target_sources(frey-prpll PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clwrap_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cudawrap.cpp
  )
  target_include_directories(frey-prpll PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  target_link_libraries(frey-prpll CUDA::cuda_driver CUDA::nvrtc)
  # Prevent linking against the real OpenCL library
  target_compile_definitions(frey-prpll PRIVATE USE_CUDA=1 CUDA_BACKEND=1)
  message(STATUS "    USE_CUDA")
else()
  find_package(OpenCL)
  if (${OpenCL_FOUND})
    target_link_libraries(frey-prpll OpenCL::OpenCL)
  else()
    target_link_libraries(frey-prpll OpenCL)
  endif()
endif()

target_include_directories(frey-prpll PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
  OUTPUT version.inc
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git describe --tags --long --dirty --always --match ${CMAKE_SOURCE_DIR} | sed 's/.*/"&"/' > ${CMAKE_CURRENT_BINARY_DIR}/version.inc
  DEPENDS ${CMAKE_SOURCE_DIR}
  )

file(
  GLOB CL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cl/*.cl
  )

add_custom_command(
  OUTPUT bundle.cpp
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_SOURCE_DIR}/genbundle.sh ${CL_SOURCES} > ${CMAKE_CURRENT_BINARY_DIR}/bundle.cpp
  DEPENDS ${CL_SOURCES}
  )
