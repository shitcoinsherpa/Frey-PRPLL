// Copyright (C) Mihai Preda

#include "Args.h"
#include "Background.h"
#include "Queue.h"
#include "Signal.h"
#include "Task.h"
#include "Worktodo.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"
#include "log.h"
#include "Context.h"
#include "TrigBufCache.h"
#include "GpuCommon.h"
#include "Gpu.h"
#include "tune.h"

#include <filesystem>
#include <thread>
// #include <format> from GCC-13 onwards

void gpuWorker(GpuCommon shared, Queue *q, i32 instance) {
  if (instance > 0) {
    initLog(("frey-prpll-"s + to_string(instance) + ".log").c_str());
    log("Frey-PRPLL %s, instance %d\n", VERSION, instance);
  }

  try {
    while (auto task = Worktodo::getTask(*shared.args, instance)) {
      constexpr int MAX_ALLOC_RETRIES = 10;
      constexpr int RETRY_DELAY_SEC = 15;

      for (int attempt = 0; ; ++attempt) {
        try {
          task->execute(shared, q, instance);
          break;  // success
        } catch (const std::bad_alloc&) {
          if (attempt >= MAX_ALLOC_RETRIES) {
            log("Worker %d: GPU memory allocation failed after %d retries. "
                "Not enough VRAM for %d concurrent workers at this FFT size. "
                "Try -workers 1, a smaller exponent, or a GPU with more VRAM.\n",
                instance, MAX_ALLOC_RETRIES, shared.args->workers);
            return;
          }
          log("Worker %d: GPU memory allocation failed (attempt %d/%d, %.1f GB in use). "
              "Waiting %ds for another worker to free memory...\n",
              instance, attempt + 1, MAX_ALLOC_RETRIES,
              float(AllocTrac::totalAllocBytes()) / (1024 * 1024 * 1024), RETRY_DELAY_SEC);
          std::this_thread::sleep_for(std::chrono::seconds(RETRY_DELAY_SEC));
        }
      }
    }
  } catch (const char *mes) {
    log("Exception \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exception \"%s\"\n", mes.c_str());
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  }
}


#if defined(__MINGW32__) || defined(__MINGW64__) || defined(__MSYS__) // for Windows
extern int putenv(char *);
#endif

int main(int argc, char **argv) {

#if defined(__MSYS__)
  // I was unable to get putenv to link in MSYS2
#elif defined(__MINGW32__) || defined(__MINGW64__)
  putenv("ROC_SIGNAL_POOL_SIZE=32");
#else
  // Required to work around a ROCm bug when using multiple queues
  setenv("ROC_SIGNAL_POOL_SIZE", "32", 0);
#endif

  int exitCode = 0;

  try {
    string mainLine = Args::mergeArgs(argc, argv);
    {
      Args args{true};
      args.parse(mainLine);
      if (!args.dir.empty()) {
        fs::current_path(args.dir);
      }
    }

    fs::path poolDir;
    {
      Args args{true};
      args.readConfig("config.txt");
      args.parse(mainLine);
      poolDir = args.masterDir;
    }

    initLog("frey-prpll.log");
    log("Frey-PRPLL %s starting\n", VERSION);

    Args args;

    if (!poolDir.empty()) { args.readConfig(poolDir / "config.txt"); }
    args.readConfig("config.txt");
    args.parse(mainLine);
    args.setDefaults();

    Context context(getDevice(args.device));

    // Set GPU memory budget: user-specified -maxAlloc, or auto-detect from VRAM
    {
      float gpuRamGB = getGpuRamGB(context.deviceId());
      if (args.maxAlloc) {
        AllocTrac::setMaxAlloc(args.maxAlloc);
        log("GPU memory budget: %.1f GB (user-specified), GPU has %.1f GB VRAM\n",
            float(args.maxAlloc) / (1024 * 1024 * 1024), gpuRamGB);
      } else if (gpuRamGB > 0) {
        // Use 90% of VRAM as the allocation limit (leave headroom for driver/OS)
        size_t autoMax = size_t(gpuRamGB * 0.9f * 1024) * 1024 * 1024;
        AllocTrac::setMaxAlloc(autoMax);
        log("GPU memory budget: %.1f GB (auto-detected from %.1f GB VRAM)\n",
            float(autoMax) / (1024 * 1024 * 1024), gpuRamGB);
      } else {
        log("GPU memory budget: %.1f GB (default, VRAM detection unavailable)\n",
            float(15ULL * 1024 * 1024 * 1024) / (1024 * 1024 * 1024));
      }
    }
    Signal signal;
    Background background;
    GpuCommon shared;
    shared.args = &args;
    TrigBufCache bufCache{&context};
    shared.bufCache = &bufCache;
    shared.background = &background;

    if (args.doCtune || args.doTune || args.doZtune || args.carryTune) {
      Queue q(context, args.profile);
      Tune tune{&q, shared};

      if (args.doCtune) {
        tune.ctune();
      } else if (args.doTune) {
        tune.tune();
      } else if (args.doZtune) {
        tune.ztune();
      } else if (args.carryTune) {
        tune.carryTune();
      }
    } else {
      {
        vector<Queue> queues;
        for (int i = 0; i < int(args.workers); ++i) { queues.emplace_back(context, args.profile); }
        vector<jthread> threads;
        for (int i = 1; i < int(args.workers); ++i) {
          threads.emplace_back(gpuWorker, shared, &queues[i], i);
        }
        gpuWorker(shared, &queues[0], 0);
      }

      // log("No more work. Add work to worktodo.txt , see -h for details.\n");
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
  }

  log("Bye\n");
  return exitCode; // not used yet.
}
