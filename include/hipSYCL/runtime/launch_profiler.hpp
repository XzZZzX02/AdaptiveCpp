/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the root directory of this project.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_LAUNCH_PROFILER_HPP
#define HIPSYCL_LAUNCH_PROFILER_HPP

// Lightweight, opt-in per-launch phase timer used by the kernel-launch
// optimization work. Enable with ACPP_LT_PROFILE=1; results are printed at
// process exit (or written to $ACPP_LT_PROFILE_OUT).
//
// The recording functions are exported from libacpp-rt with default
// visibility so that they can also be called from the header-only SYCL glue
// code that gets compiled into user applications.

#include <cstdint>
#include <chrono>

#if defined(__GNUC__) || defined(__clang__)
#define ACPP_LT_PROFILE_EXPORT __attribute__((visibility("default")))
#else
#define ACPP_LT_PROFILE_EXPORT
#endif

extern "C" {
ACPP_LT_PROFILE_EXPORT int acpp_lp_enabled();
ACPP_LT_PROFILE_EXPORT void acpp_lp_record(int slot, unsigned long long ns);
ACPP_LT_PROFILE_EXPORT void acpp_lp_increment(int slot);
}

namespace hipsycl {
namespace rt {
namespace launch_profiler {

enum class slot : int {
  QueueSubmitTotal = 0,
  ExecuteSubmission,
  CreateTask,
  ExecutorSubmit,
  InsertEvent,
  EventPoolObtain,
  HipEventRecord,
  EventMakeShared,
  HipSubmitKernel,
  SscpArgMap,
  SscpConfigBuild,
  SscpConfigFinalize,
  SscpCodeObjectLookup,
  SscpLaunchModule,
  HipGetFunction,
  HipLaunchKernel,
  KernelCacheGet,
  SscpPreparedLookup,
  InstantSubmissionCount,
  TraditionalSubmissionCount,
  NumSlots
};

inline bool enabled() noexcept { return acpp_lp_enabled() != 0; }

inline void increment(slot s) noexcept {
  acpp_lp_increment(static_cast<int>(s));
}

inline void record(slot s, unsigned long long ns) noexcept {
  acpp_lp_record(static_cast<int>(s), ns);
}

class scope {
public:
  explicit scope(slot s) noexcept : _slot{s}, _enabled{enabled()} {
    if (_enabled)
      _start = now();
  }
  scope(const scope &) = delete;
  scope &operator=(const scope &) = delete;
  ~scope() noexcept {
    if (_enabled)
      record(_slot, now() - _start);
  }

private:
  static unsigned long long now() noexcept {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  slot _slot;
  bool _enabled;
  unsigned long long _start = 0;
};

} // namespace launch_profiler
} // namespace rt
} // namespace hipsycl

#endif
