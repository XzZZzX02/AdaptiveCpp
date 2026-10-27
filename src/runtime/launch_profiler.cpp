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
#include "hipSYCL/runtime/launch_profiler.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

constexpr int num_slots =
    static_cast<int>(hipsycl::rt::launch_profiler::slot::NumSlots);

struct slot_data {
  std::atomic<unsigned long long> count{0};
  std::atomic<unsigned long long> total_ns{0};
  std::atomic<unsigned long long> max_ns{0};
};

slot_data g_data[num_slots];

bool g_enabled = []() {
  const char *value = std::getenv("ACPP_LT_PROFILE");
  return value && value[0] == '1';
}();

const char *slot_names[num_slots] = {
    "queue_submit_total",   "execute_submission",   "create_task",
    "executor_submit",      "insert_event",         "event_pool_obtain",
    "hip_event_record",     "event_make_shared",    "hip_submit_kernel",
    "sscp_arg_map",         "sscp_config_build",    "sscp_config_finalize",
    "sscp_code_object_get", "sscp_launch_module",   "hip_get_function",
    "hip_launch_kernel",    "kernel_cache_get",     "instant_submissions",
    "traditional_submissions"};

struct report_writer {
  ~report_writer() {
    if (!g_enabled)
      return;
    const char *path = std::getenv("ACPP_LT_PROFILE_OUT");
    FILE *out = path ? std::fopen(path, "a") : stderr;
    if (!out)
      out = stderr;
    std::fprintf(out, "===== AdaptiveCpp launch profiler (pid=%d) =====\n",
                 static_cast<int>(getpid()));
    for (int i = 0; i < num_slots; ++i) {
      unsigned long long count = g_data[i].count.load(std::memory_order_relaxed);
      if (count == 0)
        continue;
      unsigned long long total =
          g_data[i].total_ns.load(std::memory_order_relaxed);
      unsigned long long max =
          g_data[i].max_ns.load(std::memory_order_relaxed);
      std::fprintf(out,
                   "LP %-24s count=%llu mean_ns=%.1f max_ns=%llu "
                   "total_ms=%.3f%s\n",
                   slot_names[i], count,
                   count ? static_cast<double>(total) /
                               static_cast<double>(count)
                         : 0.0,
                   max, static_cast<double>(total) / 1.0e6,
                   i >= static_cast<int>(hipsycl::rt::launch_profiler::slot::
                                             InstantSubmissionCount)
                       ? " (counter)"
                       : "");
    }
    std::fflush(out);
    if (path && out != stderr)
      std::fclose(out);
  }
};

report_writer g_report_writer;

} // anonymous namespace

extern "C" ACPP_LT_PROFILE_EXPORT int acpp_lp_enabled() {
  return g_enabled ? 1 : 0;
}

extern "C" ACPP_LT_PROFILE_EXPORT void
acpp_lp_record(int slot, unsigned long long ns) {
  if (slot < 0 || slot >= num_slots)
    return;
  slot_data &data = g_data[slot];
  data.count.fetch_add(1, std::memory_order_relaxed);
  data.total_ns.fetch_add(ns, std::memory_order_relaxed);
  auto current_max = data.max_ns.load(std::memory_order_relaxed);
  while (ns > current_max &&
         !data.max_ns.compare_exchange_weak(current_max, ns,
                                            std::memory_order_relaxed)) {
  }
}

extern "C" ACPP_LT_PROFILE_EXPORT void acpp_lp_increment(int slot) {
  if (slot < 0 || slot >= num_slots)
    return;
  g_data[slot].count.fetch_add(1, std::memory_order_relaxed);
}
