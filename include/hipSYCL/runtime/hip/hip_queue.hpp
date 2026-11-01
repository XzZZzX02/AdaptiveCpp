/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_HIP_QUEUE_HPP
#define HIPSYCL_HIP_QUEUE_HPP

#include "../executor.hpp"
#include "../inorder_queue.hpp"
#include "../generic/host_timestamped_event.hpp"
#include "../code_object_invoker.hpp"

#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/reflection_map.hpp"
#include "hip_instrumentation.hpp"

// Avoid including HIP headers to prevent conflicts with CUDA
struct ihipStream_t;

#include <string>
#include <vector>
#include <cstdint>

namespace hipsycl {
namespace rt {

class hip_queue;
class hip_backend;
class hip_hardware_context;
class hip_event_pool;

class hip_multipass_code_object_invoker : public multipass_code_object_invoker {
public:
  hip_multipass_code_object_invoker(hip_queue* q);

  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               const std::string &kernel_name_tag,
                               const std::string &kernel_body_name) override;
  virtual ~hip_multipass_code_object_invoker(){}
private:
  hip_queue* _queue;
};


class hip_sscp_code_object_invoker : public sscp_code_object_invoker {
public:
  hip_sscp_code_object_invoker(hip_queue* q)
  : _queue{q} {}

  virtual ~hip_sscp_code_object_invoker(){}

  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               std::string_view kernel_name,
                               const rt::hcf_kernel_info* kernel_info,
                               const kernel_configuration& config) override;
private:
  hip_queue* _queue;
};

class hip_queue : public inorder_queue
{
public:
  hip_queue(hip_backend* be, device_id dev, int priority = 0);

  ihipStream_t* get_stream() const;

  virtual ~hip_queue();

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() override;
  virtual std::shared_ptr<dag_node_event> create_queue_completion_event() override;

  virtual result submit_memcpy(memcpy_operation&, const dag_node_ptr&) override;
  virtual result submit_kernel(kernel_operation&, const dag_node_ptr&) override;
  virtual result submit_prefetch(prefetch_operation &, const dag_node_ptr&) override;
  virtual result submit_memset(memset_operation&, const dag_node_ptr&) override;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(const dag_node_ptr& evt) override;
  virtual result submit_external_wait_for(const dag_node_ptr& node) override;

  virtual result wait() override;

  virtual device_id get_device() const override;
  virtual void* get_native_type() const override;

  virtual result query_status(inorder_queue_status& status) override;

  result submit_multipass_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &backend_kernel_name, const rt::range<3> &grid_size,
      const rt::range<3> &block_size, unsigned dynamic_shared_mem,
      void **kernel_args, std::size_t* arg_sizes, std::size_t num_args);

  result submit_sscp_kernel_from_code_object(
      hcf_object_id hcf_object, std::string_view kernel_name,
      const rt::hcf_kernel_info *kernel_info, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const kernel_configuration &config) override;

  const host_timestamped_event& get_timing_reference() const {
    return _reference_event;
  }
private:
  void activate_device() const;

  const device_id _dev;
  ihipStream_t* _stream;
  host_timestamped_event _reference_event;
  hip_backend* _backend;
  // Cached per-queue data that would otherwise be re-queried on every launch.
  hip_hardware_context* _hw_context = nullptr;
  hip_event_pool* _event_pool = nullptr;
  std::string _target_arch;
  hip_multipass_code_object_invoker _multipass_code_object_invoker;
  hip_sscp_code_object_invoker _sscp_code_object_invoker;
  std::shared_ptr<kernel_cache> _kernel_cache;

  // SSCP submission data
  common::spin_lock _sscp_submission_spin_lock;
  glue::jit::cxx_argument_mapper _arg_mapper;
  kernel_configuration _config;
  glue::jit::reflection_map _reflection_map;

  // Prepared-launch shortcut: repeated launches of the same kernel with the
  // same arguments and launch geometry can reuse the computed kernel
  // configuration ids instead of rebuilding the configuration from scratch.
  // Only used when ACPP_ADAPTIVITY_LEVEL <= 1: at level 2 the invariant
  // argument specialization (IADS) can change the configuration over time,
  // so configurations must be re-evaluated on every launch.
  struct sscp_prepared_launch {
    bool valid = false;
    hcf_object_id hcf_object = 0;
    const rt::hcf_kernel_info *kernel_info = nullptr;
    rt::range<3> num_groups{};
    rt::range<3> group_size{};
    unsigned local_mem_size = 0;
    // Values that can influence the configuration: pointer values (alignment /
    // allocation aliasing), plus values of parameters annotated as
    // specialized / function-call-specialization. Ordinary scalar values are
    // deliberately not part of the signature: at adaptivity level <= 1 they do
    // not affect the configuration, so kernels whose scalar arguments change
    // every launch can still reuse the prepared descriptor.
    std::vector<uint64_t> signature;
    // Code object and kernel function are immutable for the lifetime of the
    // kernel cache (unload only happens at runtime shutdown), so they can be
    // reused directly by the prepared-launch fast path.
    const rt::code_object *object = nullptr;
    void *function = nullptr;
    kernel_configuration::id_type binary_configuration_id{};
    kernel_configuration::id_type code_object_configuration_id{};
  };
  static constexpr int _num_prepared_launches = 4;
  sscp_prepared_launch _prepared_launches[_num_prepared_launches];
  std::vector<uint64_t> _prepared_signature_scratch;
  int _next_prepared_launch = 0;
  int _adaptivity_level = 1;
  unsigned _wavefront_size = 64;
};

}
}

#endif
