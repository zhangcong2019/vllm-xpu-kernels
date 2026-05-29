#include <sycl/sycl.hpp>

#include <algorithm>
#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t val[4];
};

// The vector width is fixed at 4 to avoid excessive branching in the kernel,
// which could degrade performance.
template <typename scalar_t, int NUM_DIMS, int VEC_SIZE = 4>
class rms_norm_kernel {
 public:
  rms_norm_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_d2_,  // input.stride(-2)
      const int64_t input_stride_d3_,  // input.stride(-3)
      const int64_t input_stride_d4_,  // input.stride(-4)
      const int64_t input_shape_d2_,   // input.size(-2)
      const int64_t input_shape_d3_,   // input.size(-3)
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride_d2(input_stride_d2_),
        input_stride_d3(input_stride_d3_),
        input_stride_d4(input_stride_d4_),
        input_shape_d2(input_shape_d2_),
        input_shape_d3(input_shape_d3_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const scalar_t* input_row;
    if constexpr (NUM_DIMS == 2) {
      // 2D for layernorm normal case [batch_size, hidden]
      input_row = input + item_ct1.get_group(2) * input_stride_d2;
    } else if constexpr (NUM_DIMS == 3) {
      // 3D for q/k norm [batch_size, num_heads, head_size]
      int batch_idx = item_ct1.get_group(2) / input_shape_d2;
      int head_idx = item_ct1.get_group(2) % input_shape_d2;
      input_row =
          input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    } else if constexpr (NUM_DIMS == 4) {
      // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
      int batch_idx = item_ct1.get_group(2) / (input_shape_d3 * input_shape_d2);
      int remaining = item_ct1.get_group(2) % (input_shape_d3 * input_shape_d2);
      int seq_idx = remaining / input_shape_d2;
      int head_idx = remaining % input_shape_d2;
      input_row = input + batch_idx * input_stride_d4 +
                  seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    auto vec_op = [&variance](
                      const vec4_t<scalar_t>& vec, int vec_size = VEC_SIZE) {
      for (int i = 0; i < vec_size; ++i) {
        float x = static_cast<float>(vec.val[i]);
        variance += x * x;
      }
    };
    auto scalar_op = [&variance](const scalar_t& val) {
      float x = static_cast<float>(val);
      variance += x * x;
    };

    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
    uintptr_t addr_in = reinterpret_cast<uintptr_t>(input_row);

    // fast path when the whole region is already aligned
    bool can_vec =
        ((addr_in & (WIDTH - 1)) == 0) && ((hidden_size & (VEC_SIZE - 1)) == 0);
    if (can_vec) {
      int64_t const num_vec_elems = hidden_size / VEC_SIZE;
      auto const* vec_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        vec_op(tmp);
      }
    } else {
      int misalignment_offset = addr_in & (WIDTH - 1);
      int alignment_bytes = WIDTH - misalignment_offset;
      int prefix_elems = alignment_bytes & (WIDTH - 1);
      prefix_elems /= sizeof(scalar_t);
      prefix_elems = prefix_elems < hidden_size ? prefix_elems : hidden_size;

      // 1. handle the possibly unaligned prefix with scalar access.
      for (int i = item_ct1.get_local_id(2); i < prefix_elems;
           i += item_ct1.get_local_range(2)) {
        scalar_op(input_row[i]);
      }

      int64_t const num_vec_elems = (hidden_size - prefix_elems) / VEC_SIZE;
      auto const* vec_in =
          reinterpret_cast<const vec4_t<scalar_t>*>(input_row + prefix_elems);
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        vec_op(tmp);
      }

      // 3. handle remaining tail elements.
      for (int i = item_ct1.get_local_id(2) + num_vec_elems * VEC_SIZE;
           i < hidden_size - prefix_elems;
           i += item_ct1.get_local_range(2)) {
        scalar_op((input_row + prefix_elems)[i]);
      }
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    scalar_t* out_row = out + item_ct1.get_group(2) * hidden_size;
    uintptr_t addr_weight = reinterpret_cast<uintptr_t>(weight);
    uintptr_t addr_out = reinterpret_cast<uintptr_t>(out_row);
    bool can_vec_out = ((addr_in & (WIDTH - 1)) == 0) &&
                       ((addr_weight & (WIDTH - 1)) == 0) &&
                       ((addr_out & (WIDTH - 1)) == 0) &&
                       ((hidden_size & (VEC_SIZE - 1)) == 0);
    if (can_vec_out) {
      auto* v_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      auto* v_w = reinterpret_cast<const vec4_t<scalar_t>*>(weight);
      auto* v_out = reinterpret_cast<vec4_t<scalar_t>*>(out_row);
      int64_t const out_num_vec_elems = hidden_size / VEC_SIZE;
      float s_variance_val = *s_variance_ptr;
      for (int idx = item_ct1.get_local_id(2); idx < out_num_vec_elems;
           idx += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> dst;
        vec4_t<scalar_t> src1 = v_in[idx];
        vec4_t<scalar_t> src2 = v_w[idx];
        for (int j = 0; j < VEC_SIZE; j++) {
          float x = static_cast<float>(src1.val[j]);
          dst.val[j] = ((scalar_t)(x * s_variance_val)) * src2.val[j];
        }
        v_out[idx] = dst;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float x = (float)input_row[idx];
        out_row[idx] = ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
      }
    }
  }

 private:
  scalar_t* __restrict__ out;          // [..., hidden_size]
  const scalar_t* __restrict__ input;  // [..., hidden_size]
  const int64_t input_stride_d2;
  const int64_t input_stride_d3;
  const int64_t input_stride_d4;
  const int64_t input_shape_d2;
  const int64_t input_shape_d3;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t>
void call_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_RANK234(num_dims, [&]() {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rms_norm_kernel<sycl_t, tensor_rank>(
              (sycl_t*)out_ptr,
              (const sycl_t*)input_ptr,
              input_stride_d2,
              input_stride_d3,
              input_stride_d4,
              input_shape_d2,
              input_shape_d3,
              (const sycl_t*)weight_ptr,
              epsilon,
              num_tokens,
              hidden_size,
              s_variance));
    });
  });
}

template <typename scalar_t>
class fused_add_rms_norm_kernel {
 public:
  fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_,     // [..., hidden_size]
      scalar_t* __restrict__ residual_,  // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      scalar_t z = (scalar_t)input[item_ct1.get_group(2) * input_stride + idx];
      z += residual[item_ct1.get_group(2) * hidden_size + idx];
      float x = (float)z;
      variance += x * x;
      residual[item_ct1.get_group(2) * hidden_size + idx] = z;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
      input[item_ct1.get_group(2) * input_stride + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
    }
  }

 private:
  scalar_t* __restrict__ input;     // [..., hidden_size]
  scalar_t* __restrict__ residual;  // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;  // local memory for variance
};

template <typename scalar_t>
void call_fused_add_rms_norm_kernel(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  auto input_ptr = input.data_ptr<scalar_t>();
  auto residual_ptr = residual.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  int64_t input_stride = input.stride(-2);
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        fused_add_rms_norm_kernel<sycl_t>(
            (sycl_t*)input_ptr,
            (sycl_t*)residual_ptr,
            input_stride,
            (const sycl_t*)weight_ptr,
            epsilon,
            num_tokens,
            hidden_size,
            s_variance));
  });
}

// Gemma variant of RMSNorm.
// Differences vs vllm::rms_norm_kernel:
//   1. weight is loaded as (weight + 1.0) in fp32, applied to the normalized
//      value still in fp32 (the (1 + w) factor is the Gemma convention).
//   2. variance is accumulated in fp32 (already true upstream, kept here).
// Fast path mirrors vllm::rms_norm_kernel: a vec4_t<scalar_t> wide load/store
// when input/output/weight are all VEC_SIZE-aligned and hidden_size is a
// multiple of VEC_SIZE; otherwise a scalar fallback runs the whole row.
//
// USE_SLM=true variant additionally caches the fp32 input in shared local
// memory during pass 1, so pass 2 reads from SLM instead of re-fetching from
// HBM. Host launcher gates this on `hidden_size * sizeof(float)` <= SLM
// budget; otherwise USE_SLM=false runs and we re-read from HBM.
template <typename scalar_t, int VEC_SIZE = 4, bool USE_SLM = false>
class gemma_rms_norm_kernel {
 public:
  gemma_rms_norm_kernel(
      scalar_t* __restrict__ out_,           // [..., hidden_size]
      const scalar_t* __restrict__ input_,   // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_,
      sycl::local_accessor<float, 1> s_input_)
      : out(out_),
        input(input_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        s_variance(s_variance_),
        s_input(s_input_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* s_input_ptr =
        s_input.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const int row = item_ct1.get_group(2);
    const scalar_t* input_row = input + row * input_stride;
    scalar_t* out_row = out + row * hidden_size;

    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
    uintptr_t addr_in = reinterpret_cast<uintptr_t>(input_row);
    uintptr_t addr_w = reinterpret_cast<uintptr_t>(weight);
    uintptr_t addr_out = reinterpret_cast<uintptr_t>(out_row);
    bool can_vec = ((addr_in & (WIDTH - 1)) == 0) &&
                   ((addr_w & (WIDTH - 1)) == 0) &&
                   ((addr_out & (WIDTH - 1)) == 0) &&
                   ((hidden_size & (VEC_SIZE - 1)) == 0);

    // ---- pass 1: variance (and optionally cache fp32 input to SLM) ----
    if (can_vec) {
      auto const* v_in =
          reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      int64_t const num_vec = hidden_size / VEC_SIZE;
      for (int i = item_ct1.get_local_id(2); i < num_vec;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = v_in[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          float x = static_cast<float>(tmp.val[j]);
          variance += x * x;
          if constexpr (USE_SLM) {
            s_input_ptr[i * VEC_SIZE + j] = x;
          }
        }
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float x = static_cast<float>(input_row[idx]);
        variance += x * x;
        if constexpr (USE_SLM) {
          s_input_ptr[idx] = x;
        }
      }
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float rrms = *s_variance_ptr;

    // ---- pass 2: output = x * rrms * (1 + w) ----
    if (can_vec) {
      auto const* v_in =
          reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      auto const* v_w = reinterpret_cast<const vec4_t<scalar_t>*>(weight);
      auto* v_out = reinterpret_cast<vec4_t<scalar_t>*>(out_row);
      int64_t const num_vec = hidden_size / VEC_SIZE;
      for (int i = item_ct1.get_local_id(2); i < num_vec;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> w_v = v_w[i];
        vec4_t<scalar_t> dst;
        if constexpr (USE_SLM) {
#pragma unroll
          for (int j = 0; j < VEC_SIZE; ++j) {
            float x = s_input_ptr[i * VEC_SIZE + j];
            float w = static_cast<float>(w_v.val[j]) + 1.0f;
            dst.val[j] = static_cast<scalar_t>(x * rrms * w);
          }
        } else {
          vec4_t<scalar_t> in_v = v_in[i];
#pragma unroll
          for (int j = 0; j < VEC_SIZE; ++j) {
            float x = static_cast<float>(in_v.val[j]);
            float w = static_cast<float>(w_v.val[j]) + 1.0f;
            dst.val[j] = static_cast<scalar_t>(x * rrms * w);
          }
        }
        v_out[i] = dst;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float x;
        if constexpr (USE_SLM) {
          x = s_input_ptr[idx];
        } else {
          x = static_cast<float>(input_row[idx]);
        }
        float w = static_cast<float>(weight[idx]) + 1.0f;
        out_row[idx] = static_cast<scalar_t>(x * rrms * w);
      }
    }
  }

 private:
  scalar_t* __restrict__ out;
  const scalar_t* __restrict__ input;
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;
  const float epsilon;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
  sycl::local_accessor<float, 1> s_input;  // size 0 if USE_SLM=false
};

// Cache the fp32 input row in SLM when it fits in this budget. 32 KB allows
// hidden_size up to 8192, covering Gemma/Qwen3.5/Llama-class models without
// hurting occupancy (Intel XPU work-groups have 128 KB SLM, 32 KB leaves
// plenty of room for multiple concurrent groups per Xe-core).
constexpr int GEMMA_RMS_SLM_BUDGET_BYTES = 32 * 1024;

template <typename scalar_t>
void call_gemma_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int64_t input_stride = input.stride(-2);
  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();

  bool use_slm = (static_cast<size_t>(hidden_size) * sizeof(float)) <=
                 GEMMA_RMS_SLM_BUDGET_BYTES;

  if (use_slm) {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      sycl::local_accessor<float, 1> s_input(sycl::range<1>(hidden_size), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          gemma_rms_norm_kernel<sycl_t, /*VEC_SIZE=*/4, /*USE_SLM=*/true>(
              (sycl_t*)out_ptr,
              (const sycl_t*)input_ptr,
              input_stride,
              (const sycl_t*)weight_ptr,
              epsilon,
              hidden_size,
              s_variance,
              s_input));
    });
  } else {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      // Pass a dummy size-1 accessor; the kernel will not touch it.
      sycl::local_accessor<float, 1> s_input(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          gemma_rms_norm_kernel<sycl_t, /*VEC_SIZE=*/4, /*USE_SLM=*/false>(
              (sycl_t*)out_ptr,
              (const sycl_t*)input_ptr,
              input_stride,
              (const sycl_t*)weight_ptr,
              epsilon,
              hidden_size,
              s_variance,
              s_input));
    });
  }
}

// Gemma variant of fused add + RMSNorm. In-place semantics:
//   residual <- input + residual   (the unnormalized sum)
//   input    <- normalize(residual) * (1 + weight)   (in fp32, then cast)
// Fast path uses vec4_t loads/stores when all four pointers and hidden_size
// are VEC_SIZE-aligned; otherwise scalar fallback.
//
// USE_SLM=true variant caches the fp32 `summed` values to shared local memory
// during pass 1, so pass 2 reads from SLM instead of from HBM residual_row.
// We still must store summed back to HBM residual_row (it's the externally
// visible output), so this only saves the pass-2 HBM read, not the write.
template <typename scalar_t, int VEC_SIZE = 4, bool USE_SLM = false>
class gemma_fused_add_rms_norm_kernel {
 public:
  gemma_fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_,         // [..., hidden_size]
      scalar_t* __restrict__ residual_,      // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_,
      sycl::local_accessor<float, 1> s_summed_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        hidden_size(hidden_size_),
        s_variance(s_variance_),
        s_summed(s_summed_) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* s_summed_ptr =
        s_summed.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const int row = item_ct1.get_group(2);
    scalar_t* input_row = input + row * input_stride;
    scalar_t* residual_row = residual + row * hidden_size;

    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
    uintptr_t addr_in = reinterpret_cast<uintptr_t>(input_row);
    uintptr_t addr_res = reinterpret_cast<uintptr_t>(residual_row);
    uintptr_t addr_w = reinterpret_cast<uintptr_t>(weight);
    bool can_vec = ((addr_in & (WIDTH - 1)) == 0) &&
                   ((addr_res & (WIDTH - 1)) == 0) &&
                   ((addr_w & (WIDTH - 1)) == 0) &&
                   ((hidden_size & (VEC_SIZE - 1)) == 0);

    // ---- pass 1: residual = input + residual; cache summed; accumulate variance ----
    if (can_vec) {
      auto* v_in = reinterpret_cast<vec4_t<scalar_t>*>(input_row);
      auto* v_res = reinterpret_cast<vec4_t<scalar_t>*>(residual_row);
      int64_t const num_vec = hidden_size / VEC_SIZE;
      for (int i = item_ct1.get_local_id(2); i < num_vec;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> in_v = v_in[i];
        vec4_t<scalar_t> res_v = v_res[i];
        vec4_t<scalar_t> sum_v;
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          float xs = static_cast<float>(in_v.val[j]) +
                     static_cast<float>(res_v.val[j]);
          variance += xs * xs;
          sum_v.val[j] = static_cast<scalar_t>(xs);
          if constexpr (USE_SLM) {
            s_summed_ptr[i * VEC_SIZE + j] = xs;
          }
        }
        v_res[i] = sum_v;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float xs = static_cast<float>(input_row[idx]) +
                   static_cast<float>(residual_row[idx]);
        variance += xs * xs;
        residual_row[idx] = static_cast<scalar_t>(xs);
        if constexpr (USE_SLM) {
          s_summed_ptr[idx] = xs;
        }
      }
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float rrms = *s_variance_ptr;

    // ---- pass 2: input = norm(residual) * (1 + w) ----
    // With USE_SLM=true we read summed from SLM and skip the HBM residual read.
    // Note: summed in SLM is fp32 (more accurate than the cast-then-recast bf16
    // round-trip in the no-SLM path). This is a strict improvement.
    if (can_vec) {
      auto const* v_res =
          reinterpret_cast<const vec4_t<scalar_t>*>(residual_row);
      auto const* v_w = reinterpret_cast<const vec4_t<scalar_t>*>(weight);
      auto* v_in = reinterpret_cast<vec4_t<scalar_t>*>(input_row);
      int64_t const num_vec = hidden_size / VEC_SIZE;
      for (int i = item_ct1.get_local_id(2); i < num_vec;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> w_v = v_w[i];
        vec4_t<scalar_t> dst;
        if constexpr (USE_SLM) {
#pragma unroll
          for (int j = 0; j < VEC_SIZE; ++j) {
            float xs = s_summed_ptr[i * VEC_SIZE + j];
            float w = static_cast<float>(w_v.val[j]) + 1.0f;
            dst.val[j] = static_cast<scalar_t>(xs * rrms * w);
          }
        } else {
          vec4_t<scalar_t> res_v = v_res[i];
#pragma unroll
          for (int j = 0; j < VEC_SIZE; ++j) {
            float xs = static_cast<float>(res_v.val[j]);
            float w = static_cast<float>(w_v.val[j]) + 1.0f;
            dst.val[j] = static_cast<scalar_t>(xs * rrms * w);
          }
        }
        v_in[i] = dst;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float xs;
        if constexpr (USE_SLM) {
          xs = s_summed_ptr[idx];
        } else {
          xs = static_cast<float>(residual_row[idx]);
        }
        float w = static_cast<float>(weight[idx]) + 1.0f;
        input_row[idx] = static_cast<scalar_t>(xs * rrms * w);
      }
    }
  }

 private:
  scalar_t* __restrict__ input;
  scalar_t* __restrict__ residual;
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;
  const float epsilon;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
  sycl::local_accessor<float, 1> s_summed;  // size 0 if USE_SLM=false
};

template <typename scalar_t>
void call_gemma_fused_add_rms_norm_kernel(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int64_t input_stride = input.stride(-2);
  auto input_ptr = input.data_ptr<scalar_t>();
  auto residual_ptr = residual.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();

  bool use_slm = (static_cast<size_t>(hidden_size) * sizeof(float)) <=
                 GEMMA_RMS_SLM_BUDGET_BYTES;

  if (use_slm) {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      sycl::local_accessor<float, 1> s_summed(sycl::range<1>(hidden_size), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          gemma_fused_add_rms_norm_kernel<sycl_t, /*VEC_SIZE=*/4, /*USE_SLM=*/true>(
              (sycl_t*)input_ptr,
              (sycl_t*)residual_ptr,
              input_stride,
              (const sycl_t*)weight_ptr,
              epsilon,
              hidden_size,
              s_variance,
              s_summed));
    });
  } else {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      sycl::local_accessor<float, 1> s_summed(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          gemma_fused_add_rms_norm_kernel<sycl_t, /*VEC_SIZE=*/4, /*USE_SLM=*/false>(
              (sycl_t*)input_ptr,
              (sycl_t*)residual_ptr,
              input_stride,
              (const sycl_t*)weight_ptr,
              epsilon,
              hidden_size,
              s_variance,
              s_summed));
    });
  }
}

}  // namespace vllm

void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_kernel", [&] {
        vllm::call_rms_norm_kernel<scalar_t>(out, input, weight, epsilon);
      });
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_fused_add_rms_norm_kernel", [&] {
        vllm::call_fused_add_rms_norm_kernel<scalar_t>(
            input, residual, weight, epsilon);
      });
}

void gemma_rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_gemma_rms_norm_kernel", [&] {
        vllm::call_gemma_rms_norm_kernel<scalar_t>(
            out, input, weight, epsilon);
      });
}

void gemma_fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_gemma_fused_add_rms_norm_kernel", [&] {
        vllm::call_gemma_fused_add_rms_norm_kernel<scalar_t>(
            input, residual, weight, epsilon);
      });
}