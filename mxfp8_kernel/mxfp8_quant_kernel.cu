#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cassert>
#include <cstdint>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#pragma STDC FENV_ACCESS ON
#include <cfenv>

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Constants for stochastic rounding
#define STOCH_ROUNDING_MIN  -2147483647
#define STOCH_ROUNDING_MAX  2147483647

#define SIGN_OFFSET_8B       7
#define EXPONENT_OFFSET_8B   4
#define EXPONENT_BIAS_8B     7
#define SIGNIFICAND_MASK_8B  0x0F
#define EXPONENT_MASK_8B     0x70
#define SIGN_MASK_8B         0x80

// Select bits x[high:low]
__device__ uint32_t sbs(uint32_t x, uint8_t high, uint8_t low) {
  return (high == 31) ? (x >> low) : ((x & ((1 << (high + 1)) - 1)) >> low);
}

// Helper function to check if a 32-bit float is zero
__device__ bool fp32_is_zero(uint32_t val) {
  return (val & (~0x80000000)) == 0;
}

// Check if a 32-bit float is infinity
__device__ bool fp32_is_infinity(uint32_t val) {
  return (val & 0x7FFFFFFF) == 0x7F800000;
}

// Check if a 32-bit float is NaN
__device__ bool fp32_is_nan(uint32_t val) {
  return ((val & 0x7F800000) == 0x7F800000) && ((val & 0x007FFFFF) != 0);
}

// Perform rounding based on stochastic or nearest rounding
__device__ int fp_accommodate_rounding(uint32_t intValuePreRounding, bool roundedMSB, bool roundedLSBs,
                                       unsigned int sign, bool stochastic,
                                       uint32_t lfsrVal, uint32_t discardedAlignedLeft) {
  uint32_t result = intValuePreRounding;

  if (stochastic) {
    // Stochastic rounding
    if (discardedAlignedLeft >= lfsrVal) {
      result = intValuePreRounding + 1;
    }
  } else {
    // Round to nearest even
    if ((((intValuePreRounding & 0x1) == 1) && (roundedMSB == 1)) ||
        (((intValuePreRounding & 0x1) == 0) && (roundedMSB == 1) && (roundedLSBs == 1))) {
      result = intValuePreRounding + 1;
    }
  }

  return result;
}

// Convert FP32 to MXFP8
__device__ void make_fp32_to_8bit(float input, uint8_t *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias,
                                  int32_t lfsrVal, bool stochastic) {
  const uint32_t inputUint = *(const uint32_t *)&input;
  int inputExponent = (inputUint & 0x7F800000) >> 23;
  uint32_t inputMantissa = inputUint & 0x007FFFFF;
  int inputSign = (inputUint & 0x80000000) >> 31;

  int unbiasedExp = inputExponent - 127;
  int minNormExp = 1 - exp_bias;
  int maxExp = ((1 << exp_width) - 1) - exp_bias;
  int minExp = minNormExp - man_width - 1;

  int32_t outputExponent = 0;
  int32_t outputMantissa = 0;
  int32_t outputSign = inputSign;

  // Handle special cases
  if (fp32_is_nan(inputUint) || fp32_is_infinity(inputUint)) {
    outputExponent = 0xF;
    outputMantissa = 0x0;
  } else if (fp32_is_zero(inputUint)) {
    outputExponent = 0x0;
    outputMantissa = 0x0;
  } else {
    if (unbiasedExp > maxExp) {
      outputExponent = maxExp + exp_bias;
      outputMantissa = (1 << man_width) - 1;  // Max mantissa
    } else if (unbiasedExp < minExp) {
      outputExponent = 0;
      outputMantissa = 0;  // Denormalized numbers
    } else {
      outputExponent = unbiasedExp + exp_bias;
      outputMantissa = inputMantissa >> (23 - man_width);
    }
  }

  // Combine components
  *output = (outputSign << SIGN_OFFSET_8B) |
            (outputExponent << man_width) |
            outputMantissa;
}

// Convert MXFP8 to FP32
__device__ void make_8bit_to_fp32(uint8_t input, float *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias) {
  int32_t inputSign = (input & SIGN_MASK_8B) >> SIGN_OFFSET_8B;
  int32_t inputExponent = (input & EXPONENT_MASK_8B) >> man_width;
  int32_t inputMantissa = input & SIGNIFICAND_MASK_8B;

  uint32_t outputUint = 0;
  if (inputExponent == 0 && inputMantissa == 0) {
    // Zero
    outputUint = inputSign << 31;
  } else {
    int32_t unbiasedExp = inputExponent - exp_bias;
    outputUint = (inputSign << 31) |
                 ((unbiasedExp + 127) << 23) |
                 (inputMantissa << (23 - man_width));
  }

  *output = *(float *)&outputUint;
}

// Kernel to quantize data to MXFP8
__global__ void Quant8BitKernel(const float *in_data, float *out_data, const int totalElements,
                                const uint8_t exp_width, const uint8_t man_width, const uint8_t exp_bias,
                                const int32_t *lfsrVal, bool stochastic) {
  CUDA_KERNEL_LOOP(i, totalElements) {
    uint8_t out_8bit;
    make_fp32_to_8bit(in_data[i], &out_8bit, exp_width, man_width, exp_bias, lfsrVal ? lfsrVal[i] : 0, stochastic);
    make_8bit_to_fp32(out_8bit, &out_data[i], exp_width, man_width, exp_bias);
  }
}

// MXFP8 quantization entry point
torch::Tensor mxfp8_cuda(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias, bool stochastic) {
  const auto num_elements = input.numel();

  torch::Tensor output = torch::empty_like(input);
  torch::Tensor rand = torch::empty_like(input, torch::dtype(torch::kInt32));
  if (stochastic) {
    rand.random_(STOCH_ROUNDING_MIN, STOCH_ROUNDING_MAX);
  }

  Quant8BitKernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), num_elements,
      exp_width, man_width, exp_bias, stochastic ? rand.data_ptr<int32_t>() : nullptr, stochastic);

  return output;
}
