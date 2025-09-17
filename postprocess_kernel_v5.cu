#include <cuda_fp16.h>
#define BX 32
#define BY 8
#define HALO 1
#define TILE_W (BX + 2*HALO)   // 34
#define TILE_H (BY + 2*HALO)   // 10
#define LUMA_R 0.299f
#define LUMA_G 0.587f
#define LUMA_B 0.114f
#define SHARPNESS_K 0.1f
#define EPSILON 1e-6f

__device__ __forceinline__ int clampi(int val, int min, int max) {
  return val < min ? min : (val > max ? max : val);
}

__device__ __forceinline__ float clampf(float val, float min_val, float max_val) {
  return fmaxf(fminf(val, max_val), min_val);
}

__device__ __forceinline__ float luma_from_rgb(float r, float g, float b) {
    // Use multiply-add for better instruction-level parallelism
  return __fmaf_rn(LUMA_B, b, __fmaf_rn(LUMA_G, g, LUMA_R * r));}

__device__ __forceinline__ float2 luma_from_rgb2(float2 r, float2 g, float2 b) {
  return make_float2(
      __fmaf_rn(LUMA_B, b.x, __fmaf_rn(LUMA_G, g.x, LUMA_R * r.x)),
      __fmaf_rn(LUMA_B, b.y, __fmaf_rn(LUMA_G, g.y, LUMA_R * r.y))
  );
}
// V5 OPTIMIZATION: Fast RGB to uint8 conversion with saturation
__device__ __forceinline__ unsigned char float_to_u8_sat(float val) {
  return __saturatef(val) * 255.0f + 0.5f;
}


extern "C" __global__ void enhance_kernel(
    const __half* __restrict__ input, // [C, H, W]
    uchar3*       __restrict__ output_hwc_rgb, // [H, W, C] format: RGB
    int width,
    int height,
    const float SHARPNESS_CONST, // 0.3f - 0.4f
    const float EDGE_ENHANCE_CONST // 1.0f
) {
  __shared__ float tile_luma[TILE_H][TILE_W]; // luma values

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int block_x0 = blockIdx.x * BX, block_y0 = blockIdx.y * BY;
  const int global_x = block_x0 + tx, global_y = block_y0 + ty;
  const int local_x = tx + HALO, local_y = ty + HALO;

  const int plane = width * height;
  const __half* __restrict__ inpR = input + 0 * plane;
  const __half* __restrict__ inpG = input + 1 * plane;
  const __half* __restrict__ inpB = input + 2 * plane;

  // Vectorized, Cooperative load
  for (int j = ty; j < TILE_H; j += BY) {
    const int y = clampi(block_y0 + j - HALO, 0, height - 1);
    const int row = y * width;
    // 17 pairs cover 34 columns; only lanes 0..16 do work
    if (tx * 2 < TILE_W) {
      // load pair [x0, x1] = [0,1], [2,3], [4,5], ...
      const int i0 = tx * 2;
      const int x0 = clampi(block_x0 + i0 - HALO    , 0, width - 1);
      const int x1 = clampi(block_x0 + i0 - HALO + 1, 0, width - 1);
      const bool can_vectorize = (x1 == x0 + 1) && ((x0 & 1) == 0); // aligned even
      if (can_vectorize) {
        // Vector loads (aligned when x0 even)
        const __half2 rr = *reinterpret_cast<const __half2*>(inpR + row + x0);
        const __half2 gg = *reinterpret_cast<const __half2*>(inpG + row + x0);
        const __half2 bb = *reinterpret_cast<const __half2*>(inpB + row + x0);

        const float2 rrf = __half22float2(rr);
        const float2 ggf = __half22float2(gg);
        const float2 bbf = __half22float2(bb);
  
        // V5 OPTIMIZATION: Vectorized luma calculation
        const float2 luma_pair = luma_from_rgb2(rrf, ggf, bbf);
        
        tile_luma[j][i0]     = luma_pair.x;
        tile_luma[j][i0 + 1] = luma_pair.y;
      } else {
        // Fallback scalar for odd/aligned edge cases
        const int idx0 = row + x0;
        const int idx1 = row + x1;
  
        const float r0 = __half2float(inpR[idx0]);
        const float g0 = __half2float(inpG[idx0]);
        const float b0 = __half2float(inpB[idx0]);
  
        const float r1 = __half2float(inpR[idx1]);
        const float g1 = __half2float(inpG[idx1]);
        const float b1 = __half2float(inpB[idx1]);
  
        tile_luma[j][i0]     = luma_from_rgb(r0, g0, b0);
        tile_luma[j][i0 + 1] = luma_from_rgb(r1, g1, b1);
      }
    }
  }
  __syncthreads();

  if (global_x >= width || global_y >= height) return; // safe to do after barrier

  const int hasL = (global_x > 0), hasR = (global_x + 1 < width), hasT = (global_y > 0), hasB = (global_y + 1 < height);
  const int horizontal_mask[3] = {hasL, 1, hasR};
  const int vertical_mask[3] = {hasT, 1, hasB};
  const int num_in_bounds = (hasL + 1 + hasR) * (hasT + 1 + hasB); // (#h) * (#v) in {4,6,9}
  const int idx = global_y * width + global_x;
  const float Lc = tile_luma[local_y][local_x];
  const float r = __half2float(inpR[idx]);
  const float g = __half2float(inpG[idx]);
  const float b = __half2float(inpB[idx]);
  // ========================
  // CAS Algorithm
  // ========================
  float Lmin = 1.0f, Lmax = -1.0f, Lsum = 0.0f;
  #pragma unroll
  for (int dy = -1; dy <= 1; dy++) {
    const int v_mask = vertical_mask[dy + 1];
    #pragma unroll
    for (int dx = -1; dx <= 1; dx++) {
      const int h_mask = horizontal_mask[dx + 1];
      const int mask = v_mask & h_mask;
      float Ln = tile_luma[local_y + dy][local_x + dx];
      Lsum += Ln * mask;
      Ln = mask ? Ln : 1.0f;
      Lmin = fminf(Lmin, Ln);
      Ln = mask ? Ln : 0.0f;
      Lmax = fmaxf(Lmax, Ln);
    }
  }
  const float inv_num_in_bounds = (num_in_bounds == 9) ? (1.0f/9.0f) : 
                                  (num_in_bounds == 6) ? (1.0f/6.0f) : 0.25f;
  const float Lavg = Lsum * inv_num_in_bounds;
  const float Ldelta = Lc - Lavg;
  const float Lcontrast = Lmax - Lmin;
  
  // gain factor
  const float contrast_term = Lcontrast + SHARPNESS_K;
  const float gain = SHARPNESS_CONST * __fdividef(Lcontrast, contrast_term);
  const float Lstar = Lc + gain * Ldelta;
  const float Lstar_clamped = clampf(Lstar, Lmin, Lmax);

  // preserve hue with scaling
  const float s_req = __fdividef(Lstar_clamped, fmaxf(Lc, EPSILON));
  const float maxRGBval = fmaxf(fmaxf(r, g), b);
  const float s_capped = maxRGBval > 0.0f ? __fdividef(1.0f, maxRGBval) : 1.0f;
  const float s = fminf(s_req, s_capped);

  // 
  output_hwc_rgb[idx].x = float_to_u8_sat(r*s);
  output_hwc_rgb[idx].y = float_to_u8_sat(g*s);
  output_hwc_rgb[idx].z = float_to_u8_sat(b*s);
}


// V5 OPTIMIZATION: Completely vectorized u8 to f16 conversion
extern "C" __global__ void bgr_u8_to_rgb_f16_planar_vectorized(
  const unsigned char* __restrict__ input,
  __half* __restrict__ output,
  int width,
  int height
) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (x >= width || y >= height) return;
  
  const int pixel_idx = y * width + x;
  const int input_base = pixel_idx * 3;
  const int plane_size = width * height;
  
  // V5 OPTIMIZATION: Load RGB values and convert in single operation
  const uchar3 rgb_u8 = make_uchar3(
      input[input_base + 2],  // R
      input[input_base + 1],  // G  
      input[input_base + 0]   // B
  );
  
  // V5 OPTIMIZATION: Vectorized conversion with single division
  const float inv_255 = 1.0f / 255.0f;
  const float3 rgb_f32 = make_float3(
      rgb_u8.x * inv_255,
      rgb_u8.y * inv_255,
      rgb_u8.z * inv_255
  );
  
  // V5 OPTIMIZATION: Direct half precision conversion
  output[pixel_idx] = __float2half(rgb_f32.x);                    // R plane
  output[pixel_idx + plane_size] = __float2half(rgb_f32.y);       // G plane  
  output[pixel_idx + 2 * plane_size] = __float2half(rgb_f32.z);   // B plane
}