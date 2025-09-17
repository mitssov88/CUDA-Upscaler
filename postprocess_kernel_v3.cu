#include <cuda_fp16.h>
#define BX 16
#define BY 16
#define HALO 1
#define TILE_W (BX + 2*HALO)   // 18
#define TILE_H (BY + 2*HALO)   // 18

extern "C" __global__ void bgr_u8_to_rgb_f16_planar(
  const unsigned char* input,
  __half* output,
  int width,
  int height
) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = y * width + x;
  output[idx] = __float2half(input[idx * 3 + 2] / 255.0f); // R channel
  output[idx + width * height] = __float2half(input[idx * 3 + 1] / 255.0f);  // G channel  
  output[idx + 2 * width * height] =  __float2half(input[idx * 3 + 0] / 255.0f);  // B channel
}

// clamped luma at (x, y)
__device__ __forceinline__ float luma_at(const __half* inpR, const __half* inpG, const __half* inpB, int width, int height, int x, int y) {
  x = max(0, min(x, width  - 1));
  y = max(0, min(y, height - 1));
  float r = __half2float(inpR[y*width + x]);
  float g = __half2float(inpG[y*width + x]);
  float b = __half2float(inpB[y*width + x]);
  return 0.299f*r + 0.587f*g + 0.114f*b;
}

extern "C" __global__ void enhance_kernel(
    const __half* input, // FLATTENED INPUT [1, C, H, W]
    unsigned char* hwc_output, // FLATTENED OUTPUT [1, H, W, C]
    int width,
    int height,
    const float SHARPNESS_CONST, // 0.3f - 0.4f
    const float EDGE_ENHANCE_CONST // 1.0f
) {
  __shared__ float tile_luma[TILE_H][TILE_W]; // luma values

  int tx = threadIdx.x, ty = threadIdx.y;
  int block_x0 = blockIdx.x * BX, block_y0 = blockIdx.y * BY;
  int global_x = block_x0 + tx;
  int global_y = block_y0 + ty;

  bool OOB = (global_x >= width || global_y >= height);

  const int plane = width * height;
  const __half* inpR = input;
  const __half* inpG = input + plane;
  const __half* inpB = input + 2 * plane;

  // Load center tile
  tile_luma[ty+HALO][tx+HALO] = luma_at(inpR, inpG, inpB, width, height, global_x, global_y);

  // edge threads load halo
  if (tx == 0) tile_luma[ty +HALO][0] = luma_at(inpR, inpG, inpB, width, height, global_x -1, global_y);
  if (tx == BX - 1) tile_luma[ty +HALO][TILE_W - 1] = luma_at(inpR, inpG, inpB, width, height, global_x + 1, global_y);
  if (ty == 0) tile_luma[0][tx +HALO] = luma_at(inpR, inpG, inpB, width, height, global_x, global_y - 1);
  if (ty == BY - 1) tile_luma[TILE_H - 1][tx +HALO] = luma_at(inpR, inpG, inpB, width, height, global_x, global_y + 1);

  // corner threads load halo
  if (tx == 0 && ty == 0) tile_luma[0][0] = luma_at(inpR, inpG, inpB, width, height, global_x - 1, global_y - 1);
  if (tx == 0 && ty == BY-1) tile_luma[TILE_H - 1][0] = luma_at(inpR, inpG, inpB, width, height, global_x - 1, global_y + 1);
  if (tx == BX-1 && ty == 0) tile_luma[0][TILE_W - 1] = luma_at(inpR, inpG, inpB, width, height, global_x + 1, global_y - 1);
  if (tx == BX-1 && ty == BY-1) tile_luma[TILE_H - 1][TILE_W - 1] = luma_at(inpR, inpG, inpB, width, height, global_x + 1, global_y + 1);

  __syncthreads();
  if (OOB) return; // even OOB threads must reach barrier

  // CAS
  int num_out_of_bounds = 0;
  float Lmin = 1.0f;
  float Lmax = -1.0f;
  float Lsum = 0.0f;
  float Lc = tile_luma[ty+HALO][tx+HALO];
  #pragma unroll
  for (int dy = -1; dy <= 1; dy++) {
    #pragma unroll
    for (int dx = -1; dx <= 1; dx++) {
      int nx = global_x + dx;
      int ny = global_y + dy;

      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        num_out_of_bounds++;
        continue;
      }

      float Ln = tile_luma[ty + HALO + dy][tx + HALO + dx];
      Lmin = fminf(Lmin, Ln);
      Lmax = fmaxf(Lmax, Ln);
      Lsum += Ln;
    }
  }

  float Lavg = Lsum / (9.0f - num_out_of_bounds);
  float Ldelta = Lc - Lavg;
  float Lcontrast = Lmax - Lmin;
  
  // gain factor
  float gain = SHARPNESS_CONST * (Lcontrast / (Lcontrast + 0.1f)); // k_sharp = 0.1f
  float Lstar = Lc + gain * Ldelta;
  float Lstar_clamped = fminf(fmaxf(Lstar, Lmin), Lmax);

  float r = __half2float(inpR[global_y*width + global_x]);
  float g = __half2float(inpG[global_y*width + global_x]);
  float b = __half2float(inpB[global_y*width + global_x]);

  float s_req = Lstar_clamped / fmaxf(Lc, 1e-6f);   // Lold = 0? avoid blow-up
  float maxRGBval = fmaxf(fmaxf(r, g), b);
  float s_capped = (maxRGBval > 0.0f) ? 1 / maxRGBval : 1.0f;
  float s = fminf(s_req, s_capped);

  float enhanced_val_r = fminf(fmaxf(r * s, 0.0f), 1.0f) * 255.0f;
  float enhanced_val_g = fminf(fmaxf(g * s, 0.0f), 1.0f) * 255.0f;
  float enhanced_val_b = fminf(fmaxf(b * s, 0.0f), 1.0f) * 255.0f;

  // 1 C H W -> 1 H W C each pixel is 3 bytes
  hwc_output[(width * global_y + global_x) * 3 + 0] = static_cast<unsigned char>(enhanced_val_r);
  hwc_output[(width * global_y + global_x) * 3 + 1] = static_cast<unsigned char>(enhanced_val_g);
  hwc_output[(width * global_y + global_x) * 3 + 2] = static_cast<unsigned char>(enhanced_val_b);
}

