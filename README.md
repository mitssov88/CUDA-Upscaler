# CUDA Upscaler (TensorRT + CUDA)

> Real-time image/video upscaling with a highly optimized GPU pipeline: TensorRT FP16 inference, ONNX->TensorRT model build, asynchronous multi-stream overlap, custom CUDA kernels, pinned memory, fused instructions, cooperative vectorized loading, with aggressive compiler optimizations, and more.

## Overview

This project demonstrates end-to-end ML systems engineering through a high-performance GPU-accelerated upscaling pipeline. Built to learn and showcase optimization techniques from engine building to kernel development and profiling.

### Architecture

The pipeline uses TensorRT engines with fixed 1×3×H×W shapes (default 360p→720p x2) and implements several key optimizations:

- **TensorRT FP16 inference** with pre-built engines for consistent performance
- **Multi-stream GPU pipeline** overlapping H2D/D2H transfers with compute
- **Tiled processing** with configurable halo regions to handle arbitrary resolutions
- **Custom CUDA kernels** for color conversion, CAS sharpening, and post-processing
- **Round Robin I/O** with pinned host memory for optimal data movement

```
            ┌──────────┐   H2D    ┌────────────┐   CUDA    ┌─────────┐  D2H
Input → CPU │ Preproc  | ───────▶  TRT Engine    ───────▶  Postproc    ──▶  CPU ──▶ Save
            └──────────┘          └────────────┘           └─────────┘
                 ▲                                                             ▲
      Pinned host buffers    Streams: H2D | Compute | D2H                      │
                 └────────────  Round-robin scheduling  ───────────────────────┘
```

## Sample Performance Results

Benchmarked on RTX 4090 @ 360p→720p x2 upscaling:

| Metric (mean)      | V1_Naive | V2_PreAlloc | V3_Async | V4_Kernels | V5_Ultimate |
|--------------------|----------|-------------|----------|------------|-------------|
| Time/Frame (ms)    | 40.6     | 32.7        | 23.4     | 22.6       | 18.2        |
| Theoretical FPS    | 24.6     | 30.6        | 42.6     | 44.2       | 55.0        |
| Speedup vs V1      | 1.0x     | 1.2x        | 1.7x     | 1.8x       | 2.2x        |
| CPU Usage (%)      | 14.3     | 13.0        | 14.8     | 21.9       | 16.6        |
| CPU Memory (MB)    | 1061     | 867         | 814      | 905        | 1013        |
| GPU Usage (%)      | 46.5     | 52.9        | 75.7     | 81.7       | 80.7        |
| GPU Memory (MB)    | 1118     | 1162        | 1132     | 1885       | 1465        |
| GPU Memory (%)     | 4.5      | 4.7         | 4.6      | 7.7        | 6.0         |

Each version demonstrates progressive optimization: briefly, V1 establishes baseline, V2 eliminates allocation overhead, V3 adds custom CUDA kernels, V4 introduces async streaming, and V5 further optimizes the CUDA kernels.


## Requirements and Setup

**System Requirements:**
- NVIDIA GPU: Turing+ (RTX 4090 recommended, 24GB VRAM)
- OS: Linux (Ubuntu 20.04)
- CUDA Toolkit: 12.1-12.9
- TensorRT: 10.13 (**versions must match exactly**)
- Python: 3.10

**Installation:**
```bash
# 1. System dependencies (TensorRT runtime + development files)
sudo apt-get update
sudo apt-get install -y tensorrt libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev libnvinfer-samples
sudo apt-get install -y libgl1 libglib2.0-0  # OpenCV dependencies
sudo ldconfig

# 2. Python environment
conda create -n upscaler python=3.10 -y
conda activate upscaler

# 3. Python packages
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir "tensorrt==10.13.0.35"  # match your system version
pip install --no-cache-dir "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
pip install onnx pycuda pynvml tqdm basicsr==1.4.2

# 4. Verify installation
python -c "import tensorrt as trt; print('TensorRT version:', trt.__version__)"
```

**Building the Engine:**
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=realesrgan_x2.onnx \
  --saveEngine=realesrgan_x2.plan \
  --fp16 \
  --memPoolSize=workspace:20000 \
  --minShapes=input:1x3x360x640 \
  --optShapes=input:1x3x360x640 \
  --maxShapes=input:1x3x360x640
```

**Compiling CUDA Kernels:**
```bash
nvcc -std=c++17 -O3 -use_fast_math -extra-device-vectorization -Xptxas=-O3,-v -Xcompiler=-O3,-ffast-math,-funroll-loops,-march=native -maxrregcount=64 --ptxas-options=-allow-expensive-optimizations=true,--warn-spills,--warn-lmem-usage,--warn-double-usage -lineinfo -arch=sm_89 -ptx postprocess_kernel_v5.cu -o postprocess_kernel_v5.ptx
```

### Usage Examples

Running the pipeline:
```bash
python upscaler_v5_ultimate.py <IMAGE_FOLDER> <ENGINE_PATH>
```

Benchmarking the pipeline (V1 - V5):
```bash
python benchmarker.py <IMAGE_FOLDER> <ENGINE_PATH>
```

