import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import sys
import os
from glob import glob

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python upscaler_v3_gpu_kernels.py <IMAGE_FOLDER> <ENGINE_PATH>")
    print("Example: python upscaler_v3_gpu_kernels.py frames/ realesrgan_x2.plan")
    sys.exit(1)

IMAGE_FOLDER = sys.argv[1]
ENGINE_PATH = sys.argv[2]

def load_test_images(folder_path, max_images=50):
    """Load test images from folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(image_paths)[:max_images]

# Load pre-compiled CUDA kernels from PTX file (much faster than inline compilation)
def load_cuda_kernels():
    """Load pre-compiled CUDA kernels from PTX file"""
    try:
        kernel_mod = cuda.module_from_file("postprocess_kernel_v3.ptx")
        enhance_kernel = kernel_mod.get_function("enhance_kernel")
        bgr_u8_to_rgb_f16_planar = kernel_mod.get_function("bgr_u8_to_rgb_f16_planar")
        # postprocess_kernel = kernel_mod.get_function("postprocess_kernel")
        print("✅ Loaded pre-compiled CUDA kernels from postprocess_kernel_v3.ptx")
        return enhance_kernel, bgr_u8_to_rgb_f16_planar
    except Exception as e:
        print(f"❌ Failed to load CUDA kernels: {e}")
        print("Make sure postprocess_kernel_v3.ptx exists in the current directory")
        raise


class GPUKernelsPipeline:
    """V3: Custom CUDA kernels for format conversion and post-processing"""
    
    def __init__(self, engine_path, max_image_size=(3, 360, 640)):
        """Initialize with custom CUDA kernels"""
        print("Initializing V3 GPU Kernels Pipeline...")
        
        # Persistent CUDA context (from V2)
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()
        
        # === V3 OPTIMIZATION: Load pre-compiled CUDA kernels ===
        self.enhance_kernel, self.bgr_u8_to_rgb_f16_planar_kernel = load_cuda_kernels()
        
        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.engine.create_execution_context()

        self.input_tensor_name = self.engine.get_tensor_name(0)
        self.output_tensor_name = self.engine.get_tensor_name(1)
        
        # Pre-allocate GPU buffers (from V2)
        max_input_size = int(np.prod(max_image_size) * 2)  # FP16
        max_output_shape = self.trt_context.get_tensor_shape(self.output_tensor_name)
        max_output_size = int(np.prod(max_output_shape) * 2)  # FP16
        
        print(f"Pre-allocating GPU buffers:")
        print(f"  Input buffer: {max_input_size/1024:.1f}KB") 
        print(f"  Output buffer: {max_output_size/1024:.1f}KB")
        
        # === V3 OPTIMIZATION: Additional GPU buffers for kernels ===
        # Raw image buffer (uint8 RGB)
        self.max_raw_size = int(max_image_size[1] * max_image_size[2] * 3)  # H*W*3 uint8
        self.d_raw = cuda.mem_alloc(self.max_raw_size)
        
        # TensorRT input buffer (FP16 CHW)
        self.d_input = cuda.mem_alloc(max_input_size)
        
        # TensorRT output buffer (FP16 CHW) 
        self.d_output = cuda.mem_alloc(max_output_size)
        
        # Final output buffer (uint8 HWC)
        # max_output_shape is (batch, channels, height, width) = (1, 3, 720, 1280)
        self.output_height = max_output_shape[2]  # height = 720
        self.output_width = max_output_shape[3]   # width = 1280
        self.max_final_size = int(self.output_height * self.output_width * 3)  # H*W*3 uint8
        
        # HWC output buffer (uint8 HWC)
        self.d_hwc_final_output = cuda.mem_alloc(self.max_final_size)
        
        # CPU output buffer - correct HWC shape
        self.max_output_shape = max_output_shape
        self.final_output = np.empty((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        print("V3 GPU Kernels Pipeline initialization complete!")
    
    def process_image_gpu(self, img_path):
        """Process image entirely on GPU with custom kernels"""
        # === MINIMAL CPU: Just load raw image data ===
        raw_img = cv2.imread(img_path)
        h, w = raw_img.shape[0], raw_img.shape[1]
        
        # Set TensorRT input shape
        batch_size = 1
        self.trt_context.set_input_shape(self.input_tensor_name, (batch_size, 3, h, w))
                
        # 1. Upload raw uint8 image to GPU
        # Ensure array is contiguous for CUDA transfer
        raw_img_contiguous = np.ascontiguousarray(raw_img)
        cuda.memcpy_htod(self.d_raw, raw_img_contiguous)
        
        # 2. Custom kernel: uint8 BGR -> FP16 RGB CHW format conversion
        block_size = (16, 16, 1)
        grid_size = ((w + 15) // 16, (h + 15) // 16, 1)
        
        self.bgr_u8_to_rgb_f16_planar_kernel(
            self.d_raw, self.d_input, np.int32(w), np.int32(h),
            block=block_size, grid=grid_size
        )
        
        # 3. TensorRT inference (still on GPU)
        self.trt_context.set_tensor_address(self.input_tensor_name, int(self.d_input))
        self.trt_context.set_tensor_address(self.output_tensor_name, int(self.d_output))
        
        success = self.trt_context.execute_v2([int(self.d_input), int(self.d_output)])
        if not success:
            raise RuntimeError(f"TensorRT inference failed for image: {img_path}")
        
        # Get output dimensions
        output_shape = self.trt_context.get_tensor_shape(self.output_tensor_name)
        out_h, out_w = output_shape[2], output_shape[3]  # Assuming NCHW
        
        
        # 4. Custom kernel: CAS enhancement + final conversion to uint8
        enhance_block = (16, 16, 1)
        enhance_grid = ((out_w + 15) // 16, (out_h + 15) // 16, 1)
        
        
        self.enhance_kernel(
            self.d_output, self.d_hwc_final_output,  # Output directly to final buffer
            np.int32(out_w), np.int32(out_h),
            np.float32(0.3),  # sharpening_strength - DISABLED to test base image
            np.float32(0.0), # edge_enhance - DISABLED to test base image  
            block=enhance_block, grid=enhance_grid
        )
        
        # 6. Single GPU -> CPU transfer at the end
        # Create a contiguous buffer for the transfer
        temp_output = np.empty((out_h, out_w, 3), dtype=np.uint8)
        cuda.memcpy_dtoh(temp_output, self.d_hwc_final_output)
        
        self.final_output[:out_h, :out_w, :] = temp_output
        
        return self.final_output[:out_h, :out_w, :].copy()
    
    def cleanup(self):
        """Clean up CUDA context"""
        # Clean up TensorRT resources first
        if hasattr(self, 'trt_context'):
            del self.trt_context
        if hasattr(self, 'engine'):
            del self.engine
        
        # Then clean up CUDA context
        if hasattr(self, 'context'):
            self.context.pop()
            self.context.detach()

def pipeline_v3_gpu_kernels(image_paths, engine_path):
    """V3: Custom CUDA kernels for format conversion and GPU post-processing"""
    
    # Initialize GPU kernels pipeline
    pipeline = GPUKernelsPipeline(engine_path)
    
    try:
        times = []
        results = []
        
        print(f"Processing {len(image_paths)} images with V3 GPU kernels pipeline...")
        print("✅ All processing (except image loading) now on GPU!")
        
        for i, img_path in enumerate(image_paths):
            start = time.time()
            
            # === V3 OPTIMIZATION: GPU-only processing pipeline ===
            final_result = pipeline.process_image_gpu(img_path)
            results.append(final_result)
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                avg_time = np.mean(times[-10:])
                fps = 1.0 / avg_time
                print(f"Processed {i+1}/{len(image_paths)} images. "
                      f"Recent avg: {avg_time*1000:.1f}ms/frame ({fps:.1f} FPS)")
    
    finally:
        # Clean up pipeline
        pipeline.cleanup()
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n=== V3 GPU KERNELS RESULTS ===")
    print(f"Average time per frame: {avg_time*1000:.1f}ms")
    print(f"Theoretical FPS: {fps:.1f}")
    print(f"Total processing time: {sum(times):.1f}s")
    print(f"V3 Optimizations:")
    print(f"  ✅ Persistent CUDA context (from V2)")
    print(f"  ✅ Pre-allocated GPU buffers (from V2)")
    print(f"  ✅ Memory pool reuse (from V2)")
    print(f"  ✅ Custom format conversion kernels")
    print(f"  ✅ GPU-based CAS post-processing")
    print(f"  ✅ Eliminated CPU preprocessing/postprocessing")
    
    return avg_time, results

if __name__ == "__main__":
    # Load test images
    image_paths = load_test_images(IMAGE_FOLDER, max_images=300)
    
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images")
    
    try:
        avg_time, results = pipeline_v3_gpu_kernels(image_paths, ENGINE_PATH)
        
        # Optional: Save a few results to verify correctness
        for i in range(min(3, len(results))):
            output_path = f"v3_result_{i}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(results[i], cv2.COLOR_RGB2BGR))
            print(f"Saved sample result: {output_path}")
            
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
