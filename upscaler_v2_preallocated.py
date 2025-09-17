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
    print("Usage: python upscaler_v2_preallocated.py <IMAGE_FOLDER> <ENGINE_PATH>")
    print("Example: python upscaler_v2_preallocated.py frames/ realesrgan_x2.plan")
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

def preprocess_cpu(image_path, target_size=(640, 360)):
    """CPU-based image preprocessing - OPTIMIZED VERSION"""
    # Load with OpenCV (CPU)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize on CPU
    img = cv2.resize(img, target_size)
    
    # Color conversion on CPU
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize on CPU (optimized)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Convert to CHW format on CPU (optimized)
    img_chw = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
    
    # Convert to FP16 on CPU (optimized)
    img_fp16 = img_chw.astype(np.float16)
    
    return img_fp16

def postprocess_cpu(output_array):
    """CPU-based post-processing - PREALLOCATED VERSION"""
    # Convert back to float32 on CPU
    result_f32 = output_array.astype(np.float32)
    
    # Simple clamp on CPU (no CAS enhancement)
    result_clamped = np.clip(result_f32, 0.0, 1.0)
    
    # Convert to uint8 on CPU
    result_u8 = (result_clamped * 255.0).astype(np.uint8)
    
    # Remove batch dimension if present (output_array is (batch_size, C, H, W))
    if len(result_u8.shape) == 4:
        result_u8 = result_u8[0]  # Take first (and only) batch item
    
    # Convert CHW -> HWC on CPU
    result_hwc = np.transpose(result_u8, (1, 2, 0))
    
    return result_hwc

class PreallocatedPipeline:
    """V2: Pre-allocated memory pipeline with persistent CUDA context"""
    
    def __init__(self, engine_path, max_image_size=(3, 360, 640)):
        """Initialize once with pre-allocated buffers"""
        print("Initializing V2 Pre-allocated Pipeline...")
        
        # === V2 OPTIMIZATION: Persistent CUDA context ===
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()
        
        # Load TensorRT engine once
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.engine.create_execution_context()

        self.input_tensor_name = self.engine.get_tensor_name(0)
        self.output_tensor_name = self.engine.get_tensor_name(1)
        
        # === V2 OPTIMIZATION: Pre-allocate all GPU buffers ===
        # Calculate maximum buffer sizes
        max_input_size = int(np.prod(max_image_size) * 2)  # FP16
        max_output_shape = self.trt_context.get_tensor_shape(self.output_tensor_name)
        max_output_size = int(np.prod(max_output_shape) * 2)  # FP16
        
        print(f"Pre-allocating GPU buffers:")
        print(f"  Input buffer: {max_input_size/1024:.1f}KB")
        print(f"  Output buffer: {max_output_size/1024:.1f}KB")
        
        # === V2 OPTIMIZATION: Memory pool reuse ===
        self.d_input = cuda.mem_alloc(max_input_size)
        self.d_output = cuda.mem_alloc(max_output_size)
        
        # Pre-allocate CPU buffers for reuse
        self.max_output_shape = max_output_shape
        self.output_array = np.empty(max_output_shape, dtype=np.float16)
        
        print("V2 Pipeline initialization complete!")
    
    def process_image(self, img_path):
        """Process single image using pre-allocated buffers"""
        # CPU preprocessing
        img_processed = preprocess_cpu(img_path)
        h, w = img_processed.shape[1], img_processed.shape[2]
        
        # Set input shape for this image
        batch_size = 1
        self.trt_context.set_input_shape(self.input_tensor_name, (batch_size, 3, h, w))
        
        # === V2 OPTIMIZATION: Reuse pre-allocated memory ===
        # CPU -> GPU transfer (reusing d_input)
        # Ensure array is contiguous for CUDA transfer
        img_processed_contiguous = np.ascontiguousarray(img_processed)
        cuda.memcpy_htod(self.d_input, img_processed_contiguous)
        
        # TensorRT inference (reusing d_output)
        self.trt_context.set_tensor_address(self.input_tensor_name, int(self.d_input))
        self.trt_context.set_tensor_address(self.output_tensor_name, int(self.d_output))
        success = self.trt_context.execute_v2([int(self.d_input), int(self.d_output)])
        
        if not success:
            raise RuntimeError(f"TensorRT inference failed for image: {img_path}")
        
        # GPU -> CPU transfer (reusing output_array)
        cuda.memcpy_dtoh(self.output_array, self.d_output)
        
        # CPU post-processing
        final_result = postprocess_cpu(self.output_array)
        
        return final_result
    
    def cleanup(self):
        """Clean up CUDA context and buffers"""
        # Clean up TensorRT resources first
        if hasattr(self, 'trt_context'):
            del self.trt_context
        if hasattr(self, 'engine'):
            del self.engine
        # Then clean up CUDA context
        if hasattr(self, 'context'):
            self.context.pop()
            self.context.detach()

def pipeline_v2_preallocated(image_paths, engine_path):
    """V2: Pre-allocated memory pipeline with persistent CUDA context"""
    
    # Initialize pipeline once with pre-allocated buffers
    pipeline = PreallocatedPipeline(engine_path)
    
    try:
        times = []
        results = []
        
        print(f"Processing {len(image_paths)} images with V2 pre-allocated pipeline...")
        
        for i, img_path in enumerate(image_paths):
            start = time.time()
            
            # === V2 OPTIMIZATION: Reuse pre-allocated buffers ===
            final_result = pipeline.process_image(img_path)
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
    
    print(f"\n=== V2 PRE-ALLOCATED RESULTS ===")
    print(f"Average time per frame: {avg_time*1000:.1f}ms")
    print(f"Theoretical FPS: {fps:.1f}")
    print(f"Total processing time: {sum(times):.1f}s")
    print(f"V2 Optimizations:")
    print(f"  ✅ Persistent CUDA context")
    print(f"  ✅ Pre-allocated GPU buffers")
    print(f"  ✅ Memory pool reuse")
    
    return avg_time, results

if __name__ == "__main__":
    # Load test images
    image_paths = load_test_images(IMAGE_FOLDER, max_images=300)
    
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images")
    
    try:
        avg_time, results = pipeline_v2_preallocated(image_paths, ENGINE_PATH)
        
        # Optional: Save a few results to verify correctness
        for i in range(min(3, len(results))):
            output_path = f"v2_result_{i}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(results[i], cv2.COLOR_RGB2BGR))
            print(f"Saved sample result: {output_path}")
            
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
