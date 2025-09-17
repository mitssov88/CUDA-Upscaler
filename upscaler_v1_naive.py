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
    print("Usage: python upscaler_v1_naive.py <IMAGE_FOLDER> <ENGINE_PATH>")
    print("Example: python upscaler_v1_naive.py frames/ realesrgan_x2.plan")
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
    """CPU-based image preprocessing - NAIVE VERSION"""
    # Load with OpenCV (CPU)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize on CPU
    img = cv2.resize(img, target_size)
    
    # Color conversion on CPU
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize on CPU (slow!)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Convert to CHW format on CPU (slow!)
    img_chw = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
    
    # Convert to FP16 on CPU (very slow!)
    img_fp16 = img_chw.astype(np.float16)
    
    return img_fp16

def postprocess_cpu(output_array):
    """CPU-based post-processing - NAIVE VERSION"""
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

def pipeline_v1_naive(image_paths, engine_path):
    """V1: Naive CPU pipeline with memory allocation per image"""
    
    # Initialize CUDA (create/destroy context per run - inefficient!)
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    
    try:
        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
            trt_context = engine.create_execution_context()

        input_tensor_name = engine.get_tensor_name(0)
        output_tensor_name = engine.get_tensor_name(1)
        
        times = []
        results = []
        
        print(f"Processing {len(image_paths)} images with naive pipeline...")
        
        for i, img_path in enumerate(image_paths):
            start = time.time()
            
            # CPU preprocessing (very slow)
            img_processed = preprocess_cpu(img_path)
            h, w = img_processed.shape[1], img_processed.shape[2]
            
            # Set input shape for each image (inefficient)
            batch_size = 1
            trt_context.set_input_shape(input_tensor_name, (batch_size, 3, h, w))
            output_shape = trt_context.get_tensor_shape(output_tensor_name)
            
            # === MEMORY ALLOCATION PER IMAGE  ===
            input_size = int(np.prod((batch_size, 3, h, w)) * 2)  # FP16
            output_size = int(np.prod(output_shape) * 2)  # FP16
            
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            # === SYNCHRONOUS MEMORY TRANSFERS ===
            # CPU -> GPU (blocking transfer)
            # Ensure array is contiguous for CUDA transfer
            img_processed_contiguous = np.ascontiguousarray(img_processed)
            cuda.memcpy_htod(d_input, img_processed_contiguous)
            
            # === SYNCHRONOUS INFERENCE ===
            trt_context.set_tensor_address(input_tensor_name, int(d_input))
            trt_context.set_tensor_address(output_tensor_name, int(d_output))
            # Synchronous execution (no streams)
            success = trt_context.execute_v2([int(d_input), int(d_output)])
            
            if not success:
                print(f"TensorRT inference failed for image {i}")
                continue
            
            # === SYNCHRONOUS MEMORY TRANSFER BACK ===
            # GPU -> CPU (blocking transfer)
            output_array = np.empty(output_shape, dtype=np.float16)
            cuda.memcpy_dtoh(output_array, d_output)
            
            # === FREE MEMORY AFTER EACH IMAGE (inefficient!) ===
            d_input.free()
            d_output.free()
            
            # === CPU POST-PROCESSING (slow, basic) ===
            final_result = postprocess_cpu(output_array)
            results.append(final_result)
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                avg_time = np.mean(times[-10:])
                fps = 1.0 / avg_time
                print(f"Processed {i+1}/{len(image_paths)} images. "
                      f"Recent avg: {avg_time*1000:.1f}ms/frame ({fps:.1f} FPS)")
    
    finally:
        # Clean up TensorRT resources first
        if 'trt_context' in locals():
            del trt_context
        if 'engine' in locals():
            del engine
        if 'runtime' in locals():
            del runtime
        
        # Then clean up CUDA context
        context.pop()
        context.detach()
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n=== V1 NAIVE RESULTS ===")
    print(f"Average time per frame: {avg_time*1000:.1f}ms")
    print(f"Theoretical FPS: {fps:.1f}")
    print(f"Total processing time: {sum(times):.1f}s")
    
    return avg_time, results

if __name__ == "__main__":
    # Load test images
    image_paths = load_test_images(IMAGE_FOLDER, max_images=300)
    
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images")
    
    try:
        avg_time, results = pipeline_v1_naive(image_paths, ENGINE_PATH)
        # Optional: Save a few results to verify correctness
        for i in range(min(3, len(results))):
            output_path = f"v1_result_{i}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(results[i], cv2.COLOR_RGB2BGR))
            print(f"Saved sample result: {output_path}")
            
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()