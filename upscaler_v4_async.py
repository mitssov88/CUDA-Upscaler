import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import sys
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import threading

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python upscaler_v4_async.py <IMAGE_FOLDER> <ENGINE_PATH>")
    print("Example: python upscaler_v4_async.py frames/ realesrgan_x2.plan")
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
        print("✅ Loaded pre-compiled CUDA kernels from postprocess_kernel_v3.ptx")
        return enhance_kernel, bgr_u8_to_rgb_f16_planar
    except Exception as e:
        print(f"❌ Failed to load CUDA kernels: {e}")
        print("Make sure postprocess_kernel_v3.ptx exists in the current directory")
        raise


class AsyncStreamProcessor:
    """Individual processing stream for async pipeline"""
    
    def __init__(self, context, stream_id, engine, kernel_enhance, kernel_u8_to_f16, max_image_size=(3, 360, 640)):
        self.ctx = context
        self.stream_id = stream_id
        self.enhance_kernel = kernel_enhance
        self.bgr_u8_to_rgb_f16_planar_kernel = kernel_u8_to_f16
        
        # Create dedicated CUDA stream for this processor
        self.stream = cuda.Stream()
        
        # Create separate TensorRT context for this stream (thread-safe)
        self.trt_context = engine.create_execution_context()
        self.input_tensor_name = engine.get_tensor_name(0)
        self.output_tensor_name = engine.get_tensor_name(1)
        
        # Pre-allocate GPU buffers for this stream
        max_input_size = int(np.prod(max_image_size) * 2)  # FP16
        max_output_shape = self.trt_context.get_tensor_shape(self.output_tensor_name)
        max_output_size = int(np.prod(max_output_shape) * 2)  # FP16
        
        # Raw image buffer (uint8 RGB)
        self.max_raw_size = int(max_image_size[1] * max_image_size[2] * 3)  # H*W*3 uint8
        self.d_raw = cuda.mem_alloc(self.max_raw_size)
        
        # TensorRT input buffer (FP16 CHW)
        self.d_input = cuda.mem_alloc(max_input_size)
        
        # TensorRT output buffer (FP16 CHW) 
        self.d_output = cuda.mem_alloc(max_output_size)
        
        # Final output buffer (uint8 HWC)
        self.output_height = max_output_shape[2]
        self.output_width = max_output_shape[3]
        self.max_final_size = int(self.output_height * self.output_width * 3)
        self.d_hwc_final_output = cuda.mem_alloc(self.max_final_size)
        
        # CPU pinned memory for faster transfers
        self.pinned_input = cuda.pagelocked_empty((max_image_size[1], max_image_size[2], 3), dtype=np.uint8)
        self.pinned_output = cuda.pagelocked_empty((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        print(f"Stream {stream_id}: Initialized with dedicated buffers and CUDA stream")
    
    def process_image_async(self, raw_img):
        """Process image asynchronously using this stream"""

        self.ctx.push()
        try:
            
          h, w = raw_img.shape[0], raw_img.shape[1]
          
          # Set TensorRT input shape for this context
          batch_size = 1
          self.trt_context.set_input_shape(self.input_tensor_name, (batch_size, 3, h, w))
          
          # === V4 OPTIMIZATION: Async pipeline with overlapped transfers ===
          
          # 1. Copy to pinned memory (faster transfer preparation)
          self.pinned_input[:h, :w, :] = raw_img
          
          # 2. Async upload raw uint8 image to GPU
          cuda.memcpy_htod_async(self.d_raw, self.pinned_input[:h, :w, :], self.stream)
          
          # 3. Custom kernel: uint8 RGB -> FP16 CHW format conversion (async)
          block_size = (16, 16, 1)
          grid_size = ((w + 15) // 16, (h + 15) // 16, 1)
          
          self.bgr_u8_to_rgb_f16_planar_kernel(
              self.d_raw, self.d_input, np.int32(w), np.int32(h),
              block=block_size, grid=grid_size, stream=self.stream
          )
          
          # 4. TensorRT inference (async on stream)
          self.trt_context.set_tensor_address(self.input_tensor_name, int(self.d_input))
          self.trt_context.set_tensor_address(self.output_tensor_name, int(self.d_output))
          
          # Execute async with stream context
          success = self.trt_context.execute_async_v3(self.stream.handle)
          if not success:
              raise RuntimeError(f"TensorRT async inference failed")
          
          # Get output dimensions
          output_shape = self.trt_context.get_tensor_shape(self.output_tensor_name)
          out_h, out_w = output_shape[2], output_shape[3]
          
          # 5. Custom kernel: CAS enhancement + final conversion (async)
          enhance_block = (16, 16, 1)
          enhance_grid = ((out_w + 15) // 16, (out_h + 15) // 16, 1)
          
          self.enhance_kernel(
              self.d_output, self.d_hwc_final_output,
              np.int32(out_w), np.int32(out_h),
              np.float32(0.3),  # sharpening_strength - DISABLED to test base image
              np.float32(0.5),  # edge_enhance - DISABLED to test base image  
              block=enhance_block, grid=enhance_grid, stream=self.stream
          )
          
          # 6. Async GPU -> CPU transfer to pinned memory
          cuda.memcpy_dtoh_async(self.pinned_output[:out_h, :out_w, :], 
                                self.d_hwc_final_output, self.stream)
          
          # 7. Synchronize stream to ensure completion
          self.stream.synchronize()
          
          # 8. Return copy of result (pinned memory is reused)
          return self.pinned_output[:out_h, :out_w, :].copy()
        
        finally:
            self.ctx.pop()
    
    def cleanup(self):
        """Clean up stream resources"""
        if hasattr(self, 'trt_context'):
            del self.trt_context
        if hasattr(self, 'stream'):
            del self.stream

class MultiStreamAsyncPipeline:
    """V4: Multi-stream async pipeline with overlapped processing"""
    
    def __init__(self, engine_path, max_image_size=(3, 360, 640), num_streams=4):
        """Initialize with multiple async processing streams"""
        print(f"Initializing V4 Multi-Stream Async Pipeline with {num_streams} streams...")
        
        # Persistent CUDA context (from V2/V3)
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.retain_primary_context()
        self.context.push() 
        
        # === V4 OPTIMIZATION: Load CUDA kernels once, share across streams ===
        self.enhance_kernel, self.bgr_u8_to_rgb_f16_planar_kernel = load_cuda_kernels()
        
        # Load TensorRT engine once
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # === V4 OPTIMIZATION: Create multiple async processing streams ===
        self.num_streams = num_streams
        self.streams = []
        
        for i in range(num_streams):
            stream_processor = AsyncStreamProcessor(
                context=self.context,
                stream_id=i,
                engine=self.engine,
                kernel_enhance=self.enhance_kernel,
                kernel_u8_to_f16=self.bgr_u8_to_rgb_f16_planar_kernel,
                max_image_size=max_image_size
            )
            self.streams.append(stream_processor)
        
        print(f"V4 Multi-Stream Async Pipeline initialization complete!")
        print(f"  🚀 {num_streams} parallel processing streams")
        print(f"  ⚡ Async CUDA operations with overlapped transfers")
        print(f"  📋 Pinned memory for faster CPU<->GPU transfers")
    
    def process_batch_async(self, image_paths):
        """Process batch of images using multiple async streams"""
        results = [None] * len(image_paths)
        
        # === V4 OPTIMIZATION: Efficient single-image pipeline ===
        if len(image_paths) <= self.num_streams:
            # Small batch: Direct stream assignment without complex threading
            return self._process_small_batch(image_paths)
        
        # === V4 OPTIMIZATION: CPU preprocessing in parallel thread pool ===
        # Preprocess images in parallel on CPU while GPU processes previous batch
        with ThreadPoolExecutor(max_workers=min(4, len(image_paths))) as cpu_executor:
            preprocessing_futures = {
                cpu_executor.submit(cv2.imread, img_path): i 
                for i, img_path in enumerate(image_paths)
            }
        
        # === V4 OPTIMIZATION: Stream-based GPU processing with load balancing ===
        def process_on_stream(args):
            stream_idx, img_idx, raw_img = args
            stream_processor = self.streams[stream_idx]
            result = stream_processor.process_image_async(raw_img)
            return img_idx, result
        
        # Process preprocessed images on GPU streams
        with ThreadPoolExecutor(max_workers=self.num_streams) as gpu_executor:
            # Collect preprocessed images
            preprocessed_images = {}
            for future, img_idx in preprocessing_futures.items():
                raw_img = future.result()
                preprocessed_images[img_idx] = raw_img
            
            # Submit to GPU streams with round-robin assignment
            gpu_futures = []
            for img_idx in range(len(image_paths)):
                stream_idx = img_idx % self.num_streams
                raw_img = preprocessed_images[img_idx]
                
                future = gpu_executor.submit(process_on_stream, 
                                           (stream_idx, img_idx, raw_img))
                gpu_futures.append(future)
            
            # Collect results in order
            for future in gpu_futures:
                img_idx, result = future.result()
                results[img_idx] = result
        
        return results
    
    def _process_small_batch(self, image_paths):
        """Optimized processing for small batches (≤ num_streams)"""
        results = [None] * len(image_paths)
        
        # For small batches, use simple concurrent processing without complex threading
        def process_single_image(args):
            img_idx, img_path, stream_idx = args
            raw_img = cv2.imread(img_path)
            stream_processor = self.streams[stream_idx]
            result = stream_processor.process_image_async(raw_img)
            return img_idx, result
        
        with ThreadPoolExecutor(max_workers=len(image_paths)) as executor:
            futures = []
            for i, img_path in enumerate(image_paths):
                stream_idx = i % self.num_streams  # Use available streams
                future = executor.submit(process_single_image, (i, img_path, stream_idx))
                futures.append(future)
            
            # Collect results
            for future in futures:
                img_idx, result = future.result()
                results[img_idx] = result
        
        return results
    
    def cleanup(self):
        """Clean up all streams and contexts"""
        # Clean up all stream processors
        for stream in self.streams:
            stream.cleanup()
        
        # Clean up TensorRT resources
        if hasattr(self, 'engine'):
            del self.engine
        
        # Clean up CUDA context
        if hasattr(self, 'context'):
            self.context.pop()
            self.context.detach()

def pipeline_v4_async_streams(image_paths, engine_path):
    """V4: Multi-stream async pipeline with overlapped processing"""
    
    # Determine optimal number of streams based on workload
    num_images = len(image_paths)
    num_streams = min(4, max(2, num_images // 10))  # 2-4 streams based on batch size
    
    # Initialize multi-stream async pipeline
    pipeline = MultiStreamAsyncPipeline(engine_path, num_streams=num_streams)
    
    try:
        times = []
        all_results = []
        
        print(f"Processing {len(image_paths)} images with V4 multi-stream async pipeline...")
        print(f"🚀 Using {num_streams} parallel processing streams")
        print(f"⚡ Async operations with overlapped CPU/GPU work")
        
        # === V4 OPTIMIZATION: Adaptive batch processing for single-image models ===
        # For single-image TensorRT models, optimal batch size = number of streams
        # This allows each stream to work on one image while others process in parallel
        optimal_batch_size = max(num_streams, min(num_streams * 2, 8))  # 2-8 images per batch
        
        for batch_start in range(0, len(image_paths), optimal_batch_size):
            batch_end = min(batch_start + optimal_batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            start = time.time()
            
            # === V4 OPTIMIZATION: Async batch processing ===
            batch_results = pipeline.process_batch_async(batch_paths)
            all_results.extend(batch_results)
            
            elapsed = time.time() - start
            # Calculate per-image time more accurately for small batches
            per_image_time = elapsed / len(batch_paths)
            times.extend([per_image_time] * len(batch_paths))
            
            processed = batch_end
            if processed % max(10, optimal_batch_size) == 0 or processed == len(image_paths):
                recent_count = min(20, len(times))
                avg_time = np.mean(times[-recent_count:])
                fps = 1.0 / avg_time
                throughput = len(batch_paths) / elapsed  # Actual batch throughput
                print(f"Processed {processed}/{len(image_paths)} images. "
                      f"Batch: {len(batch_paths)} imgs in {elapsed*1000:.1f}ms "
                      f"({throughput:.1f} imgs/sec, {avg_time*1000:.1f}ms/img avg)")
    
    finally:
        # Clean up pipeline
        pipeline.cleanup()
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n=== V4 MULTI-STREAM ASYNC RESULTS ===")
    print(f"Average time per frame: {avg_time*1000:.1f}ms")
    print(f"Theoretical FPS: {fps:.1f}")
    print(f"Total processing time: {sum(times):.1f}s")
    print(f"V4 Optimizations:")
    print(f"  ✅ Persistent CUDA context (from V2)")
    print(f"  ✅ Pre-allocated GPU buffers (from V2)")
    print(f"  ✅ Custom format conversion kernels (from V3)")
    print(f"  ✅ GPU-based CAS post-processing (from V3)")
    print(f"  ✅ Multiple async CUDA streams ({num_streams} streams)")
    print(f"  ✅ Overlapped CPU preprocessing and GPU processing")
    print(f"  ✅ Pinned memory for faster transfers")
    print(f"  ✅ Thread pool parallelization")
    
    return avg_time, all_results

if __name__ == "__main__":
    # Load test images
    image_paths = load_test_images(IMAGE_FOLDER, max_images=300)
    
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images")
    
    try:
        avg_time, results = pipeline_v4_async_streams(image_paths, ENGINE_PATH)
        
        # Optional: Save a few results to verify correctness
        for i in range(min(3, len(results))):
            output_path = f"v4_result_{i}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(results[i], cv2.COLOR_RGB2BGR))
            print(f"Saved sample result: {output_path}")
            
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()