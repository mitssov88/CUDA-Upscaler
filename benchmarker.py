#!/usr/bin/env python3
"""
Complete benchmark suite with minimal overhead metrics
Usage: python benchmarker.py test_images/ realesrgan_x2_fp16_360.plan
"""

import sys
import os
from pathlib import Path
from glob import glob

# Import your pipeline versions
from upscaler_v1_naive import pipeline_v1_naive
from upscaler_v2_preallocated import pipeline_v2_preallocated  
from upscaler_v3_gpu_kernels import pipeline_v3_gpu_kernels
from upscaler_v4_async import pipeline_v4_async_streams
from upscaler_v5_ultimate import pipeline_v5_ultimate

# Import metrics system
from metrics_measurement import (
    benchmark_with_minimal_overhead,
    print_comparison_table,
    CoreMetrics
)

def load_test_images(folder_path, max_images=50):
    """Load test images from folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(image_paths)[:max_images]

def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmarker.py <IMAGE_FOLDER> <ENGINE_PATH>")
        sys.exit(1)
    
    image_folder = sys.argv[1] 
    engine_path = sys.argv[2]
    
    # Load test images
    image_paths = load_test_images(image_folder, max_images=300)
    if not image_paths:
        print(f"No images found in {image_folder}")
        sys.exit(1)
    
    print(f"Benchmarking {len(image_paths)} images")
    print(f"Engine: {engine_path}")
    print("-" * 50)
    
    # Define benchmark configurations with different sample rates
    # Use lower sample rates for slower versions to minimize overhead
    pipeline_configs = [
        ("V1_Naive", lambda imgs: pipeline_v1_naive(imgs, engine_path), 1.0),        # 1Hz - minimal overhead
        ("V2_PreAlloc", lambda imgs: pipeline_v2_preallocated(imgs, engine_path), 1.0),  # 2Hz - low overhead
        ("V3_Async", lambda imgs: pipeline_v3_gpu_kernels(imgs, engine_path), 1.0),           # 2Hz 
        ("V4_Kernels", lambda imgs: pipeline_v4_async_streams(imgs, engine_path), 1.0),       # 2Hz
        ("V5_Ultimate", lambda imgs: pipeline_v5_ultimate(imgs, engine_path), 1.0), # 2Hz
    ]
    
    # Collect metrics for all versions
    all_metrics = {}
    
    for version_name, pipeline_func, sample_rate_hz in pipeline_configs:
        print(f"\n{'='*20} BENCHMARKING {version_name} {'='*20}")
        
        try:
            # Run benchmark with minimal overhead metrics
            metrics = benchmark_with_minimal_overhead(
                pipeline_func=pipeline_func,
                image_paths=image_paths,
                version_name=version_name,
                sample_rate_hz=sample_rate_hz
            )
            
            all_metrics[version_name] = metrics
            
        except Exception as e:
            print(f"Error benchmarking {version_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comprehensive comparison
    if all_metrics:
        print_comparison_table(all_metrics)
        
        # Generate summary insights
        generate_performance_insights(all_metrics)
        
        # Save results to file
        save_benchmark_results(all_metrics, "benchmark_results.txt")
    
    print("\nBenchmarking complete!")

def generate_performance_insights(metrics_dict):
    """Generate key insights from benchmark results"""
    if len(metrics_dict) < 2:
        print(f"\n{'='*60}")
        print("                    BENCHMARK RESULTS")
        print('='*60)
        version, metrics = list(metrics_dict.items())[0]
        print(f"📊 {version} Performance:")
        print(f"   Time per frame: {metrics.avg_time_per_frame*1000:.1f}ms")
        print(f"   Theoretical FPS: {metrics.theoretical_fps:.1f}")
        print(f"   CPU usage: {metrics.avg_cpu_usage:.1f}%")
        print(f"   GPU usage: {metrics.avg_gpu_usage:.1f}%")
        print(f"   GPU memory: {metrics.avg_gpu_memory_util:.1f}% utilization")
        print('='*60)
        return
        
    versions = list(metrics_dict.keys())
    baseline = versions[0]
    final = versions[-1]
    
    baseline_metrics = metrics_dict[baseline]
    final_metrics = metrics_dict[final]
    
    # Calculate improvements
    fps_improvement = final_metrics.theoretical_fps / baseline_metrics.theoretical_fps
    cpu_reduction = baseline_metrics.avg_cpu_usage - final_metrics.avg_cpu_usage  
    gpu_increase = final_metrics.avg_gpu_usage - baseline_metrics.avg_gpu_usage
    
    print(f"\n{'='*60}")
    print("                    KEY INSIGHTS")
    print('='*60)
    print(f"🚀 Overall speedup: {fps_improvement:.1f}x faster ({baseline_metrics.theoretical_fps:.1f} → {final_metrics.theoretical_fps:.1f} FPS)")
    print(f"🔧 CPU load reduction: {cpu_reduction:.1f}% ({baseline_metrics.avg_cpu_usage:.1f}% → {final_metrics.avg_cpu_usage:.1f}%)")
    print(f"⚡ GPU utilization increase: +{gpu_increase:.1f}% ({baseline_metrics.avg_gpu_usage:.1f}% → {final_metrics.avg_gpu_usage:.1f}%)")
    print(f"💾 GPU memory usage: {final_metrics.avg_gpu_memory_used:.0f}MB ({final_metrics.avg_gpu_memory_used/final_metrics.gpu_memory_total*100:.1f}% of {final_metrics.gpu_memory_total:.0f}MB)")
    print(f"🖥️  CPU memory usage: {final_metrics.avg_cpu_memory_used:.0f}MB")
    
    # Identify biggest performance jumps
    prev_fps = baseline_metrics.theoretical_fps
    printed_title = False
    for i, (version, metrics) in enumerate(list(metrics_dict.items())[1:], 1):
        improvement = metrics.theoretical_fps / prev_fps
        if improvement > 1.3:  # Significant improvement
            if not printed_title:
                print(f"\n📈 Biggest performance improvements:")
                printed_title = True
            print(f"   {versions[i-1]} → {version}: {improvement:.1f}x speedup")
        prev_fps = metrics.theoretical_fps
    
    print('='*60)

def save_benchmark_results(metrics_dict, filename):
    """Save detailed benchmark results to file"""
    with open(filename, 'w') as f:
        f.write("CUDA Pipeline Optimization Benchmark Results\n")
        f.write("="*50 + "\n\n")
        
        for version, metrics in metrics_dict.items():
            f.write(f"{version} Results:\n")
            f.write(f"  Time per frame: {metrics.avg_time_per_frame*1000:.2f}ms\n")
            f.write(f"  Theoretical FPS: {metrics.theoretical_fps:.2f}\n")
            f.write(f"  CPU usage: {metrics.avg_cpu_usage:.1f}% (max: {metrics.max_cpu_usage:.1f}%)\n")
            f.write(f"  CPU memory: {metrics.avg_cpu_memory_used:.0f}MB (max: {metrics.max_cpu_memory_used:.0f}MB)\n")
            f.write(f"  GPU usage: {metrics.avg_gpu_usage:.1f}% (max: {metrics.max_gpu_usage:.1f}%)\n")
            f.write(f"  GPU memory: {metrics.avg_gpu_memory_used:.0f}MB (max: {metrics.max_gpu_memory_used:.0f}MB)\n")
            f.write(f"  GPU memory util: {metrics.avg_gpu_memory_used/metrics.gpu_memory_total*100:.1f}%\n")
            f.write("\n")
    
    print(f"Detailed results saved to: {filename}")

if __name__ == "__main__":
    main()