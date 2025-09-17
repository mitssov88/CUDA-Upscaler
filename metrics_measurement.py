import psutil
import time
import threading
import numpy as np
import pynvml
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

@dataclass
class CoreMetrics:
    """Essential performance metrics only"""
    # Timing
    avg_time_per_frame: float
    total_processing_time: float
    theoretical_fps: float
    
    # CPU
    avg_cpu_usage: float
    max_cpu_usage: float
    avg_cpu_memory_used: float  # MB
    max_cpu_memory_used: float  # MB
    
    # GPU
    avg_gpu_usage: float
    max_gpu_usage: float
    avg_gpu_memory_used: float  # MB
    max_gpu_memory_used: float  # MB
    gpu_memory_total: float     # MB

class MinimalMonitor:
    """Ultra-lightweight monitor for essential metrics only"""
    
    def __init__(self, sample_interval=0.5):  # Default 2Hz
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Initialize GPU monitoring
        self._init_gpu()
        
        # Get process handle for CPU memory
        self.process = psutil.Process()
        
        # Pre-allocate small arrays (5 minutes max at 2Hz = 600 samples)
        max_samples = int(300 / sample_interval)
        self.max_samples = max_samples
        self.reset_metrics()
    
    def _init_gpu(self):
        """Initialize GPU monitoring once"""
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Cache total GPU memory
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_memory_total_mb = gpu_mem.total / (1024**2)
            
            self.gpu_available = True
            print(f"GPU monitoring: {self.gpu_memory_total_mb:.0f}MB total VRAM")
        except Exception as e:
            print(f"GPU monitoring unavailable: {e}")
            self.gpu_available = False
            self.gpu_memory_total_mb = 0
    
    def reset_metrics(self):
        """Reset metric arrays"""
        self.cpu_usage = np.zeros(self.max_samples, dtype=np.float32)
        self.cpu_memory = np.zeros(self.max_samples, dtype=np.float32)
        self.gpu_usage = np.zeros(self.max_samples, dtype=np.float32)
        self.gpu_memory = np.zeros(self.max_samples, dtype=np.float32)
        self.sample_count = 0
    
    def _monitor_loop(self):
        """Lightweight monitoring loop"""
        while self.monitoring and self.sample_count < self.max_samples:
            idx = self.sample_count
            
            # CPU usage (fast method)
            self.cpu_usage[idx] = psutil.cpu_percent(interval=0.01)
            
            # CPU memory (process-specific)
            try:
                mem_info = self.process.memory_info()
                self.cpu_memory[idx] = mem_info.rss / (1024**2)  # RSS in MB
            except:
                self.cpu_memory[idx] = 0
            
            # GPU metrics (if available)
            if self.gpu_available:
                try:
                    # GPU utilization
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.gpu_usage[idx] = gpu_util.gpu
                    
                    # GPU memory
                    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_memory[idx] = gpu_mem.used / (1024**2)
                    
                except Exception:
                    if idx == 0:  # Print error once
                        print("GPU monitoring error, using zeros")
                    self.gpu_usage[idx] = 0
                    self.gpu_memory[idx] = 0
            else:
                self.gpu_usage[idx] = 0
                self.gpu_memory[idx] = 0
            
            self.sample_count += 1
            time.sleep(self.sample_interval)
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if self.monitoring:
            return
        
        self.reset_metrics()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Monitoring started: {1/self.sample_interval:.1f}Hz sampling")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print(f"Monitoring stopped: {self.sample_count} samples")
    
    def get_metrics(self, processing_time: float, num_frames: int) -> CoreMetrics:
        """Calculate final metrics"""
        if self.sample_count == 0:
            return self._zero_metrics()
        
        # Use only valid samples
        n = self.sample_count
        
        # Timing
        avg_time_per_frame = processing_time / num_frames
        theoretical_fps = 1.0 / avg_time_per_frame
        
        # CPU metrics
        cpu_usage_data = self.cpu_usage[:n]
        cpu_memory_data = self.cpu_memory[:n]
        
        avg_cpu_usage = float(np.mean(cpu_usage_data))
        max_cpu_usage = float(np.max(cpu_usage_data))
        avg_cpu_memory = float(np.mean(cpu_memory_data))
        max_cpu_memory = float(np.max(cpu_memory_data))
        
        # GPU metrics
        gpu_usage_data = self.gpu_usage[:n]
        gpu_memory_data = self.gpu_memory[:n]
        
        avg_gpu_usage = float(np.mean(gpu_usage_data))
        max_gpu_usage = float(np.max(gpu_usage_data))
        avg_gpu_memory = float(np.mean(gpu_memory_data))
        max_gpu_memory = float(np.max(gpu_memory_data))
        
        return CoreMetrics(
            avg_time_per_frame=avg_time_per_frame,
            total_processing_time=processing_time,
            theoretical_fps=theoretical_fps,
            avg_cpu_usage=avg_cpu_usage,
            max_cpu_usage=max_cpu_usage,
            avg_cpu_memory_used=avg_cpu_memory,
            max_cpu_memory_used=max_cpu_memory,
            avg_gpu_usage=avg_gpu_usage,
            max_gpu_usage=max_gpu_usage,
            avg_gpu_memory_used=avg_gpu_memory,
            max_gpu_memory_used=max_gpu_memory,
            gpu_memory_total=self.gpu_memory_total_mb
        )
    
    def _zero_metrics(self) -> CoreMetrics:
        """Return zero metrics if no data collected"""
        return CoreMetrics(
            avg_time_per_frame=0.0, total_processing_time=0.0, theoretical_fps=0.0,
            avg_cpu_usage=0.0, max_cpu_usage=0.0,
            avg_cpu_memory_used=0.0, max_cpu_memory_used=0.0,
            avg_gpu_usage=0.0, max_gpu_usage=0.0,
            avg_gpu_memory_used=0.0, max_gpu_memory_used=0.0,
            gpu_memory_total=self.gpu_memory_total_mb
        )

# Simple context manager
@contextmanager
def monitor_performance(sample_rate_hz=2.0):
    """
    Context manager for performance monitoring
    
    Args:
        sample_rate_hz: Samples per second (default 2Hz for low overhead)
                       1Hz = ultra-low overhead (~1% CPU impact)
                       2Hz = low overhead (~2-3% CPU impact)
                       5Hz = medium overhead (~5-8% CPU impact)
    """
    interval = 1.0 / sample_rate_hz
    monitor = MinimalMonitor(sample_interval=interval)
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()

def benchmark_with_minimal_overhead(pipeline_func, image_paths, version_name, 
                                   sample_rate_hz=2.0):
    """
    Benchmark pipeline with minimal monitoring overhead
    
    Args:
        pipeline_func: Function to benchmark
        image_paths: List of image paths to process
        version_name: Name for logging
        sample_rate_hz: Monitoring frequency (1-5 Hz recommended)
    """
    
    with monitor_performance(sample_rate_hz) as monitor:
        start_time = time.time()
        
        # Run the pipeline
        results = pipeline_func(image_paths)
        
        end_time = time.time()
        processing_time = end_time - start_time
    
    # Get metrics
    metrics = monitor.get_metrics(processing_time, len(image_paths))
    
    # Print results
    print(f"\n=== {version_name.upper()} METRICS ===")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Avg time/frame: {metrics.avg_time_per_frame*1000:.1f}ms")
    print(f"Theoretical FPS: {metrics.theoretical_fps:.1f}")
    print(f"CPU usage: {metrics.avg_cpu_usage:.1f}% (max: {metrics.max_cpu_usage:.1f}%)")
    print(f"CPU memory: {metrics.avg_cpu_memory_used:.0f}MB (max: {metrics.max_cpu_memory_used:.0f}MB)")
    print(f"GPU usage: {metrics.avg_gpu_usage:.1f}% (max: {metrics.max_gpu_usage:.1f}%)")
    print(f"GPU memory util: {metrics.avg_gpu_memory_used:.0f}MB (max: {metrics.max_gpu_memory_used:.0f}MB)")

    return metrics

def print_comparison_table(metrics_dict):
    """Print clean comparison table"""
    if not metrics_dict:
        return
    
    print("\n" + "="*80)
    print("                    PERFORMANCE COMPARISON")
    print("="*80)
    
    versions = list(metrics_dict.keys())
    
    # Header
    print(f"{'Metric':<20} | " + " | ".join(f"{v:>12}" for v in versions))
    print("-" * (20 + len(versions) * 15))
    
    # Get baseline for speedup calculation
    baseline_fps = list(metrics_dict.values())[0].theoretical_fps
    
    # Timing metrics
    print(f"{'Time/Frame (ms)':<20} | " + " | ".join(f"{m.avg_time_per_frame*1000:>12.1f}" for m in metrics_dict.values()))
    print(f"{'Theoretical FPS':<20} | " + " | ".join(f"{m.theoretical_fps:>12.1f}" for m in metrics_dict.values()))
    print(f"{'Speedup vs V1':<20} | " + " | ".join(f"{m.theoretical_fps/baseline_fps:>11.1f}x" for m in metrics_dict.values()))
    
    print()
    
    # CPU metrics
    print(f"{'CPU Usage (%)':<20} | " + " | ".join(f"{m.avg_cpu_usage:>12.1f}" for m in metrics_dict.values()))
    print(f"{'CPU Memory (MB)':<20} | " + " | ".join(f"{m.avg_cpu_memory_used:>12.0f}" for m in metrics_dict.values()))
    
    print()
    
    # GPU metrics
    print(f"{'GPU Usage (%)':<20} | " + " | ".join(f"{m.avg_gpu_usage:>12.1f}" for m in metrics_dict.values()))
    print(f"{'GPU Memory (MB)':<20} | " + " | ".join(f"{m.avg_gpu_memory_used:>12.0f}" for m in metrics_dict.values()))
    print(f"{'GPU Memory (%)':<20} | " + " | ".join(
        f"{((m.avg_gpu_memory_used / m.gpu_memory_total) * 100.0 if m.gpu_memory_total else 0.0):>12.1f}"
        for m in metrics_dict.values()
    ))    
    print("="*80)

# Example usage for your benchmarker
def updated_benchmarker_main():
    """Updated main function for benchmarker.py"""
    
    # Configuration
    baseline_sample_rate = 1.0  # 1Hz for minimal overhead on slow baseline
    optimized_sample_rate = 2.0  # 2Hz for better detail on fast versions
    
    pipeline_configs = [
        ("V1_Naive", lambda imgs: pipeline_v1_naive(imgs, engine_path), baseline_sample_rate),
        ("V2_PreAlloc", lambda imgs: pipeline_v2_preallocated(imgs, engine_path), optimized_sample_rate),
        ("V3_Async", lambda imgs: pipeline_v3_async(imgs, engine_path), optimized_sample_rate),
        ("V4_Kernels", lambda imgs: pipeline_v4_kernels(imgs, engine_path), optimized_sample_rate),
        ("V5_SharedMem", lambda imgs: pipeline_v5_shared_memory(imgs, engine_path), optimized_sample_rate),
        ("V6_Optimized", lambda imgs: pipeline_v6_fully_optimized(imgs, engine_path), optimized_sample_rate)
    ]
    
    all_metrics = {}
    
    for version_name, pipeline_func, sample_rate in pipeline_configs:
        print(f"\n{'='*20} BENCHMARKING {version_name} {'='*20}")
        
        try:
            metrics = benchmark_with_minimal_overhead(
                pipeline_func=pipeline_func,
                image_paths=image_paths,
                version_name=version_name,
                sample_rate_hz=sample_rate
            )
            all_metrics[version_name] = metrics
            
        except Exception as e:
            print(f"Error benchmarking {version_name}: {e}")
            continue
    
    # Print comparison
    if all_metrics:
        print_comparison_table(all_metrics)
        
        # Save results
        save_results_to_file(all_metrics, "benchmark_results.txt")

def save_results_to_file(metrics_dict, filename):
    """Save results to text file"""
    with open(filename, 'w') as f:
        f.write("Minimal Overhead Benchmark Results\n")
        f.write("="*50 + "\n\n")
        
        for version, metrics in metrics_dict.items():
            f.write(f"{version}:\n")
            f.write(f"  Time/frame: {metrics.avg_time_per_frame*1000:.2f}ms\n")
            f.write(f"  FPS: {metrics.theoretical_fps:.2f}\n")
            f.write(f"  CPU: {metrics.avg_cpu_usage:.1f}% ({metrics.avg_cpu_memory_used:.0f}MB)\n")
            f.write(f"  GPU: {metrics.avg_gpu_usage:.1f}% ({metrics.avg_gpu_memory_used:.0f}MB)\n")
            f.write(f"  GPU util: {metrics.avg_gpu_memory_used/metrics.gpu_memory_total*100:.1f}%\n\n")
    
    print(f"Results saved to: {filename}")