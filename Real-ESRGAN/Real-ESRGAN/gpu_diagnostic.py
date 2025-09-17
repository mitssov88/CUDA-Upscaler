import tensorrt as trt
import torch
import os

print("=== GPU DIAGNOSTIC ===")

# Check PyTorch GPU info
if torch.cuda.is_available():
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
    print(f"PyTorch current device: {torch.cuda.current_device()}")
    print(f"PyTorch device name: {torch.cuda.get_device_name()}")
    print(f"PyTorch device memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print("PyTorch CUDA not available")

print("\n=== TensorRT Info ===")
print(f"TensorRT version: {trt.__version__}")

# Check TensorRT GPU info
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)

# Try to get GPU info through TensorRT
try:
    # Test with a simple network to see what TensorRT can handle
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Add a simple layer to test
    input_tensor = network.add_input("input", trt.DataType.FLOAT, (1, 3, 512, 512))
    identity = network.add_identity(input_tensor)
    network.mark_output(identity.get_output(0))
    
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    
    print("[INFO] Testing TensorRT with 512x512 input...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        print("✅ TensorRT can handle 512x512 input")
    else:
        print("❌ TensorRT failed on 512x512 input")
        
except Exception as e:
    print(f"❌ TensorRT test failed: {e}")

print("\n=== ONNX Model Check ===")
onnx_path = "realesrgan_x4_fp16.onnx"
if os.path.exists(onnx_path):
    size_mb = os.path.getsize(onnx_path) / (1024*1024)
    print(f"ONNX file size: {size_mb:.1f} MB")
    
    # Try to parse ONNX
    try:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, "rb") as f:
            if parser.parse(f.read()):
                print("✅ ONNX parsing successful")
                print(f"Network layers: {network.num_layers}")
                print(f"Network inputs: {network.num_inputs}")
                print(f"Network outputs: {network.num_outputs}")
                
                # Check input shape
                input_tensor = network.get_input(0)
                print(f"Input shape: {input_tensor.shape}")
                
            else:
                print("❌ ONNX parsing failed")
                for i in range(parser.num_errors):
                    print(f"  Error {i}: {parser.get_error(i)}")
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")
else:
    print(f"❌ ONNX file not found: {onnx_path}")

print("\n=== Memory Test ===")
# Test GPU memory allocation
try:
    if torch.cuda.is_available():
        # Try allocating a large tensor
        large_tensor = torch.randn(1, 3, 2048, 2048, device='cuda')
        print(f"✅ Successfully allocated 2048x2048 tensor on GPU")
        del large_tensor
        torch.cuda.empty_cache()
        
        # Try even larger
        huge_tensor = torch.randn(1, 3, 4096, 4096, device='cuda')
        print(f"✅ Successfully allocated 4096x4096 tensor on GPU")
        del huge_tensor
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"❌ GPU memory test failed: {e}") 