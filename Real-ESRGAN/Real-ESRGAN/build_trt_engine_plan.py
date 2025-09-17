import tensorrt as trt

########################################################
## Convert ONNX -> TRT with 4 Optimization Profiles ####
########################################################

onnx_path = "real_esrgan_x2_fp16.onnx"
engine_path = "real_esrgan_fp16.plan"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, logger)

with open(onnx_path, "rb") as f:
    assert parser.parse(f.read()), "Failed to parse ONNX"

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

# VRAM capacity
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 29)

# Add 4 optimization profiles — one per tile/context
for _ in range(4):
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min=(1, 3, 180, 320), opt=(1, 3, 360, 640), max=(1, 3, 540, 960))
    config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
with open(engine_path, "wb") as f:
    f.write(serialized_engine)

print("✅ TensorRT engine built with 4 optimization profiles")