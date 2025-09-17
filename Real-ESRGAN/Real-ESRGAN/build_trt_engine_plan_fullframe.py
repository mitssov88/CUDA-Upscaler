import tensorrt as trt

onnx_model_path = "realesrgan_x2.onnx"
engine_output_path = "realesrgan_x2.plan"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

print(f"[INFO] Loading ONNX model from {onnx_model_path}")
with open(onnx_model_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("[ERROR] Failed to parse ONNX model")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3000)   # 24 GiB
# ---- allow only cuBLAS(+Lt) tactics ----
allowed = int(trt.TacticSource.CUBLAS) | int(trt.TacticSource.CUBLAS_LT)
config.set_tactic_sources(allowed) 
profile = builder.create_optimization_profile()

input_name = network.get_input(0).name
profile.set_shape(input_name, min=(1, 3, 360, 640), opt=(4, 3, 720, 1280), max=(4, 3, 720, 1280))
config.add_optimization_profile(profile)

print("[INFO] Building serialized engine (full frame)...")
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("[ERROR] Failed to build engine")

with open(engine_output_path, "wb") as f:
    f.write(serialized_engine)

print(f"✅ Full-frame TensorRT engine saved as: {engine_output_path}")