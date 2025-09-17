import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet
import tensorrt as trt

# building the model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4) # 4 -> 2

ckpt = torch.load('experiments/pretrained_models/RealESRGAN_x4plus.pth', map_location='cpu')

if 'params_ema' in ckpt:
    state_dict = ckpt['params_ema']
elif 'params' in ckpt:
    state_dict = ckpt['params']
else:
    state_dict = ckpt

print("conv_first.weight shape:", state_dict['conv_first.weight'].shape)

# print("ELLO")
# print(model)
# print(ckpt.keys())
# Export to ONNX
# use exponential moving avg weights if possible
if 'params_ema' in ckpt:
    model.load_state_dict(ckpt['params_ema'], strict=True)
else:
    model.load_state_dict(ckpt, strict=True)


model.eval().half().cuda() # Switch to FP16 and move to GPU

dummy_input = torch.randn(1, 3, 360, 640).half().cuda()

# export the model with dynamic input shape
torch.onnx.export(
    model,
    dummy_input,
    'real_esrgan_x2_fp16.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}, 'output': {0: 'batch', 2: 'height', 3: 'width'}}, # batch size, height and width are dynamic
    opset_version=11
)

print("✅ Export complete: real_esrgan_x2_fp16.onnx")
