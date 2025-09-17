import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet


def main(args):
    # An instance of the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    keyname = 'params_ema'
    model.load_state_dict(torch.load(args.input)[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.eval().half().cuda()

    # An example input (batch size 4 for testing)
    x = torch.rand(4, 3, 360, 640).half().cuda()
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx.export(
            model,
            x,
            args.output,
            opset_version=13,
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}, 'output': {0: 'batch', 2: 'height', 3: 'width'}})
    # print(torch_out.shape)


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='RealESRGAN_x2plus.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='realesrgan_x2.onnx', help='Output onnx path')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)
