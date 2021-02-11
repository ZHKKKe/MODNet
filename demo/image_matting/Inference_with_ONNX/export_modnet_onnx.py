"""
Export onnx model

Arguments:
    --ckpt-path --> Path of last checkpoint to load
    --output-path --> path of onnx model to be saved

example:
python export_modnet_onnx.py \
    --ckpt-path=modnet_photographic_portrait_matting.ckpt \
    --output-path=modnet.onnx

output:
ONNX model with dynamic input shape: (batch_size, 3, height, width) &
                        output shape: (batch_size, 1, height, width)                  
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.models.onnx_modnet import MODNet



if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of pre-trained MODNet')
    parser.add_argument('--output-path', type=str, required=True, help='path of output onnx model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    # define model & load checkpoint
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    state_dict = torch.load(args.ckpt_path)
    modnet.load_state_dict(state_dict)
    modnet.eval()

    # prepare dummy_input
    batch_size = 1
    height = 512
    width = 512
    dummy_input = Variable(torch.randn(batch_size, 3, height, width)).cuda()

    # export to onnx model
    torch.onnx.export(modnet.module, dummy_input, args.output_path, export_params = True, opset_version=11,
                    input_names = ['input'], output_names = ['output'], 
                    dynamic_axes = {'input': {0:'batch_size', 2:'height', 3:'width'},
                                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
