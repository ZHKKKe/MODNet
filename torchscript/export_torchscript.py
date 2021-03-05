"""
Export TorchScript model of MODNet

Arguments:
    --ckpt-path: path of the checkpoint that will be converted
    --output-path: path for saving the TorchScript model

Example:
    python export_torchscript.py \
        --ckpt-path=modnet_photographic_portrait_matting.ckpt \
        --output-path=modnet_photographic_portrait_matting.torchscript
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modnet_torchscript


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the TorchScript model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print(args.ckpt_path)
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    # create MODNet and load the pre-trained ckpt
    modnet = modnet_torchscript.MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    state_dict = torch.load(args.ckpt_path)
    modnet.load_state_dict(state_dict)
    modnet.eval()

    # export to TorchScript model
    scripted_model = torch.jit.script(modnet.module)
    torch.jit.save(scripted_model, os.path.join(args.output_path))
