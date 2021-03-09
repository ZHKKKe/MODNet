import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from . import modnet_torchscript

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    parser.add_argument('--out-dir', type=str, required=True, help='path for saving the TorchScript model')
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.ckpt_path):
        print('Cannot find checkpoint path: {0}'.format(args.ckpt_path))
        exit()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=True)
    # modnet = nn.DataParallel(modnet).cuda()
    modnet = modnet.cuda()
    ckpt = torch.load(args.ckpt)

    # if use more than one GPU
    if 'module.' in ckpt.keys():
        ckpt = OrderedDict()
        for k, v in ckpt.items():
            k = k.replace('module.', '')
            ckpt[k] = v

    modnet.load_state_dict(ckpt)
    modnet.eval()

    scripted_model = torch.jit.script(modnet)
    torch.jit.save(scripted_model, os.path.join(args.out_dir,'modnet.pt'))

