
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.modnet import MODNet

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt-path', type=str, default="./pretrained/modnet_webcam_portrait_matting.ckpt", help='path of pre-trained MODNet')
parser.add_argument('--cpu', action='store_true', default=False, help="use cpu inferece")
args = parser.parse_args()


torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
pretrained_ckpt = args.ckpt_path
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)
modnet.load_state_dict(torch.load(pretrained_ckpt, map_location='cpu'))
if not args.cpu:
    modnet.cuda()
modnet.eval()

print('Init WebCam...')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print('Start matting...')
while(True):
    _, frame_np = cap.read()
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    frame_np = cv2.flip(frame_np, 1)

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :] if args.cpu else frame_tensor[None, :, :, :].cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, inference=True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
    view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Exit...')
