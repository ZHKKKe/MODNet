import os

import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.modnet import MODNet


torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
modnet.eval()


def offline_matting(video_path, save_path, fps=30):
    # video capture
    vc = cv2.VideoCapture(video_path)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    if not rval:
        print('Read video {} failed.'.format(video_path))
        exit()

    num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    h, w = frame.shape[:2]

    # video writer
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    print('Start matting...')
    with tqdm(range(int(num_frame)))as t:
        for c in t:
            frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_np = cv2.resize(frame_np, (672, 512), cv2.INTER_AREA)
            # frame_np = frame_np[:, 120:792, :]

            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if GPU:
                frame_tensor = frame_tensor.cuda()

            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)

            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
            fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            fg_np = cv2.cvtColor(fg_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            fg_np = cv2.resize(fg_np, (w, h))

            video_writer.write(fg_np)
            rval, frame = vc.read()
            c += 1

    video_writer.release()
    print('Save video to {}'.format(args.save_path))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='./sample/video.mp4')
    parser.add_argument('--save_path', type=str, default='./sample/matte.mp4')
    parser.add_argument('--fps', type=int, default=30)

    args = parser.parse_args()

    if not args.save_path.endswith('avi'):
        args.save_path = os.path.splitext(args.save_path)[0] + '.avi'

    offline_matting(args.video_path, args.save_path, args.fps)


