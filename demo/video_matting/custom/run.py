import os
import cv2
import argparse
import numpy as np
from PIL import Image
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


def matting(video, result, alpha_matte=False, fps=30):
    # video capture
    vc = cv2.VideoCapture(video)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    if not rval:
        print('Failed to read the video: {0}'.format(video))
        exit()

    num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    h, w = frame.shape[:2]
    if w >= h:
        rh = 512
        rw = int(w / h * 512)
    else:
        rw = 512
        rh = int(h / w * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(result, fourcc, fps, (w, h))

    print('Start matting...')
    with tqdm(range(int(num_frame)))as t:
        for c in t:
            frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_np = cv2.resize(frame_np, (rw, rh), cv2.INTER_AREA)

            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if GPU:
                frame_tensor = frame_tensor.cuda()

            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)

            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
            if alpha_matte:
                view_np = matte_np * np.full(frame_np.shape, 255.0)
            else:
                view_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            view_np = cv2.cvtColor(view_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            view_np = cv2.resize(view_np, (w, h))
            video_writer.write(view_np)

            rval, frame = vc.read()
            c += 1

    video_writer.release()
    print('Save the result video to {0}'.format(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='input video file')
    parser.add_argument('--result-type', type=str, default='fg', choices=['fg', 'matte'], 
                        help='matte - save the alpha matte; fg - save the foreground')
    parser.add_argument('--fps', type=int, default=30, help='fps of the result video')

    print('Get CMD Arguments...')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print('Cannot find the input video: {0}'.format(args.video))
        exit()

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

    result = os.path.splitext(args.video)[0] + '_{0}.mp4'.format(args.result_type)
    alpha_matte = True if args.result_type == 'matte' else False
    matting(args.video, result, alpha_matte, args.fps)
