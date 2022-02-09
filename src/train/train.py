import os
import argparse
import logging
import logging.handlers

import torch
import torch.nn as nn

import neptune.new as neptune

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter
from src.train.dataset import SegDataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', type=str, required=True, help='path to dataset')
    parser.add_argument('--modelsPath', type=str, required=True, help='path to save trained MODNet models')
    parser.add_argument('--pretrainedPath', type=str, help='path of pre-trained MODNet')
    parser.add_argument('--startEpoch', type=int, default=-1, help='epoch to start with')
    parser.add_argument('--batchCount', type=int, default=16, help='batches count')
    args = parser.parse_args()
    return args

def train(modnet, datasetPath: str, batch_size: int, startEpoch: int, modelsPath: str):
    lr = 0.01       # learn rate
    epochs = 40     # total epochs

    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1, last_epoch=startEpoch)

    dataset = SegDataset(os.path.join(datasetPath, "images"), os.path.join(datasetPath, "masks"))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    project = 'motionlearning/modnet'
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmJhMzJiMC1iY2M0LTQ5NGUtOGI0OS0yYzA4ZmNiNTRiMmEifQ=='
    neptuneRun = neptune.init(project = project,
                            api_token = api_token,
                            source_files=[])

    for epoch in range(0, epochs):
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            semantic_loss, detail_loss, matte_loss, semantic_iou = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte, semantic_scale=1, detail_scale=10, matte_scale=1)
            if idx % 100 == 0:
                logger.info(f'idx: {idx}, semantic_loss: {semantic_loss:.5f}, detail_loss: {detail_loss:.5f}, matte_loss: {matte_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
        logger.info(f'Epoch: {epoch}, semantic_loss: {semantic_loss:.5f}, detail_loss: {detail_loss:.5f}, matte_loss: {matte_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
        
        neptuneRun["training/epoch/semantic_loss"].log(semantic_loss)
        neptuneRun["training/epoch/detail_loss"].log(detail_loss)
        neptuneRun["training/epoch/matte_loss"].log(matte_loss)
        neptuneRun["training/epoch/semantic_iou"].log(semantic_iou)

        modelPath = os.path.join(modelsPath, f"model_epoch{epoch}.ckpt")
        torch.save(modnet.state_dict(), modelPath)
        logger.info(f"model saved to {modelPath}")
        lr_scheduler.step()

    torch.save(modnet.state_dict(), os.path.join(modelsPath, "model.ckpt"))

    neptuneRun.stop()

def tune(modnet, datasetPath: str, batch_size: int, modelsPath: str):
    pass

def twoStepTrain(datasetPath: str, batch_size: int, startEpoch, pretrainedPath, modelsPath):

    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet)

    if pretrainedPath is not None:
        modnet.load_state_dict(
            torch.load(pretrainedPath)
        )

    train(modnet, datasetPath, batch_size, startEpoch, modelsPath)
    tune(modnet, datasetPath, batch_size, modelsPath)

args = parseArgs()

train(args.datasetPath, args.batchCount, args.startEpoch, args.pretraindPath, args.modelsPath)