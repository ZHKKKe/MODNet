import os
import glob
import cv2
from pathlib import Path
from typing import Tuple, Union

import torch
import torchvision
from torch.utils.data import Dataset

from src.train.trimap import makeTrimap

class SegDataset(Dataset):
    """A custom Dataset(torch.utils.data) implement three functions: __init__, __len__, and __getitem__.
    Datasets are created from PTFDataModule.
    """

    def __init__(
        self,
        frame_dir: Union[str, Path],
        mask_dir: Union[str, Path]
    ) -> None:

        self.frame_dir = Path(frame_dir)
        self.mask_dir = Path(mask_dir)
        self.image_names = glob.glob(f"{self.frame_dir}/*.jpg") 
        self.mask_names = [os.path.join(self.mask_dir,(x.split('/')[-1])[:-4]+".png") for x in self.image_names] 
        #print(self.mask_names)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((512,512)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        
        ])
        self.transform2 = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((512,512)),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_pth = self.image_names[index]
        mask_pth = self.mask_names[index]

        frame = cv2.imread(frame_pth)
        frame = self.transform(frame)

        mask = cv2.imread(mask_pth,cv2.IMREAD_GRAYSCALE)
        trimap = torch.from_numpy(makeTrimap(mask)).float()
        trimap = torch.unsqueeze(trimap,0)
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0).float()

        mask = self.transform2(mask)
        trimap = self.transform2(trimap)
        
        return frame, trimap, mask

    def __len__(self):
        return len(self.image_names)