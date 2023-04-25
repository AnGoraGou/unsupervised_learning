import os
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import glob
import random
# import wandb
import math
import json              
from mpl_toolkits.mplot3d import Axes3D


# ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the file path to the saved model
# model_path = '/workspace/byol_11thApr_1740.pth'
# gnet = torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))   # torch.device('cuda')
# print(model)

img_paths = glob.glob('/workspace/Data/solo_train/*/*.tif')[6]
print(img_paths)

print(f'Number of images: {(img_paths)}')


class ImagesDataset(Dataset):
    def __init__(self, image_size):
        super().__init__()
        # self.folder = folder
        self.paths = img_paths
        self.image_size = image_size


        self.transform = transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomRotation(degrees=(0, 135)),
            transforms.Resize(math.floor(image_size*1.2)),
            transforms.CenterCrop(image_size),
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5)),# change from 9 to 5 and 2 to 0.5
            transforms.ToTensor(),

        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths
        # print(path)
        img = Image.open(path)
        # img = img.convert('RGB')
        img = self.transform(img)
        # img_name = str(path).split('/')[-1]

        return img

# main


if __name__ == '__main__':


    ############################################################################
    ds_c = ImagesDataset(IMAGE_SIZE)
    img_loader = DataLoader(ds_c, batch_size=1, num_workers=2, shuffle=False)
    # torch.set_grad_enabled(False)
    count = 0
    for img in img_loader:
      # print(img.shape)
      # create a tensor o

      # reshape the tensor to (1024, 1024, 3)
      img = img.reshape(1024, 1024, 3)
      ################################################  print(img.size())  #32 3 512 512
      img_name = '/workspace/Visualize/V3/'+ 'aug_'+str(count)+'.png'
      # img = np.transpose(img[0], (1, 2, 0))
      # plt.imshow(img_rgb)
      plt.imshow(img)
      plt.savefig(img_name)
      count+=1

