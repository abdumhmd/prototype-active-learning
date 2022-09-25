import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import pandas as pd
import numpy as np
import os
import random
import cv2
import Config
import albumentations as A
import albumentations.augmentations.functional as F
from utils.utils import get_patch_coordinates,crop_image
from dataset.dataset import Psuedo

from model.PolypNet import PolypNet

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset,DataLoader
from PIL import Image





class PolypDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    def setup(self,stage=None):
        # Get the image file names and make sure they are valid images.
        images_filenames = list(sorted(os.listdir(Config.IMAGES_DIRECTORY)))
        correct_images_filenames = [i for i in images_filenames if cv2.imread(
            os.path.join(Config.IMAGES_DIRECTORY, i)) is not None]
        # Shuffle the images list before split. Using a random seed.
        random.seed(42)
        random.shuffle(correct_images_filenames)
        #Take a smaller portion of the original dataset
        sub=int(len(correct_images_filenames)*.5)
        correct_images_filenames=correct_images_filenames[:sub]
        # Perform train valid test split of 800:150:50
        train_size = int(len(correct_images_filenames)*.8)
        test_size = int(len(correct_images_filenames)*.1)
        train_images_filenames = correct_images_filenames[:train_size]
        val_images_filenames = correct_images_filenames[train_size:-test_size]
        test_images_filenames = images_filenames[-test_size:]
        # print(len(train_images_filenames), len(val_images_filenames), len(test_images_filenames))

        self.train_data = Psuedo(train_images_filenames, Config.IMAGES_DIRECTORY,
                        Config.MASKS_DIRECTORY, transform=Config.train_transform)
        self.valid_data = Psuedo(val_images_filenames, Config.IMAGES_DIRECTORY,
                        Config.MASKS_DIRECTORY, transform=Config.val_transform)
        self.test_data = Psuedo(val_images_filenames, Config.IMAGES_DIRECTORY,
                        Config.MASKS_DIRECTORY, transform=Config.test_transform)                        
    def train_dataloader(self):

        return DataLoader(self.train_data,batch_size=Config.BATCH_SIZE,shuffle = False)

    def val_dataloader(self):  

        return DataLoader(self.valid_data,batch_size=Config.BATCH_SIZE,shuffle = False)

    def test_dataloader(self):

        return DataLoader(self.test_data,batch_size=Config.BATCH_SIZE,shuffle = False)


ployp_data=PolypDataModule()

ployp_data.setup()

classifier = PolypNet()
pl.Trainer()
trainer = pl.Trainer(accelerator='auto',callbacks=[TQDMProgressBar(refresh_rate=20)], max_epochs=200,detect_anomaly=True)  # for Colab: set refresh rate to 20 instead of 10 to avoid freezing
trainer.fit(classifier, ployp_data )

