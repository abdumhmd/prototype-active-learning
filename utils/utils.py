#
# Created on Sun Aug 21 2022
#
# The MIT License (MIT)
# Copyright (c) 2022 Abdurahman A. Mohammed
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import copy
import random

import albumentations as A
import albumentations.augmentations.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from albumentations.pytorch import ToTensorV2


def visualize_augmentations(dataset, idx=0, samples=5):
    """Visualizes sample augmentations

    Args:
        dataset (torchvision dataset): Dataset to be visualized
        idx (int, optional): Index of image to be augmented. Defaults to 0.
        samples (int, optional): How many variations to visualize. Defaults to 5.
    """
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 15))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()



random.seed(42)

def get_patch_coordinates(grayscale_cam):
    """
    Get a rectangle coordinate of the grayscale
    """
    grayscale_cam=grayscale_cam.permute(1,2,0)
    min_row, min_col, max_row, max_col = 0, 0, 0, 0

    for row in range(grayscale_cam.shape[0]):

        if 1 in grayscale_cam[row,:, :]:
            min_row = row
            break

    for row in range(grayscale_cam.shape[0]-1,-1, -1):
        if 1 in grayscale_cam[row,:, :]:
            max_row = row
            break

    for col in range(grayscale_cam.shape[1]):
        if 1 in grayscale_cam[:,col,:]:
            min_col = col
            break

    for col in range(grayscale_cam.shape[1]-1, -1, -1):
        if 1 in grayscale_cam[:,col,:]:
            max_col = col
            break

    return min_row, min_col, max_row, max_col

def crop_image(coordinates,image):
    """Crop a region of image that does not contain the region of interest

    Args:
        coordinates (tuple): Contains min_row,min_col,max_row,max_col
        image (nd.array): The image to be cropped
    Return:
        img (nd.array): The cropped image

    """

    min_row, min_col, max_row, max_col = coordinates


    if(min_row>0 and min_col>0):
        img=image[:,0:min_row,0:min_col]
        
        img=transforms.ToPILImage()(img).resize((256,256))
        img=transforms.ToTensor()(img)
    else:
        
        img=image[:,max_row:len(image[1]),max_col:len(image[1])]
        
        img=transforms.ToPILImage()(img).resize((256,256))
        img=transforms.ToTensor()(img)

    return img


