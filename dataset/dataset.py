from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import random
import torchvision.transforms as transforms
from utils.utils import get_patch_coordinates,crop_image
import torch
#Binarize mask
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 0.0] = 0.0
    mask[(mask > 0.0) ] = 1.0
    return mask


class Psuedo(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        
    def __len__(self): 
        return len(self.images_filenames)


    def __getitem__(self, idx):
        #Get a record of filename
        image_filename = self.images_filenames[idx]

        #Read Image and respective mask as numpy arrays
        image=np.array(Image.open(os.path.join(self.images_directory, image_filename)).convert("RGB"),dtype=np.uint8)
        mask=np.array(Image.open( os.path.join(self.masks_directory, image_filename)).convert("L"),dtype=np.uint8)

        #Binarizing mask
        mask = preprocess_mask(mask)
        
        #Apply transforms on image
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        image=transforms.ToTensor()(image)
        mask=transforms.ToTensor()(mask)
        

        cls=random.choice([0,1])

        if cls==1:
            return image/255.,torch.tensor(1)
        
        else:
            # Change mask shape from (256,256) to (1,256,256)
            # mask=mask.unsqueeze(0)

            #get a Bounding box for the Polyp so that we avoid including it in our crop
            x = get_patch_coordinates(mask)

            #Crop a region that does not contain a region of interest
            img=crop_image(x,image)

            #return the image and label
            return img/255.,torch.tensor(0)

