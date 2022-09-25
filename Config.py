import os
IMAGES_DIRECTORY = os.path.join("../Learn2Recover/kvasir", "images")
MASKS_DIRECTORY = os.path.join("../Learn2Recover/kvasir", "masks")
BATCH_SIZE=8

import albumentations as A
import albumentations.augmentations.functional as F

train_transform = A.Compose(
     [
         A.HorizontalFlip(p=0.5),
         A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                           shift_limit=0.1, p=1, border_mode=0),
         A.PadIfNeeded(min_height=256, min_width=256,
                      always_apply=True, border_mode=0),
        A.Resize(height=256, width=256, always_apply=True),
         A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
         A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
         A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
         A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        
    ]
)
val_transform = A.Compose(
     [
        A.PadIfNeeded(min_height=256, min_width=256,
                      always_apply=True, border_mode=0),
        A.Resize(height=256, width=256, always_apply=True),
        
    ]
)
test_transform = A.Compose(
     [
        A.PadIfNeeded(min_height=256, min_width=256,
                      always_apply=True, border_mode=0),
        A.Resize(height=256, width=256, always_apply=True),
        
    ]
)


lr=1e-5