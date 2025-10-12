import os, math, yaml, random, cv2, numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_device(opt):
    if opt == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return opt

def seed_everything(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.up4 = up(512, 256)
        self.up3 = up(256+256, 128)
        self.up2 = up(128+128, 64)
        self.up1 = up(64+64, 64)
        self.up0 = up(64+64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)

        if in_ch != 3:
            self.layer0[0] = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.pool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        u4 = self.up4(x4)
        u3 = self.up3(torch.cat([u4, x3], dim=1))
        u2 = self.up2(torch.cat([u3, x2], dim=1))
        u1 = self.up1(torch.cat([u2, x1], dim=1))
        u0 = self.up0(torch.cat([u1, x0], dim=1))
        return self.outc(u0)

def bce_dice_loss(logits, mask):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
    probs = torch.sigmoid(logits)
    smooth=1.0
    inter = (probs*mask).sum()
    dice = 1 - (2*inter + smooth)/ (probs.sum() + mask.sum() + smooth)
    return bce + dice

class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, size=512, aug=True):
        self.images = images
        self.masks = masks
        self.tf_train = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
        self.tf_val = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
        self.aug = aug

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        img = cv2.imread(self.images[i], cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        msk = (msk>127).astype('uint8')*255
        img = img[...,None]
        if self.aug:
            out = self.tf_train(image=img, mask=msk)
        else:
            out = self.tf_val(image=img, mask=msk)
        x = out['image'].float()
        y = out['mask'].float().unsqueeze(0)/255.0
        return x, y

def list_pairs(images_dir, masks_dir):
    ims = []
    mks = []
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')):
            im = os.path.join(images_dir, fn)
            mk = os.path.join(masks_dir, fn)
            if os.path.exists(mk):
                ims.append(im); mks.append(mk)
    ims, mks = zip(*sorted(zip(ims, mks)))
    return list(ims), list(mks)
