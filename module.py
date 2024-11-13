# custom dataset class
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
class BrainMriDataset(Dataset):
    def __init__(self, df:pd.DataFrame, transforms):
        # df contains the paths to all files
        self.df = df
        # transforms is the set of data augmentation operations we use
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 2], 0)

        augmented = self.transforms(image=image,
                                    mask=mask)

        image = augmented['image'] # Dimension (3, 255, 255)
        mask = augmented['mask']   # Dimension (255, 255)

        # We notice that the image has one more dimension (3 color channels), so we have to one one "artificial" dimension to the mask to match it
        mask = np.expand_dims(mask, axis=0) # Dimension (1, 255, 255)

        return image, mask

# UNET model
import torch
import torch.nn as nn
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        # batch normalization
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True))
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = 64
        # Define convolutional layers
        # These are used in the "down" path of the U-Net,
        # where the image is successively downsampled
        self.conv_down1 = double_conv(3, base)
        self.conv_down2 = double_conv(base, base*2)
        self.conv_down3 = double_conv(base*2, base*4)
        self.conv_down4 = double_conv(base*4, base*8)

        # Define max pooling layer for downsampling
        self.maxpool = nn.MaxPool2d(2)

        # Define upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Define convolutional layers
        # These are used in the "up" path of the U-Net,
        # where the image is successively upsampled
        self.conv_up3 = double_conv(base*4 + base*8, base*4)
        self.conv_up2 = double_conv(base*2 + base*4, base*2)
        self.conv_up1 = double_conv(base*2 + base, base)

        # Define final convolution to output correct number of classes
        # 1 because there are only two classes (tumor or not tumor)
        self.last_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        # Forward pass through the network

        # Down path
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        x = self.conv_down4(x)

        # Up path
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)

        # Final output
        out = self.last_conv(x)
        out = torch.sigmoid(out)

        return out

# Loss functions
def dice_coef_loss(inputs:torch.Tensor, target:torch.Tensor):
    inputs = inputs.flatten(1)
    target = target.flatten(1)
    # make target binary
    
    intersection = 2.0 * ((target * inputs).sum(dim=1))
    union = target.sum(dim=1) + inputs.sum(dim=1)
    dice = intersection / (union+1e-7)
    mean_dice_over_batch = dice.mean()
    return 1 - mean_dice_over_batch
def dice_coef_metric(inputs:torch.Tensor, target:torch.Tensor):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union
def bce_dice_loss(inputs:torch.Tensor, target:torch.Tensor):
    
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore

# Training method


def compute_iou(model, loader, threshold=0.3,device:torch.device = torch.device('cpu')):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
    #model.eval()
    valloss = 0

    with torch.no_grad():
        data:torch.Tensor
        target:torch.Tensor
        for i_step, (data, target) in enumerate(loader):

            data = data.to(device)
            target = target.to(device)
            outputs:torch.Tensor = model(data)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

        #print("Threshold:  " + str(threshold) + "  Validation DICE score:", valloss / i_step)
    return valloss / i_step
