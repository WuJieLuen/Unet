import torch
import pandas as pd
from module import BrainMriDataset, UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import matplotlib.pyplot as plt
# prepare dataset
device = torch.device("cuda")
PATCH_SIZE = 128
transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),

    A.Normalize(p=1.0),
    ToTensorV2(),
])
df: pd.DataFrame = pd.read_csv("dataframe.csv", index_col=0)

test_dataset = BrainMriDataset(df, transforms=transforms)
# 
def generate_random_indices(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices[:num_samples]

# load model from pt file
unet = torch.load('Vanila_UNet_model.pt')
unet2 = torch.load('Origin_Vanila_UNet_model.pt')
unet.to(device)
unet2.to(device)
#unet = UNet().to(device)

# get random indices
num_samples = 2
sample_indices = df[df["diagnosis"] == 1].sample(num_samples).index
# Create a subplot with the input images and predictions
fig, axes = plt.subplots(num_samples, 4, figsize=(10, 5))

for i, idx in enumerate(sample_indices):
    image:torch.Tensor
    image, mask = test_dataset[idx]
    mask = mask[0, :, :]
    
    prediction:torch.Tensor = unet(image.unsqueeze(0).to(device))
    prediction = prediction[0, 0, :, :].data.cpu().numpy()

    prediction2:torch.Tensor = unet2(image.unsqueeze(0).to(device))
    prediction2 = prediction2[0, 0, :, :].data.cpu().numpy()

    image = image.permute(1, 2, 0).numpy()
    image = image - image.min()
    image = image / image.max()
    # use threshold to get binary mask
    prediction = (prediction > 0.5).astype(float)
    prediction2 = (prediction2 > 0.5).astype(float)
    # Plot the input image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"Brain MRI")
    
    axes[i, 1].imshow(mask)
    axes[i, 1].set_title(f"Tumor Ground Truth")

    # Plot the prediction
    axes[i, 2].imshow(prediction)
    axes[i, 2].set_title(f"Modify U-Net Prediction")
    axes[i, 3].imshow(prediction2)
    axes[i, 3].set_title(f"Origin U-Net Prediction")

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()