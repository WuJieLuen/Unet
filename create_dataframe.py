import sys
import os
import glob
import random
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# Path to all data
DATA_PATH = "./lgg-mri-segmentation/kaggle_3m/"
# File path line length images for later sorting
BASE_LEN = 89 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_ <-!!!43.tif)
END_IMG_LEN = 4 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->.tif)
END_MASK_LEN = 9 # (/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->_mask.tif)
# img size
IMG_SIZE = 512
from pathlib import Path
# Initialize a list to store directory and path info
data_map = []
# Get all subdirectories
sub_dirs = [x for x in Path(DATA_PATH).iterdir() if x.is_dir()]
# Iterate over subdirectories
for sub_dir in sub_dirs:
    # Get all files in subdirectory
    files = [x for x in sub_dir.iterdir() if x.is_file()]

    # Extend data_map with dirname and path info
    data_map.extend([[str(sub_dir.name), str(file)] for file in files])
# Create DataFrame
df = pd.DataFrame(data_map, columns=["dirname", "path"])
# print(df.head())
# print(df['path'].str.contains("mask").value_counts())

# We create 2 dataframes that contain the paths to the images and masks respectively
# Masks/Not masks
df_imgs = df[~df['path'].str.contains("mask")]
df_masks = df[df['path'].str.contains("mask")]

# Data sorting
imgs = sorted(df_imgs["path"].values, key=lambda x : int(x[BASE_LEN:-END_IMG_LEN]) if x[BASE_LEN:-END_IMG_LEN].isdigit() else 0)
masks = sorted(df_masks["path"].values, key=lambda x : int(x[BASE_LEN:-END_MASK_LEN]) if x[BASE_LEN:-END_MASK_LEN].isdigit() else 0)
# Sanity check
# print(len(imgs), len(masks))
# print(df_imgs)
#[9] Final dataframe
df = pd.DataFrame({"patient": df_imgs.dirname.values,
                       "image_path": imgs,
                   "mask_path": masks})

#[11] Adding A/B column for diagnosis
def positiv_negativ_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0 : return 1
    else: return 0
df["diagnosis"] = df["mask_path"].apply(lambda m: positiv_negativ_diagnosis(m))
print(df)
df.to_csv("dataframe.csv")
print("DataFrame have saved as dataframe.csv")

input("Press Enter to End...")