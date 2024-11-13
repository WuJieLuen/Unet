# User Configuration
sample_number = 5   # Number of samples to show (Ramdaomly selected)
type = "tumor"      # "tumor" or "sane"
is_show = True      # True for show, False for not show
is_save = True     # True for save, False for not save

# Script Start
import pandas as pd
import cv2
df = pd.read_csv("dataframe.csv", index_col=0)
sample_tumor_index = df[df["diagnosis"] == 1].sample(sample_number).index
sample_sane_index = df[df["diagnosis"] == 0].sample(sample_number).index
tumor_image_path = df.iloc[sample_tumor_index]["image_path"]
sane_image_path = df.iloc[sample_sane_index]["image_path"]
tumor_mask_path = df.iloc[sample_tumor_index]["mask_path"]
sane_mask_path = df.iloc[sample_sane_index]["mask_path"]

if type == "tumor":
    for ind,(path_tumor,path_tumor_mask) in enumerate(zip(tumor_image_path,tumor_mask_path)):
        file_name = f'{type}_{ind}.jpg'
        img = cv2.imread(path_tumor)
        mask = cv2.imread(path_tumor_mask)
        # concat image and mask
        concat = cv2.hconcat([img, mask])
        if is_show:
            cv2.imshow(file_name, concat)
        if is_save:
            cv2.imwrite(file_name, concat)
            print(f"{file_name} saved.")
            pass
else:
    for ind,(path_sane,path_sane_mask) in enumerate(zip(sane_image_path,sane_mask_path)):
        file_name = f'{type}_{ind}.jpg'
        img = cv2.imread(path_sane)
        mask = cv2.imread(path_sane_mask)
        # concat image and mask
        concat = cv2.hconcat([img, mask])
        if is_show:
            cv2.imshow(file_name, concat)
        if is_save:
            cv2.imwrite(file_name, concat)
            print(f"{file_name} saved.")
            pass
        pass
    pass
pass

if is_show and not is_save:
    print("Press Enter to End...")
    cv2.waitKey(0)
    pass
if is_save and not is_show:
    input("Press Enter to End...")
    pass
if is_show and is_save:
    print("Press Enter to End...")
    cv2.waitKey(0)
    pass
if not is_show and not is_save:
    print("Nothing to do.")
    pass