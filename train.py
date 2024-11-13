# Meta parameters
#---------------------------------------------------------------------------------#
model_name = 'Vanila_UNet'
PATCH_SIZE = 128 # 128 or 256
batch = 32
learning_rate = 1e-4
num_ep = 20
patience = 10
checkpoints_period = -1 # -1 for no checkpoint
is_use_pretrained = False
#---------------------------------------------------------------------------------#

# Check torch is available
import torch
import torch.nn.functional as F
print('Check Envoriment...')
device = torch.device('cuda:0')
if device:
    print(f'{device} is available.')
    print(f'PyTorch version: {torch.__version__}')
    print('Envoriment is ready.\n')

# create dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from module import BrainMriDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
df: pd.DataFrame = pd.read_csv("dataframe.csv", index_col=0)
train_df: pd.DataFrame
val_df: pd.DataFrame
test_df: pd.DataFrame
train_df, val_df = train_test_split(df, stratify=df.diagnosis, test_size=0.1)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print('Prepare train, val, test data...')
train_df, test_df = train_test_split(train_df, stratify=train_df.diagnosis, test_size=0.15)
train_df = train_df.reset_index(drop=True)
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
train_dataset = BrainMriDataset(train_df, transforms=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_dataset = BrainMriDataset(val_df, transforms=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_dataset = BrainMriDataset(test_df, transforms=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
print(f"Train: {train_df.shape} \nVal: {val_df.shape} \nTest: {test_df.shape}\n")

# Creat new UNet Model and optimizer
from module import UNet
if is_use_pretrained:
    print('Use pretrained model...')
    unet = torch.load('Vanila_UNet_model.pt')
    unet_optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    unet_optimizer.load_state_dict(torch.load('Vanila_UNet_optimizer.pt'))
    # use new learning rate
    for param_group in unet_optimizer.param_groups:
        param_group['lr'] = learning_rate
else:   
    print('Create new model and optimizer...')
    unet = UNet().to(device)
    unet_optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    pass


# start trainning
import numpy as np
from tqdm import tqdm
from module import bce_dice_loss
loss_history = []
train_history = []
val_history = []
best_val_loss = 100

unet.train()  # Enter train mode
for epoch in range(num_ep):
    # We store the training loss and dice scores
    losses = []
    train_iou = []

    # Add tqdm to the loop (to visualize progress)
    data:torch.Tensor
    target:torch.Tensor
    for i_step, (data, target) in enumerate(tqdm(train_dataloader, desc=f"Training epoch {epoch+1}/{num_ep}")):
        data = data.to(device)
        target = target.to(device)
        target = (target > 128).float()
        outputs:torch.Tensor = unet(data)
        # out_cut = np.copy(outputs.data.cpu().numpy())

        # If the score is less than a threshold (0.5), the prediction is 0, otherwise its 1
        # out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        # out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
        # train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())

        loss:torch.Tensor = bce_dice_loss(outputs, target)
        losses.append(loss.item())
        # train_iou.append(train_dice)

        # Reset the gradients
        unet_optimizer.zero_grad()
        # Perform backpropagation to compute gradients
        loss.backward()
        # Update the parameters with the computed gradients
        unet_optimizer.step()

    #val_mean_iou = compute_iou(model, val_loader,device = device)
    #loss_history.append(np.array(losses).mean())
    #train_history.append(np.array(train_iou).mean())
    #val_history.append(val_mean_iou)

    print("Epoch [%d]" % (epoch))
    # print("Mean loss on train:", np.array(losses).mean(),
    #       "\nMean DICE on train:", np.array(train_iou).mean(),
    #       "\nMean DICE on validation:", val_mean_iou)
    epoch_mean_loss = np.array(losses).mean()
    print("Mean loss on train:", np.array(losses).mean())

    # change learning rate according to loss
    # if abs(epoch_mean_loss - best_val_loss) < 1:
    #     for param_group in optimizer.param_groups:
    #         print(f'Learning rate changed to {param_group["lr"]} from {param_group["lr"]*0.8}')
    #         param_group['lr'] = param_group['lr']*0.8
    #     pass

    # Save the model if the validation loss is the best we've seen so far
    if epoch_mean_loss < best_val_loss:
        best_val_loss = epoch_mean_loss
        torch.save(unet, f'{model_name}_model.pt')
        torch.save(unet_optimizer.state_dict(), f'{model_name}_optimizer.pt')
        pass
    pass
    # Save as checkpoint by period
    if checkpoints_period >0: 
        if epoch % checkpoints_period == 0:
            torch.save(unet, f'{model_name}_model_{epoch}.pt')
            pass
    # Early stopping if the validation loss does not improve for 'patience' epochs
    early_stop_counter = 0
    if epoch_mean_loss > best_val_loss:
        early_stop_counter += 1
    else:
        early_stop_counter = 0
        pass
    if early_stop_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break
# plot loss history
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title('Loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# plot train history


print('Training is done.')
print('Model saved as unet_model.pt.\n')

#input("Press Enter to Continue...")