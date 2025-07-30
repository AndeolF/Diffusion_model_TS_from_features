import torch  # type: ignore
import numpy as np
import math
import random
from datafactory.dataloader_perso import load_scaled_dataloaders
import gc
import psutil

gc.collect()
device = "cuda" if torch.cuda.is_available() else "cpu"
name_dataset = "dataset_50_run_05s_downsample"

train_loader, val_loader, test_loader, feat_scaler, ts_scaler = load_scaled_dataloaders(
    dataset_path="./Data/" + name_dataset,
    batch_size=1024,
    scale_features=True,
    scale_series=True,
)

del val_loader
del test_loader
del feat_scaler
del ts_scaler

pretrained_model = torch.load(
    "checkpoint/dataset_50_run_05s_downsample/final_VAEpretrain_model.pth",
    map_location=torch.device(device),
)
pretrained_model.float().to(device).eval()


sum_x = None
sum_x2 = None
count = 0

print(len(train_loader))

for batch, data in enumerate(train_loader):
    if batch % 500 == 0:
        print(f"Batch {batch}")

    if (batch + 1) != len(train_loader):
        infer_ts, _ = data
        x_1 = infer_ts.float().to(device)
        x_encode, _ = pretrained_model.encoder(x_1)
        x_encode = x_encode.detach().cpu().numpy()  # shape: (B, latent_dim)

        if x_encode.ndim == 1:
            x_encode = np.expand_dims(x_encode, 0)

        if sum_x is None:
            sum_x = np.sum(x_encode, axis=0)
            sum_x2 = np.sum(x_encode**2, axis=0)
        else:
            sum_x += np.sum(x_encode, axis=0)
            sum_x2 += np.sum(x_encode**2, axis=0)

        count += x_encode.shape[0]

del train_loader
gc.collect()

mean = sum_x / count
std_latent_data_train_encode = np.sqrt(sum_x2 / count - mean**2).astype(np.float32)

print(std_latent_data_train_encode.shape)

np.save(
    "Data/" + name_dataset + "/std_latent_data_train_encode.npy",
    std_latent_data_train_encode,
)
