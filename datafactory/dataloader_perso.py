from torch.utils.data import Dataset, DataLoader  # type: ignore
from datafactory.dataset import T2SDataset
import torch  # type: ignore
import numpy as np
from sklearn.preprocessing import StandardScaler
from .dataset_perso import DatasetFeaturesSeries
import pickle
import os


def load_scaled_dataloaders(
    dataset_path, batch_size=32, scale_features=True, scale_series=True
):
    # 1. Load .npy arrays
    train_feat = np.load(f"{dataset_path}/train_features.npy").astype(np.float32)
    train_ts = np.load(f"{dataset_path}/train_time_series.npy").astype(np.float32)
    val_feat = np.load(f"{dataset_path}/val_features.npy").astype(np.float32)
    val_ts = np.load(f"{dataset_path}/val_time_series.npy").astype(np.float32)
    test_feat = np.load(f"{dataset_path}/test_features.npy").astype(np.float32)
    test_ts = np.load(f"{dataset_path}/test_time_series.npy").astype(np.float32)

    # 2. Standardize
    if scale_features:
        feature_scaler = StandardScaler()
        train_feat = feature_scaler.fit_transform(train_feat)
        val_feat = feature_scaler.transform(val_feat)
        test_feat = feature_scaler.transform(test_feat)
    else:
        feature_scaler = None

    if scale_series:
        ts_scaler = StandardScaler()
        train_ts = ts_scaler.fit_transform(train_ts)
        val_ts = ts_scaler.transform(val_ts)
        test_ts = ts_scaler.transform(test_ts)
        #
        # Reshape to 2D if needed
        # train_ts_2d = train_ts.reshape(-1, 1)
        # ts_scaler.fit(train_ts_2d)
        # train_ts = ts_scaler.transform(train_ts.reshape(-1, 1)).reshape(train_ts.shape)
        # val_ts = ts_scaler.transform(val_ts.reshape(-1, 1)).reshape(val_ts.shape)
        # test_ts = ts_scaler.transform(test_ts.reshape(-1, 1)).reshape(test_ts.shape)
    else:
        ts_scaler = None

    path_to_features_scaler = dataset_path + "/features_scaler.pkl"
    path_to_ts_scaler = dataset_path + "/ts_scaler.pkl"

    if not os.path.isfile(path_to_features_scaler):
        with open(path_to_features_scaler, "wb") as file:
            pickle.dump(feature_scaler, file)

    if not os.path.isfile(path_to_ts_scaler):
        with open(path_to_ts_scaler, "wb") as file:
            pickle.dump(ts_scaler, file)

    # 3. Wrap into PyTorch Dataset
    train_set = DatasetFeaturesSeries(train_feat, train_ts)
    val_set = DatasetFeaturesSeries(val_feat, val_ts)
    test_set = DatasetFeaturesSeries(test_feat, test_ts)

    # 4. Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, feature_scaler, ts_scaler
