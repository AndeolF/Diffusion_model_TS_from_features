import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc
import psutil
import os

dataset_name = "dataset_50_run_05s_downsample"
# dataset_name = "dataset_05s_windows"


def give_info(tab_info):
    xmax = max(tab_info)
    xmin = min(tab_info)
    print()
    print(f"Max : {xmax}")
    print(f"Min : {xmin}")
    plt.hist(tab_info, bins=100)
    plt.xlim(xmin, xmax)
    plt.title("Distribution des moyennes")
    plt.show()


def compute_rowwise_std_batched(data, batch_size=10000):
    stds = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i : i + batch_size]
        std_batch = np.std(batch, axis=1)
        stds.append(std_batch)
    return np.concatenate(stds)


def filter_ts_by_mask_batched(ts_data, valid_mask, batch_size=10000):
    filtered_chunks = []
    for i in range(0, ts_data.shape[0], batch_size):
        batch_data = ts_data[i : i + batch_size]
        batch_mask = valid_mask[i : i + batch_size]
        filtered_chunk = batch_data[batch_mask]
        filtered_chunks.append(filtered_chunk)
    return np.vstack(filtered_chunks)


def filter_data(dataset_name, type="train", verbose=False):
    ts_train = np.load("Data/" + dataset_name + "/" + type + "_time_series.npy")
    features_train = np.load("Data/" + dataset_name + "/" + type + "_features.npy")

    with open("Data/" + dataset_name + "/ts_scaler.pkl", "rb") as file:
        ts_scaler = pickle.load(file)

    ts_train_std = ts_scaler.transform(ts_train)

    del ts_train
    gc.collect()

    means = ts_train_std.mean(axis=1)
    print(means.shape)

    stds = compute_rowwise_std_batched(data=ts_train_std)
    print(stds.shape)

    if verbose:
        print("\nshape initial")
        print(ts_train_std.shape)
        give_info(means)
        give_info(stds)

    mean_z = (means - np.mean(means)) / np.std(means)
    std_z = (stds - 1) / np.std(stds)
    valid_mask = (np.abs(mean_z) < 3) & (np.abs(std_z) < 3)

    TS_filtered = ts_train[valid_mask]

    features_filtered = features_train[valid_mask]

    means_filtered = TS_filtered.mean(axis=1)
    stds_filtered = compute_rowwise_std_batched(data=TS_filtered)

    if verbose:
        print("\nshape des filtré")
        print(TS_filtered.shape)

        give_info(means_filtered)
        give_info(stds_filtered)

    return TS_filtered, features_filtered


if True:
    tab_type_to_filter = [
        "train",
        "test",
        "val",
    ]
    for type_to_filter in tab_type_to_filter:
        TS_filtered, features_filtered = filter_data(
            dataset_name, type_to_filter, verbose=False
        )

        dataset_clean_name = "Data/" + dataset_name + "_clean"
        os.makedirs(dataset_clean_name, exist_ok=True)

        np.save(
            dataset_clean_name + "/" + type_to_filter + "_time_series.npy",
            TS_filtered,
        )
        np.save(
            dataset_clean_name + "/" + type_to_filter + "_features.npy",
            features_filtered,
        )
