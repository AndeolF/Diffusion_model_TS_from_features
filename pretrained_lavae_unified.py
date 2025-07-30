import argparse
import numpy as np
import os
import random
import torch  # type: ignore
from model.pretrained.vqvae import vqvae
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns  # type: ignore
from datafactory.dataloader import loader_provider
from datafactory.dataloader_perso import load_scaled_dataloaders


def seed_everything(seed_value=11):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed: {seed_value}")


def plot_comparison(real, reconstructed, save_path):
    for i in range(len(real)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(real[i], label="Real")
        axs[0].set_title(f"Real Sample {i}")
        axs[1].plot(reconstructed[i], label="Reconstructed")
        axs[1].set_title(f"Reconstructed Sample {i}")
        axs[0].legend()
        axs[1].legend()
        plt.savefig(f"{save_path}/comparison_{i}.png")
        plt.show()
        plt.close()


def plot_pca_tsne(real_samples, reconstructed_samples, save_path):
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=min(len(real_samples), 30))
    combined_samples = np.vstack((real_samples, reconstructed_samples))
    combined_samples_pca = pca.fit_transform(combined_samples)
    combined_samples_tsne = tsne.fit_transform(combined_samples)
    labels = ["Real"] * len(real_samples) + ["Reconstructed"] * len(
        reconstructed_samples
    )
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(
        x=combined_samples_pca[:, 0],
        y=combined_samples_pca[:, 1],
        hue=labels,
        ax=axs[0],
    )
    axs[0].set_title("PCA")
    sns.scatterplot(
        x=combined_samples_tsne[:, 0],
        y=combined_samples_tsne[:, 1],
        hue=labels,
        ax=axs[1],
    )
    axs[1].set_title("t-SNE")
    plt.legend()
    # plt.savefig(f"{save_path}/pca_tsne.png")
    plt.show()
    plt.close()


def inference(model, test_loader, device, save_dir, num_samples=None, verbose=False):
    model.eval()
    real_samples = []
    reconstructed_samples = []
    z_samples = []
    print(f"number batch in test : {len(test_loader)}")
    with torch.no_grad():
        for i, (test_ts, train_feat) in enumerate(test_loader):
            if test_ts == None:
                continue
            if num_samples is not None and i >= num_samples:  # control sample number
                break
            real_sample = test_ts.float().to(device)

            loss, recon_error, reconstructed_sample, z = model.shared_eval(
                real_sample, None, mode="test"
            )
            print(verbose)
            if reconstructed_sample.dim() == 1:
                reconstructed_sample = reconstructed_sample.unsqueeze(0)
            if real_sample.dim() == 2:
                real_sample = real_sample.unsqueeze(0)
            real_np = real_sample.squeeze().cpu().numpy()
            reconstructed_np = reconstructed_sample.squeeze().cpu().numpy()
            real_samples.append(real_np)
            reconstructed_samples.append(reconstructed_np)
            if verbose:
                print(f"Real sample shape: {real_sample.shape}")
                print(f"Reconstructed sample shape: {reconstructed_sample.shape}")
                print(
                    f"Loss for batch {i}: {loss.item()}, Reconstruction Error: {recon_error.item()}"
                )
                plot_comparison(real_np, reconstructed_np, save_dir)
                break

    print("\n\n\n")
    real_samples = np.concatenate(real_samples, axis=0)
    reconstructed_samples = np.concatenate(reconstructed_samples, axis=0)
    if verbose:
        plot_pca_tsne(real_samples, reconstructed_samples, save_dir)
    mae = np.mean(np.abs(real_samples - reconstructed_samples))
    mse = np.mean((real_samples - reconstructed_samples) ** 2)
    rmse = np.sqrt(mse)
    print(f"MAE: {mae}\n")
    print(f"RMSE: {rmse}\n")


def any_length_evaluation(real_samples):
    samples_by_length = {24: [], 48: [], 96: []}
    for sample in real_samples:
        length = sample.shape[1]
        samples_by_length[length].append(sample)
    stacked_by_length = {}
    for length, samples in samples_by_length.items():
        stacked_by_length[length] = np.concatenate(samples, axis=0)
    stacked_24 = stacked_by_length[24]
    stacked_48 = stacked_by_length[48]
    stacked_96 = stacked_by_length[96]
    return stacked_24, stacked_48, stacked_96


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name", type=str, default="05s_windows", help="dataset name"
)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument(
    "--num_training_updates",
    type=int,
    default=60000,
    help="number of training updates/epochs",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="results/saved_pretrained_models/",
    help="denoiser model save path",
)
# Model-specific parameters
parser.add_argument(
    "--general_seed", type=int, default=11, help="seed for random number generation"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-3, help="learning rate for the optimizer"
)
parser.add_argument(
    "--block_hidden_size",
    type=int,
    default=128,
    help="hidden size of the blocks in the network",
)
parser.add_argument(
    "--num_residual_layers",
    type=int,
    default=2,
    help="number of residual layers in the model",
)
parser.add_argument(
    "--res_hidden_size",
    type=int,
    default=256,
    help="hidden size of the residual layers",
)
parser.add_argument(
    "--embedding_dim", type=int, default=64, help="dimension of the embeddings"
)
parser.add_argument(
    "--num_embeddings", type=int, default=128, help="number of embeddings in the VQ-VAE"
)
parser.add_argument(
    "--compression_factor", type=int, default=4, help="compression factor"
)
parser.add_argument(
    "--commitment_cost",
    type=float,
    default=0.25,
    help="commitment cost used in the loss function",
)
parser.add_argument(
    "--mix_train", type=bool, default=False, help="whether to use mixture training"
)
parser.add_argument(
    "--reuse_previous_model",
    type=int,
    default=1,
    help="0 no re use, other re use if possible",
)
args = parser.parse_args()


def pretrain_VAE(save_dir, args, time_checkback=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ndevice: {device}")
    seed_everything(args.general_seed)

    train_loader, val_loader, test_loader, feat_scaler, ts_scaler = (
        load_scaled_dataloaders(
            dataset_path="./Data/dataset_50_run_05s_downsample",
            batch_size=1024,
            scale_features=True,
            scale_series=True,
        )
    )
    train_loss = []
    val_loss = []
    num_batch = len(train_loader)
    print(f"num_batch : {num_batch}")

    if args.reuse_previous_model:
        path_model = os.path.join(save_dir, f"final_VAEpretrain_model.pth")
        if os.path.isfile(path_model):
            print("re use model")
            model = torch.load(
                path_model,
                map_location=torch.device(device),
            )
            optimizer = model.configure_optimizers(lr=args.learning_rate)
            with torch.no_grad():
                model.eval()
                epoch_val_loss = 0
                for j, data in enumerate(val_loader):
                    val_ts, val_feat = data
                    tensor_all_data_in_batch = (
                        val_ts.clone().detach().float().to(device)
                    )
                    v_loss, recon_error, x_recon, z = model.shared_eval(
                        tensor_all_data_in_batch, optimizer, "val"
                    )
                    epoch_val_loss += v_loss.item()

                epoch_val_loss = epoch_val_loss / len(val_loader)

            print(f"val_Loss initial: {epoch_val_loss}")
            val_loss.append(epoch_val_loss)
        else:
            print("no model available")
            model = vqvae(args).to(device)
            optimizer = model.configure_optimizers(lr=args.learning_rate)
    else:
        print("no re use model")
        model = vqvae(args).to(device)
        optimizer = model.configure_optimizers(lr=args.learning_rate)

    for epoch in range(2000):
        epoch_train_loss = 0
        epoch_val_loss = 0
        print(f"\nepoch: {epoch}\n")

        # Train
        model.train()
        for i, data in enumerate(train_loader):
            if i % 500 == 0:
                print(i)
            train_ts, train_feat = data
            tensor_all_data_in_batch = train_ts.clone().detach().float().to(device)
            t_loss, recon_error, x_recon, z = model.shared_eval(
                tensor_all_data_in_batch, optimizer, "train"
            )
            epoch_train_loss += t_loss.item()

        epoch_train_loss = epoch_train_loss / len(train_loader)

        # VALIDATION
        with torch.no_grad():
            model.eval()
            for j, data in enumerate(val_loader):
                val_ts, val_feat = data
                tensor_all_data_in_batch = val_ts.clone().detach().float().to(device)
                v_loss, recon_error, x_recon, z = model.shared_eval(
                    tensor_all_data_in_batch, optimizer, "val"
                )
                epoch_val_loss += v_loss.item()

            epoch_val_loss = epoch_val_loss / len(val_loader)

        print(
            f"Epoch: {epoch}, train_Loss: {epoch_train_loss}, val_Loss: {epoch_val_loss}"
        )

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        # SAVE
        if epoch_val_loss == min(val_loss):
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model, os.path.join(save_dir, f"final_VAEpretrain_model.pth"))
            print(f"Saved Model from epoch: {epoch}")

        # CHECK CALLBACK
        epsilon = 0.00001
        if len(val_loss) > time_checkback:
            list_checkback = []
            for i in range(time_checkback + 1):
                list_checkback.append(val_loss[-i - 1])
            if min(list_checkback) + epsilon > list_checkback[-1]:
                print("callback reach")
                break


if __name__ == "__main__":
    save_dir = "checkpoint/dataset_50_run_05s_downsample"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if False:
        print("Train model...")
        pretrain_VAE(save_dir=save_dir, args=args)

    if True:
        print("Starting inference...")

        model = torch.load(
            os.path.join(save_dir, "final_VAEpretrain_model.pth"), map_location=device
        )
        train_loader, val_loader, test_loader, feat_scaler, ts_scaler = (
            load_scaled_dataloaders(
                dataset_path="./Data/dataset_50_run_05s_downsample",
                batch_size=32,
                scale_features=True,
                scale_series=True,
            )
        )

        inference(model, test_loader, device, save_dir, num_samples=None, verbose=True)
