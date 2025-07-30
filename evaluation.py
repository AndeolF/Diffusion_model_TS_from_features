import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# from dtaidistance.dtw_ndim import distance as multi_dtw_distance
from evaluate.ts2vec import initialize_ts2vec
# from evaluate.feature_based_measures import (
#     calculate_mdd,
#     calculate_acd,
#     calculate_sd,
#     calculate_kd,
# )
import os
import datetime

# from evaluate.utils import (
#     show_with_start_divider,
#     show_with_end_divider,
#     determine_device,
#     write_json_data,
# )
import argparse
import torch  # type: ignore
from scipy.stats import norm

from model.backbone.rectified_flow import RectifiedFlow
from datafactory.dataloader_perso import load_scaled_dataloaders
from model.denoiser.mlp import MLP
from model.denoiser.transformer import Transformer
from model.backbone.DDPM import DDPM
import math
import pycatch22 as catch22  # type: ignore
from scipy.signal import welch
from scipy.signal import firwin
import torch.nn.functional as F  # type: ignore
from tqdm import tqdm

# model_8_mse_beta_10_08_no_curriculum

###################################################
#                    MRR                          #
###################################################


# def calculate_mrr(ori_data, gen_data, k=None):
#     n_batch_size = ori_data.shape[0]
#     n_generations = gen_data.shape[3]
#     k = n_generations if k is None else k

#     mrr_scores = np.zeros(n_batch_size)

#     for batch_idx in range(n_batch_size):
#         similarities = []
#         for gen_idx in range(k):
#             real_sequence = ori_data[batch_idx]
#             generated_sequence = gen_data[batch_idx, :, :, gen_idx]
#             similarity = cosine_similarity(real_sequence, generated_sequence)
#             similarities.append(np.mean(similarity))

#         sorted_indices = np.argsort(similarities)[::-1]
#         rank = None
#         for idx in sorted_indices:
#             if similarities[idx] > therehold:
#                 rank = idx + 1
#                 break

#         mrr_scores[batch_idx] = 1.0 / rank if rank is not None else 0.0

#     return np.mean(mrr_scores)


###################################################
#             other reconstruct:CRPS              #
###################################################


def calculate_crps(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_timesteps = ori_data.shape[1]
    n_series = ori_data.shape[2]
    n_generations = gen_data.shape[3]
    crps_values = []

    for i in range(n_samples):
        total_crps = 0

        for j in range(n_series):
            crps_list = []

            for k in range(n_generations):
                mean = gen_data[i, :, j, k].mean()
                std_dev = gen_data[i, :, j, k].std()
                if std_dev == 0:
                    std_dev += 1e-8
                obs_value = ori_data[i, :, j]
                cdf_obs = np.where(obs_value < mean, 0, 1)

                cdf_pred = norm.cdf(obs_value, loc=mean, scale=std_dev)

                crps = np.mean((cdf_obs - cdf_pred) ** 2)
                crps_list.append(crps)

            average_crps = np.mean(crps_list)
            total_crps += average_crps

        crps_values.append(total_crps / n_series)

    crps_values = np.array(crps_values)
    average_crps = crps_values.mean()
    return average_crps


# def evaluate_muldata(args, ori_data, gen_data):
#     show_with_start_divider(f"Evalution with settings:{args}")

#     # Parse configs
#     method_list = args.method_list
#     dataset_name = args.dataset_name
#     model_name = args.model_name
#     device = args.device
#     evaluation_save_path = args.evaluation_save_path

#     now = datetime.datetime.now()
#     formatted_time = now.strftime("%Y%m%d-%H%M%S")
#     combined_name = f"{model_name}_{dataset_name}_{formatted_time}_multi"

#     if not isinstance(method_list, list):
#         method_list = method_list.strip("[]")
#         method_list = [method.strip() for method in method_list.split(",")]
#     if gen_data is None:
#         show_with_end_divider("Error: Generated data not found.")
#         return None

#     result = {}

#     if "CRPS" in method_list:
#         mdd = calculate_crps(ori_data, gen_data)
#         result["CRPS"] = mdd
#     if "MRR" in method_list:
#         mrr = calculate_mrr(ori_data, gen_data)
#         result["MRR"] = mrr

#     if isinstance(result, dict):
#         evaluation_save_path = os.path.join(
#             evaluation_save_path, f"{combined_name}.json"
#         )
#         write_json_data(result, evaluation_save_path)
#         print(f"Evaluation denoiser_results saved to {evaluation_save_path}.")

#     show_with_end_divider(f"Evaluation done. Results:{result}.")

#     return result


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_frechet(ori_data,gen_data,device):
    fid_model = torch.load(
        "checkpoint/dataset_50_run_05s_downsample/fid_model.pth",map_location=device
    )
    ori_repr = fid_model.encode(
        ori_data, encoding_window="full_series"
    )
    gen_repr = fid_model.encode(
        gen_data, encoding_window="full_series"
    )
    cfid = calculate_fid(ori_repr, gen_repr)
    return cfid

def calculate_ed(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0
        for j in range(n_series):
            distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_distance_eu += distance
        distance_eu.append(total_distance_eu / n_series)

    distance_eu = np.array(distance_eu)
    average_distance_eu = distance_eu.mean()
    return average_distance_eu


# def calculate_dtw(ori_data, comp_data):
#     distance_dtw = []
#     n_samples = ori_data.shape[0]
#     for i in range(n_samples):
#         distance = multi_dtw_distance(
#             ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True
#         )
#         distance_dtw.append(distance)

#     distance_dtw = np.array(distance_dtw)
#     average_distance_dtw = distance_dtw.mean()
#     return average_distance_dtw


import numpy as np


def calculate_mse(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    mse_values = []

    for i in range(n_samples):
        total_mse = 0
        for j in range(n_series):
            mse = np.mean((ori_data[i, :, j] - gen_data[i, :, j]) ** 2)
            total_mse += mse
        mse_values.append(total_mse / n_series)

    mse_values = np.array(mse_values)
    average_mse = mse_values.mean()
    return average_mse


def calculate_wape(ori_data, gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    wape_values = []

    for i in range(n_samples):
        total_absolute_error = 0
        total_actual_value = 0

        for j in range(n_series):
            absolute_error = np.abs(ori_data[i, :, j] - gen_data[i, :, j])
            total_absolute_error += np.sum(absolute_error)
            total_actual_value += np.sum(np.abs(ori_data[i, :, j]))

        if total_actual_value != 0:
            wape = total_absolute_error / total_actual_value
        else:
            wape = np.nan

        wape_values.append(wape)

    wape_values = np.array(wape_values)
    average_wape = np.nanmean(wape_values)
    return average_wape


def get_power_t_schedule(step, gamma=3.0, device="cuda"):
    """
    Génère une séquence t non uniforme de [1, 0], avec plus de pas vers t ~ 0
    gamma > 1 → plus dense vers 0
    gamma < 1 → plus dense vers 1
    """
    s = torch.linspace(0, 1, steps=step, device=device)
    t = 1 - s**gamma
    return t


def infer(args):
    step = args.total_step
    cfg_scale = args.cfg_scale
    generation_save_path_result = args.generation_save_path_result
    usepretrainedvae = args.usepretrainedvae
    device = args.device

    print(
        f"Inference config::Step: {step}\t CFG Scale: {cfg_scale}\t Use Pretrained VAE: {usepretrainedvae}"
    )
    if args.save_data:
        os.makedirs(generation_save_path_result, exist_ok=True)
    #
    train_loader, val_loader, test_loader, feat_scaler, ts_scaler = (
        load_scaled_dataloaders(
            dataset_path="./Data/dataset_50_run_05s_downsample",
            batch_size=args.batch_size,
            scale_features=True,
            scale_series=True,
        )
    )
    #
    dataloader = val_loader
    print("number of batch:", len(dataloader))
    max_batch_to_see = 200

    pretrained_model = torch.load(
        args.pretrained_model_path,
        map_location=torch.device(device),
    )
    pretrained_model.float().to(device).eval()
    model = {"DiT": Transformer, "MLP": MLP}.get(args.denoiser)
    if model:
        model = model().to(args.device)
    else:
        raise ValueError(f"No denoiser found")
    model.encoder = pretrained_model.encoder
    model.load_state_dict(torch.load(args.save_model_path)["model"])
    model.to(device).eval()
    saved_dict = torch.load(
        args.save_model_path, map_location=torch.device(args.device)
    )
    print(f"\nLoss val : {saved_dict["loss_val"]}\n")
    backbone = {
        "flowmatching": RectifiedFlow(args.device),
        "ddpm": DDPM(args.total_step, args.device),
    }.get(args.backbone)
    if backbone:
        if args.backbone == "flowmatching":
            rf = backbone
        elif args.backbone == "ddpm":
            ddpm = backbone
    else:
        raise ValueError(f"No backbone found")

    x_1_list = []
    x_t_list = []
    feat_list = []
    y_list = []
    x_1_latent_list = []
    x_t_latent_dec_list = []
    x_infer_list = []
    compteur = 0
    std_ref = np.load(
        "Data/dataset_50_run_05s_downsample/std_latent_data_train_encode.npy"
    )
    std_ref = torch.from_numpy(std_ref).to(device)

    f_s = 1200  # Hz, fréquence d’échantillonnage
    f_c = 230  # Hz, fréquence de coupure
    numtaps = 101  # taille du noyau (doit être impair pour symétrie)
    window_choose =  ("kaiser", 8.6)
    # ("kaiser", 8.6) "hamming"
    #
    # Noyau FIR passe-bas
    kernel_np = firwin(numtaps=numtaps, cutoff=f_c, fs=f_s, window=window_choose)
    kernel = torch.tensor(kernel_np, dtype=torch.float32).view(1, 1, -1).to(args.device)

    with torch.no_grad():
        for batch, data in enumerate(
            tqdm(dataloader, total=min(max_batch_to_see, len(dataloader)), ncols=100)
        ):
            # if batch % 20 == 0:
            #     print(f"Generating {batch}th Batch TS...")

            infer_ts, feat = data
            x_1 = infer_ts.float().to(device)
            feat = feat.float().to(device)
            x_1_latent, before = model.encoder(x_1)
            x_1_latent_copy = x_1_latent.clone()

            # WITH A TEMPORAL STD FILTRED WHITE NOISE
            # x_0_ts = torch.randn_like(infer_ts).float().to(device)
            # x_0_ts = x_0_ts.unsqueeze(1)
            # x_0_filtred_ts = F.conv1d(x_0_ts, kernel, padding=numtaps // 2)
            # x_0_filtred_ts = x_0_filtred_ts.squeeze(1)
            # x_0_filtred_latent, _ = model.encoder(x_0_filtred_ts)
            # x_t_latent = (x_0_filtred_latent / x_0_filtred_latent.std()) * std_ref

            # WITH A WHITE STD NOISE
            x_t_latent = torch.randn_like(x_1_latent_copy).float().to(device)

            # WITH A PINK STD NOISE
            # x_t_initial = generate_pink_noise_nd(x_1.shape).to(device)
            # x_t_latent, before = model.encoder(x_t_initial)

            x_t_latent = (x_t_latent / x_t_latent.std()) * std_ref

            # t SCHEDULER POWER
            t_schedule = get_power_t_schedule(step=step+1, gamma=3.0, device=device)

            for j in range(step):
                if args.backbone == "flowmatching":

                    # t SCHEDULER UNIFORM
                    # t = (
                    #     torch.round(
                    #         torch.full(
                    #             (x_t_latent.shape[0],), j * 1.0 / step, device=device
                    #         )
                    #         * step
                    #     )
                    #     / step
                    # )

                    # t SCHEDULER POWER
                    t = t_schedule[-j - 1].expand(x_t_latent.shape[0])
                    t_next = t_schedule[-j - 2].expand(x_t_latent.shape[0])
                    # print(f"\nt :{torch.mean(t)}")
                    # print(f"t_next :{torch.mean(t_next)}")
                    # print(f"dt : {torch.mean(t_next-t)}")

                    # # # EVAL WITH EULER  min(torch.mean(t_next-t),1/args.total_step)
                    pred_uncond = model(input=x_t_latent, t=t, text_input=None)
                    pred_cond = model(input=x_t_latent, t=t, text_input=feat)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t_latent = rf.euler(x_t=x_t_latent, v=pred, dt=1/args.total_step)

                    # EVAL WITH HEUN
                    # x_t_latent = rf.heun(model=model, x_t=x_t_latent, t=torch.mean(t), t_next=torch.mean(t_next), feat=feat, cfg_scale=cfg_scale, dt=torch.mean(t_next-t))

                elif args.backbone == "ddpm":
                    t = torch.full(
                        (x_t_latent.size(0),),
                        math.floor(step - 1 - j),
                        dtype=torch.long,
                        device=device,
                    )
                    pred_uncond = model(input=x_t_latent, t=t, text_input=None)
                    pred_cond = model(input=x_t_latent, t=t, text_input=feat)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t_latent = ddpm.p_sample(x_t, pred, t)

            x_t_latent_dec = x_t_latent.clone()
            x_t, after = pretrained_model.decoder(x_t_latent, length=x_1.shape[-1])
            x_1_through_VAE, after = pretrained_model.decoder(
                x_1_latent_copy, length=x_1.shape[-1]
            )

            # TEST WITH A FLITRED ON THE OUTPUT
            x_t = x_t.unsqueeze(1)
            x_t_filtred = F.conv1d(x_t, kernel, padding=numtaps // 2)
            x_t_filtred = x_t_filtred.squeeze(1)
            x_t = x_t_filtred

            x_1 = x_1.detach().cpu().numpy().squeeze()
            x_1_through_VAE = x_1_through_VAE.detach().cpu().numpy().squeeze()
            x_t = x_t.detach().cpu().numpy().squeeze()
            feat = feat.detach().cpu().numpy().squeeze()

            x_1_list.append(x_1_through_VAE)
            x_t_list.append(x_t)
            feat_list.append(feat)

            x_t_latent_dec = x_t_latent_dec.detach().cpu().numpy().squeeze()
            x_1_latent_copy = x_1_latent_copy.detach().cpu().numpy().squeeze()

            x_t_latent_dec_list.append(x_t_latent_dec)
            x_1_latent_list.append(x_1_latent_copy)

            compteur += 1
            if compteur == max_batch_to_see:
                break

    x_1_array = np.concatenate(x_1_list, axis=0)
    x_t_array = np.concatenate(x_t_list, axis=0)
    feat_array = np.concatenate(feat_list, axis=0)

    x_t_latent_dec_array = np.concatenate(x_t_latent_dec_list, axis=0)
    x_1_latent_array = np.concatenate(x_1_latent_list, axis=0)

    # x_1 = x_1_array[:, :, np.newaxis]
    # x_t = x_t_array[:, :, np.newaxis]
    # feat = feat_array[:, :, np.newaxis]
    x_1 = x_1_array
    x_t = x_t_array
    feat = feat_array

    # SAVE FIGURE

    if args.save_data:
        np.save(os.path.join(generation_save_path_result, "x_1.npy"), x_1)
        np.save(os.path.join(generation_save_path_result, "x_t.npy"), x_t)
        np.save(os.path.join(generation_save_path_result, "features.npy"), feat)
        np.save(
            os.path.join(generation_save_path_result, "x_t_latent_dec_array.npy"),
            x_t_latent_dec_array,
        )
        np.save(
            os.path.join(generation_save_path_result, "x_1_latent_array.npy"),
            x_1_latent_array,
        )

    print(x_1_latent_array.mean(), x_t_latent_dec_array.mean())
    print(x_1_latent_array.std(), x_t_latent_dec_array.std())
    print(x_t.shape)
    print(f"moyenne de l'ecart type de x_t : {np.mean(x_t.std(axis=0))}")
    print(f"moyenne de l'ecart type de x_1 : {np.mean(x_1.std(axis=0))}")

    return x_1, x_t, feat, x_t_latent_dec_array, x_1_latent_array


def generate_pink_noise_nd(shape, alpha=1.0):
    white_noise = np.random.randn(*shape)
    noise_fft = np.fft.fftn(white_noise)

    # Fréquences nD
    freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    freq_magnitude = np.sqrt(sum(f**2 for f in freqs))

    # Éviter division par zéro à f = 0
    freq_magnitude[freq_magnitude == 0] = 1.0

    # Filtre en 1/f^alpha
    filter_mask = 1.0 / (freq_magnitude ** (alpha / 2))
    filtered_fft = noise_fft * filter_mask  # Garde la phase, atténue les amplitudes

    pink_noise = np.fft.ifftn(filtered_fft).real

    # Normalisation
    pink_noise = (pink_noise - pink_noise.mean()) / pink_noise.std()
    return torch.tensor(pink_noise, dtype=torch.float32)


def calculer_psd_moyen(series_temporelles, name="", fs=1200.0, show=False):
    # Initialiser une liste pour stocker les PSDs
    psds = []

    # Calculer la PSD pour chaque série temporelle
    for serie in series_temporelles:
        frequences, psd = welch(serie, fs=fs)
        psds.append(psd)

    # Convertir la liste des PSDs en un tableau NumPy
    psds = np.array(psds)

    # Calculer la PSD moyenne
    psd_moyen = np.mean(psds, axis=0)

    if show:
        # Afficher la PSD moyenne
        plt.figure()
        plt.semilogy(frequences, psd_moyen)
        plt.xlabel("Fréquence")
        plt.ylabel("Densité spectrale de puissance")
        plt.title(f"PSD Moyenne des Séries Temporelles de {name}")
        plt.show()

    return psd_moyen,frequences


# def plot_log_scale(psd_ref,psd_gen,frequences):
#     plt.figure(figsize=(10, 6))

#     plt.plot(psd_ref, label="psd_ref")
#     plt.plot(psd_gen, label="psd_gen")

#     plt.yscale("log")  # Mettre l'axe des y en échelle logarithmique
#     plt.xlabel("Index")
#     plt.ylabel("Valeur (échelle log)")
#     plt.title("Graphique en échelle logarithmique")
#     plt.legend()
#     plt.grid(True, which="both", ls="-")
#     plt.show()


def plot_log_scale(psd_ref, psd_gen, frequences):
    plt.figure(figsize=(10, 6))

    plt.plot(frequences, psd_ref, label="psd_ref")
    plt.plot(frequences, psd_gen, label="psd_gen")

    plt.yscale("log")  # Mettre l'axe des y en échelle logarithmique
    # plt.xscale("log")  # Mettre l'axe des x en échelle logarithmique si nécessaire
    plt.xlabel("Fréquence")
    plt.ylabel("Valeur (échelle log)")
    plt.title("Graphique en échelle logarithmique")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()

# Test fonction bruit rose :
# array_1 = np.zeros(1200)
# array_2 = np.zeros(1200)
# bruit_blanc_1 = np.random.randn(*array_1.shape)
# bruit_blanc_2 = np.random.randn(*array_2.shape)
# bruit_rose_1 = generate_pink_noise_nd(bruit_blanc_1.shape)
# bruit_rose_2 = generate_pink_noise_nd(bruit_blanc_2.shape)
# list_bruit_blanc = [bruit_blanc_1, bruit_blanc_2]
# list_bruit_rose = [bruit_rose_1, bruit_rose_2]
# calculer_psd_moyen(list_bruit_blanc, name="blanc")
# calculer_psd_moyen(list_bruit_rose, name="rose")


# def evaluate_data(args, ori_data, gen_data):
#     show_with_start_divider(f"Evalution with settings:{args}")

#     # Parse configs
#     method_list = args.method_list
#     dataset_name = args.dataset_name
#     model_name = args.model_name
#     device = args.device
#     evaluation_save_path = args.evaluation_save_path

#     now = datetime.datetime.now()
#     formatted_time = now.strftime("%Y%m%d-%H%M%S")
#     combined_name = f"{model_name}_{dataset_name}_{formatted_time}"

#     if not isinstance(method_list, list):
#         method_list = method_list.strip("[]")
#         method_list = [method.strip() for method in method_list.split(",")]

#     if gen_data is None:
#         show_with_end_divider("Error: Generated data not found.")
#         return None
#     if ori_data.shape != gen_data.shape:
#         print(
#             f"Original data shape: {ori_data.shape}, Generated data shape: {gen_data.shape}."
#         )
#         show_with_end_divider(
#             "Error: Generated data does not have the same shape with original data."
#         )
#         return None

#     result = {}

#     if "C-FID" in method_list:
#         fid_model = initialize_ts2vec(np.transpose(ori_data, (0, 2, 1)), device)
#         ori_repr = fid_model.encode(
#             np.transpose(ori_data, (0, 2, 1)), encoding_window="full_series"
#         )
#         gen_repr = fid_model.encode(
#             np.transpose(gen_data, (0, 2, 1)), encoding_window="full_series"
#         )
#         cfid = calculate_fid(ori_repr, gen_repr)
#         result["C-FID"] = cfid

#     ori_data = np.transpose(ori_data, (0, 2, 1))
#     gen_data = np.transpose(gen_data, (0, 2, 1))

#     if "MSE" in method_list:
#         mse = calculate_mse(ori_data, gen_data)
#         result["MSE"] = mse
#     if "WAPE" in method_list:
#         wape = calculate_wape(ori_data, gen_data)
#         result["WAPE"] = wape

#     ori_data = np.transpose(ori_data, (0, 2, 1))
#     gen_data = np.transpose(gen_data, (0, 2, 1))

#     if isinstance(result, dict):
#         evaluation_save_path = os.path.join(
#             evaluation_save_path, f"{combined_name}.json"
#         )
#         write_json_data(result, evaluation_save_path)
#         print(f"Evaluation denoiser_results saved to {evaluation_save_path}.")

#     show_with_end_divider(f"Evaluation done. Results:{result}.")

#     return result


def compute_features(x):
    """OBTAINED VIA THE CATCH22 WEBSITE"""
    res = catch22.catch22_all(x, catch24=True)
    return res["values"]


def compute_features_from_2D_array(array):
    result = []
    for ts in array:
        feature = compute_features(ts)
        result.append(feature)
    return np.array(result)


#
#
#


def get_args():

    parser = argparse.ArgumentParser(description="Test Features2S model")

    parser.add_argument(
        "--method_list",
        type=str,
        default="MSE,WAPE,MRR",
        help="metric list [MSE,WAPE,MRR]",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset_50_run_05s_downsample",
        help="dataset name",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoint/",
        help="checkpoint path",
    )
    parser.add_argument(
        "--reuse_previous_model",
        type=int,
        default=1,
        help="0 no re use, other re use if possible",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--epochs", type=int, default=20000, help="training epochs")

    parser.add_argument("--cfg_scale", type=float, default=5, help="CFG Scale")

    # model specific
    parser.add_argument("--usepretrainedvae", default=True, help="pretrained vae")
    parser.add_argument(
        "--total_step", type=int, default=100, help="sampling from [0,1]"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="flowmatching",
        help="flowmatching or ddpm or edm",
    )
    parser.add_argument("--denoiser", type=str, default="DiT", help="DiT or MLP")

    # for inference
    # ID DU MODEL
    # ID DU MODEL
    parser.add_argument("--checkpoint_id", type=int, default=22, help="model id")
    # ID DU MODEL
    # ID DU MODEL

    parser.add_argument(
        "--run_multi",
        type=bool,
        default=False,
        help="run multi times for CRPS,MAP,MRR,NDCG",
    )
    parser.add_argument(
        "--save_data",
        type=int,
        default=0,
        help="0 no save, other save if possible",
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.pretrained_model_path = (
        f"checkpoint/dataset_50_run_05s_downsample/final_VAEpretrain_model.pth"
    )

    args.checkpoint_path = os.path.join(
        args.checkpoint_path,
        "{}".format(args.dataset_name),
    )
    args.save_model_path = os.path.join(
        args.checkpoint_path,
        "{}_{}".format(args.backbone, args.denoiser),
        "model_{}.pth".format(args.checkpoint_id),
    )

    args.generation_save_path = os.path.join(
        args.checkpoint_path,
        "{}_{}".format(args.backbone, args.denoiser),
        "generation_model{}".format(args.checkpoint_id),
        "{}_{}_{}_cfg{}_step{}".format(
            args.backbone,
            args.denoiser,
            args.dataset_name,
            args.cfg_scale,
            args.total_step,
        ),
    )
    args.generation_save_path_result = os.path.join(args.generation_save_path)
    return args


#
#
#


if __name__ == "__main__":

    import pickle

    with open("Data/dataset_50_run_05s_downsample_for_scaler/features_scaler.pkl", "rb") as file:
        features_scaler = pickle.load(file)

    with open("Data/dataset_50_run_05s_downsample_for_scaler/ts_scaler.pkl", "rb") as file:
        ts_scaler = pickle.load(file)

    args = get_args()
    x_1, x_t, features, x_t_latent_dec_array, x_t_latent_enc_array = infer(args)

    ts_ref_destandardise = ts_scaler.inverse_transform(x_1)
    ts_gen_destandardise = ts_scaler.inverse_transform(x_t)

    psd_mean_ref,frequences = calculer_psd_moyen(x_1, name="ref",fs=1200.0,show=False)
    psd_mean_gen,_ = calculer_psd_moyen(x_t, name="gen",fs=1200.0,show=False)

    #
    plot_log_scale(psd_mean_ref, psd_mean_gen,frequences)
    #

    distance_psd = np.linalg.norm(psd_mean_ref - psd_mean_gen)
    print("\ndistance_psd")
    print(distance_psd)

    features_compute_ref = np.array(
        compute_features_from_2D_array(ts_ref_destandardise)
    )
    features_compute_gen = np.array(
        compute_features_from_2D_array(ts_gen_destandardise)
    )

    standardise_features_compute_ref = features_scaler.transform(features_compute_ref)
    standardise_features_compute_gen = features_scaler.transform(features_compute_gen)

    distance_features_ref_gen = np.linalg.norm(
        standardise_features_compute_ref - standardise_features_compute_gen,
        axis=-1,
        keepdims=True,
    )

    print("\nDistance features :")
    print(np.mean(distance_features_ref_gen))

    # print("\nFID :")
    # x_1 = x_1.reshape(x_1.shape[0], x_1.shape[1], 1)
    # x_t = x_t.reshape(x_t.shape[0], x_t.shape[1], 1)
    # fid = get_frechet(ori_data=x_1, gen_data=x_t,device=args.device)
    # print(fid)


#
#
#


# """evaluate our model"""
# x_1 = np.load(os.path.join(args.generation_save_path, "run_0", "x_1.npy"))
# x_t = np.load(os.path.join(args.generation_save_path, "x_t.npy"))
# x_t_latent_dec_array = np.load(
#     os.path.join(args.generation_save_path, "run_0", "x_t_latent_dec_array.npy")
# )
# x_t_latent_enc_array = np.load(
#     os.path.join(args.generation_save_path, "run_0", "x_t_latent_enc_array.npy")
# )
# x_1 = np.transpose(x_1, (0, 2, 1))
# x_t = np.transpose(x_t, (0, 2, 1))
# # print(f'x_1 shape:{x_1.shape}')
# # print(f'x_t shape:{x_t.shape}')
# evaluate_data(args, ori_data=x_1, gen_data=x_t)  # batch, dim , time length

# therehold = 0.5
# all_x_t = []
# for run_index in range(10):
#     """Choice 1 : evaluate our model"""
#     args.generation_save_path_result = os.path.join(
#         args.generation_save_path, f"run_{run_index}"
#     )
#     x_1 = np.load(os.path.join(args.generation_save_path_result, "x_1.npy"))
#     x_t = np.load(os.path.join(args.generation_save_path_result, "x_t.npy"))

#     x_t_expanded = np.expand_dims(x_t, axis=-1)
#     all_x_t.append(x_t_expanded)

# x_t_all = np.concatenate(all_x_t, axis=-1)
# evaluate_muldata(args, ori_data=x_1, gen_data=x_t_all)
