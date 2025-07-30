import argparse
import torch  # type: ignore
from matplotlib import pyplot as plt
from model.denoiser.mlp import MLP
from model.denoiser.transformer import Transformer
from datafactory.dataloader import loader_provider
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from matplotlib.animation import FuncAnimation
import os
import numpy as np
import math
import random
from datafactory.dataloader_perso import load_scaled_dataloaders
import pycatch22 as catch22  # type: ignore
from scipy.signal import welch
from scipy.signal import firwin
import torch.nn.functional as F  # type: ignore


def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True


def compute_features(x):
    """OBTAINED VIA THE CATCH22 WEBSITE"""
    res = catch22.catch22_all(x, catch24=True)
    return res["values"]


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
    # os.makedirs(generation_save_path_result, exist_ok=True)  dataset_50_run_05s_downsample
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
    print("val dataset length:", len(dataloader))

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
    print("Loss val :")
    print(saved_dict["loss_val"])
    print()
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
    list_psd = []
    compteur = 0
    std_ref = np.load(
        "Data/dataset_50_run_05s_downsample/std_latent_data_train_encode.npy"
    )
    std_ref = torch.from_numpy(std_ref).to(device)

    f_s = 1200  # Hz, fréquence d’échantillonnage
    f_c = 230  # Hz, fréquence de coupure
    numtaps = 101  # taille du noyau (doit être impair pour symétrie)
    window_choose = ("kaiser", 8.6)
    # ("kaiser", 8.6)  "hamming"
    # Noyau FIR passe-bas
    #
    kernel_np = firwin(numtaps=numtaps, cutoff=f_c, fs=f_s, window=window_choose)
    kernel = torch.tensor(kernel_np, dtype=torch.float32).view(1, 1, -1).to(args.device)

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            print(f"Generating {batch}th Batch TS...")

            infer_ts, infer_feat = data
            x_1 = infer_ts.float().to(device)
            infer_feat = infer_feat.float().to(device)

            x_1_latent, before = model.encoder(x_1)
            x_1_latent_copy = x_1_latent.clone()
            x_t_latent = torch.randn_like(x_1_latent).float().to(device)
            # x_t = generate_pink_noise_nd(x_t.shape).to(device)
            x_t_latent = x_t_latent / x_t_latent.std() * std_ref

            # t SCHEDULER POWER
            t_schedule = get_power_t_schedule(step=step + 1, gamma=2.5, device=device)

            for j in range(step):
                if args.backbone == "flowmatching":
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

                    # EVAL WITH EULER euler(self, x_t, v, dt)  torch.mean(t_next - t)
                    pred_uncond = model(input=x_t_latent, t=t, text_input=None)
                    pred_cond = model(input=x_t_latent, t=t, text_input=infer_feat)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t_latent = rf.euler(
                        x_t=x_t_latent, v=pred, dt=1 / args.total_step
                    )

                    # EVAL WITH HEUN

                elif args.backbone == "ddpm":
                    t = torch.full(
                        (x_t_latent.size(0),),
                        math.floor(step - 1 - j),
                        dtype=torch.long,
                        device=device,
                    )
                    pred_uncond = model(input=x_t_latent, t=t, text_input=None)
                    pred_cond = model(input=x_t_latent, t=t, text_input=infer_feat)

                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t_latent = ddpm.p_sample(x_t_latent, pred, t)

                # In order to see the whole process step by step ones
                if batch == 0:
                    x_t_mid, after = pretrained_model.decoder(
                        x_t_latent, length=x_1.shape[-1]
                    )
                    # TEST WITH A FLITRED
                    x_t_mid = x_t_mid.unsqueeze(1)
                    x_t_filtred = F.conv1d(x_t_mid, kernel, padding=numtaps // 2)
                    x_t_filtred = x_t_filtred.squeeze(1)
                    x_t_mid = x_t_filtred

                    x_t_mid = x_t_mid.detach().cpu().numpy().squeeze()
                    x_infer_list.append(x_t_mid[0])
            x_t_latent_dec = x_t_latent.clone()
            x_t, after = pretrained_model.decoder(x_t_latent, length=x_1.shape[-1])

            # Then we add the true one
            # So the x_infer_list a pour forme les x_t temporaire

            x_1 = x_1.detach().cpu().numpy().squeeze()

            # TEST WITH A FLITRED
            x_t = x_t.unsqueeze(1)
            x_t_filtred = F.conv1d(x_t, kernel, padding=numtaps // 2)
            x_t_filtred = x_t_filtred.squeeze(1)
            x_t = x_t_filtred

            x_t = x_t.detach().cpu().numpy().squeeze()
            infer_feat = infer_feat.detach().cpu().numpy().squeeze()
            x_1_list.append(x_1)
            x_t_list.append(x_t)
            feat_list.append(infer_feat)

            x_t_latent_dec = x_t_latent_dec.detach().cpu().numpy().squeeze()
            x_1_latent_copy = x_1_latent_copy.detach().cpu().numpy().squeeze()

            x_t_latent_dec_list.append(x_t_latent_dec)
            x_1_latent_list.append(x_1_latent_copy)

            compteur += 1
            if compteur == 1:
                break

    x_1_array = np.concatenate(x_1_list, axis=0)
    x_t_array = np.concatenate(x_t_list, axis=0)
    feat_array = np.concatenate(feat_list, axis=0)

    x_t_latent_dec_array = np.concatenate(x_t_latent_dec_list, axis=0)
    x_1_latent_array = np.concatenate(x_1_latent_list, axis=0)

    print("x_1_array")
    print(x_1_array.shape)

    x_1 = x_1_array[:, :, np.newaxis]
    x_t = x_t_array[:, :, np.newaxis]
    feat = feat_array

    print("x_1")
    print(x_1.shape)

    print(x_1_latent_array.mean(), x_t_latent_dec_array.mean())
    print(x_1_latent_array.std(), x_t_latent_dec_array.std())

    return x_1, x_t, feat, x_t_latent_dec_array, x_1_latent_array, x_infer_list


def generate_pink_noise_nd(shape, alpha=1.0):
    """
    Génère du bruit rose en nD avec décroissance spectrale en 1/f^alpha.
    """
    # 1. Génère bruit blanc
    white_noise = np.random.randn(*shape)

    # 2. FFT
    noise_fft = np.fft.fftn(white_noise)

    # 3. Construire la magnitude fréquentielle
    freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    freq_magnitude = np.sqrt(sum(f**2 for f in freqs))
    freq_magnitude[0] = 1  # pour éviter la division par zéro à f=0

    # 4. Filtrage en 1/f^α
    filtered_fft = noise_fft / (freq_magnitude ** (alpha / 2))

    # 5. Revenir au domaine spatial
    pink_noise = np.fft.ifftn(filtered_fft).real

    # 6. Normalisation
    pink_noise = (pink_noise - pink_noise.mean()) / pink_noise.std()

    return torch.tensor(pink_noise, dtype=torch.float32)


def display_psd(series_temporelles, name="", fs=1200.0, verbose=True):
    psd = welch(series_temporelles, fs=fs)
    frequences, values = psd

    # Convertir la liste des PSDs en un tableau NumPy
    values = np.array(values)

    if verbose:
        # Afficher la PSD moyenne
        plt.figure()
        plt.semilogy(frequences, values)
        plt.xlabel("Fréquence")
        plt.ylabel("Densité spectrale de puissance")
        plt.title(f"PSD Moyenne des Séries Temporelles de {name}")
        plt.show()

    return psd


def display_double_psd(psd_1, psd_2, name_1="1", name_2="2"):

    freqs_1, values_1 = psd_1
    freqs_2, values_2 = psd_2

    plt.figure(figsize=(10, 6))
    plt.plot(freqs_1, values_1, label=name_1)
    plt.plot(freqs_2, values_2, label=name_2)
    plt.yscale("log")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Densité Spectrale de Puissance")
    plt.title("Comparaison de deux PSD")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference flow matching model")

    #
    #
    #

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

    # Pas bien compris ce machin la
    parser.add_argument("--cfg_scale", type=float, default=4, help="CFG Scale")

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
    parser.add_argument("--checkpoint_id", type=int, default=33, help="model id")
    # ID DU MODEL
    # ID DU MODEL

    parser.add_argument(
        "--run_multi",
        type=bool,
        default=False,
        help="run multi times for CRPS,MAP,MRR,NDCG",
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
        "generation",
        "{}_{}_{}_{}_{}".format(
            args.backbone,
            args.denoiser,
            args.dataset_name,
            args.cfg_scale,
            args.total_step,
        ),
    )

    #
    #
    #

    if args.run_multi:
        # single
        args.generation_save_path_result = os.path.join(args.generation_save_path)
        x_1, x_t, x_t_latent_dec_array, x_t_latent_enc_array, x_infer_list = infer(args)
        for run_index in range(10):
            # multi
            args.generation_save_path_result = os.path.join(
                args.generation_save_path, f"run_{run_index}"
            )
            (
                x_1,
                x_t,
                features_ref,
                x_t_latent_dec_array,
                x_t_latent_enc_array,
                x_infer_list,
            ) = infer(args)
    else:
        args.generation_save_path_result = os.path.join(args.generation_save_path)
        (
            x_1,
            x_t,
            features_ref,
            x_t_latent_dec_array,
            x_t_latent_enc_array,
            x_infer_list,
        ) = infer(args)

        #
        import pickle

        # dataset_50_run_05s_downsample_for_scaler
        with open(
            "Data/dataset_50_run_05s_downsample_for_scaler/features_scaler.pkl", "rb"
        ) as file:
            features_scaler = pickle.load(file)

        with open(
            "Data/dataset_50_run_05s_downsample_for_scaler/ts_scaler.pkl", "rb"
        ) as file:
            ts_scaler = pickle.load(file)

        for i in range(30):
            np.set_printoptions(
                precision=6, suppress=True, floatmode="fixed", linewidth=100
            )
            ts_ref_destandardise = ts_scaler.inverse_transform(x_1[i].reshape(1, -1))
            ts_gen_destandardise = ts_scaler.inverse_transform(x_t[i].reshape(1, -1))
            print(f"shape ts_ref_destandardise : {ts_ref_destandardise.shape}")

            features_compute_ref = np.array(compute_features(ts_ref_destandardise[0]))
            features_compute_gen = np.array(compute_features(ts_gen_destandardise[0]))
            print("\n\n")
            print(features_compute_ref[-1])
            print(features_compute_gen[-1])
            print("\n\n")

            standardise_features_compute_ref = features_scaler.transform(
                features_compute_ref.reshape(1, -1)
            )
            standardise_features_compute_gen = features_scaler.transform(
                features_compute_gen.reshape(1, -1)
            )

            print("catch22 features :")
            # print(
            #     f"features compute from series ref :\n {standardise_features_compute_ref[0]}"
            # )
            print(f"features no std gen : \n{features_compute_gen}")
            print(
                f"features compute from series genereted :\n {standardise_features_compute_gen[0]}"
            )
            print(f"features no std ref : \n{features_compute_ref}")
            print(f"features ref :\n {features_ref[i]}")
            print()

            plt.figure(figsize=(10, 5))
            plt.plot(x_1[i].squeeze(), label="Série originale (x₁)", linewidth=2)
            plt.plot(x_t[i].squeeze(), label="Série générée (xₜ)", linewidth=2)
            # plt.plot(
            #     ts_ref_destandardise[0],
            #     label="Série originale (x₁)",
            #     linewidth=2,
            # )
            # plt.plot(ts_gen_destandardise[0], label="Série générée (xₜ)", linewidth=2)

            # Re-passer le latent bruité (x_t_latent_enc_array) par le décodeur VAE
            # pour obtenir une reconstruction de x₁ sans débruitage
            # x_t_latent_enc = (
            #     torch.tensor(x_t_latent_enc_array[i])
            #     .unsqueeze(0)
            #     .float()
            #     .to(args.device)
            # )
            # x_recon_enc, _ = torch.load(
            #     args.pretrained_model_path, map_location=torch.device(args.device)
            # ).decoder(x_t_latent_enc, length=x_1.shape[1])
            # x_recon_enc = x_recon_enc.squeeze().detach().cpu().numpy()

            # plt.plot(
            #     x_recon_enc, label="Latent bruité reconstruit (x̃ₜ)", linestyle="--"
            # )

            plt.title(f"Série temporelle {i}")
            plt.xlabel("Temps")
            plt.ylabel("Valeur")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        def get_ts_gif(x_infer_list, x_ref):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            x = np.arange(len(x_infer_list[0]))
            (line,) = ax.plot(x, x_infer_list[0], color="cornflowerblue", lw=3)
            (fixed_line,) = ax.plot(x, x_ref, color="black", lw=2, label="static line")
            ax.set_ylim(-2, 2)

            def init():
                line.set_ydata([np.nan] * len(x))
                return line, fixed_line

            def update(frame):
                if frame >= 100:
                    line.set_ydata(x_infer_list[-1])
                else:
                    line.set_ydata(x_infer_list[frame])
                return line, fixed_line

            ani = FuncAnimation(
                fig, update, init_func=init, frames=150, interval=100, blit=True
            )
            ani.save(f"animation_{args.backbone}.gif", fps=25, writer="imagemagick")
            plt.close(fig)

        def get_psd_gif(x_infer_list, x_ref):
            fs = 2400.0  # Fréquence d'échantillonnage réelle

            # 1. Pré-calculer les PSDs
            freqs_ref, psd_ref = welch(x_ref, fs=fs, nperseg=256)
            psd_list = []
            freqs = None
            for x in x_infer_list:
                f, Pxx = welch(x, fs=fs, nperseg=256)
                psd_list.append(Pxx)
                if freqs is None:
                    freqs = f

            # 2. Création de la figure
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            (line,) = ax.plot(freqs, psd_list[0], color="cornflowerblue", lw=2)
            (fixed_line,) = ax.plot(
                freqs, psd_ref, color="black", lw=2, label="static PSD"
            )

            ax.set_ylim(0, np.max(psd_list) * 1.1)
            ax.set_xlim(0, fs // 2)
            ax.set_xlabel("Fréquence (Hz)")
            ax.set_ylabel("Densité spectrale (PSD)")
            ax.legend()
            ax.set_yscale("log")
            ax.set_ylim(1e-17, np.max(psd_list) * 1.2)

            # 3. Fonctions d'animation
            def init():
                line.set_ydata([np.nan] * len(freqs))
                return line, fixed_line

            def update(frame):
                if frame >= 100:
                    line.set_ydata(psd_list[-2])
                else:
                    line.set_ydata(psd_list[frame])
                return line, fixed_line

            # 4. Animation
            ani = FuncAnimation(
                fig, update, init_func=init, frames=150, interval=100, blit=True
            )

            ani.save(f"animation_psd_{args.backbone}.gif", fps=25, writer="imagemagick")
            plt.close(fig)

        print()
        x_ref = x_1[0].squeeze(1)
        x_gen = x_t[0].squeeze(1)
        get_ts_gif(x_infer_list, x_ref)
        get_psd_gif(x_infer_list, x_ref)

        # print()
        # print("les psd de ref et gen :")
        psd_ref = display_psd(x_ref, name="ref", fs=1200.0, verbose=False)
        psd_gen = display_psd(x_gen, name="gen", fs=1200.0, verbose=False)
        display_double_psd(psd_ref, psd_gen, name_1="ref", name_2="gen")
