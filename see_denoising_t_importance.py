import torch  # type: ignore
import numpy as np
import math
import random
from datafactory.dataloader_perso import load_scaled_dataloaders
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from model.denoiser.transformer import Transformer
from model.denoiser.mlp import MLP
import torch.nn.functional as F  # type: ignore
import matplotlib.pyplot as plt
from t_giver import TGiverFromError
from tqdm import tqdm


if True:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = 100
    cfg_scale = 10
    num_of_batch = 20

    train_loader, val_loader, test_loader, feat_scaler, ts_scaler = (
        load_scaled_dataloaders(
            dataset_path="./Data/dataset_20_run_05s_windows_clean",
            batch_size=32,
            scale_features=True,
            scale_series=True,
        )
    )

    std_ref = np.load(
        "Data/dataset_20_run_05s_windows_clean/std_latent_data_train_encode.npy"
    )
    std_ref = torch.from_numpy(std_ref).to(device)

    dataloader = val_loader

    pretrained_model = torch.load(
        "checkpoint/dataset_20_run_05s_windows/final_VAEpretrain_model.pth",
        map_location=torch.device(device),
    )
    pretrained_model.float().to(device).eval()

    backbone = RectifiedFlow(device)

    model = Transformer
    model = model().to(device)
    model.encoder = pretrained_model.encoder
    model.load_state_dict(
        torch.load(
            "checkpoint/dataset_20_run_05s_windows/flowmatching_DiT/model_29.pth"
        )["model"]
    )
    model.to(device).eval()

    for param in pretrained_model.parameters():
        param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False

    tab_vt_intermediaire_gen = []
    tab_v_intermediaire_ref = []

    with torch.no_grad():
        for batch, data in enumerate(
            tqdm(dataloader, total=min(num_of_batch, len(dataloader)), ncols=100)
        ):
            tab_vt_intermediaire_gen_batch = []
            tab_v_intermediaire_ref_batch = []
            # if batch % 20 == 0:
            #     print(f"Generating {batch}th Batch TS...")

            infer_ts, feat = data
            x_1 = infer_ts.float().to(device)
            feat = feat.float().to(device)

            x_1_latent, before = model.encoder(x_1)
            x_1_latent_copy = x_1_latent.clone()
            x_t_latent = torch.randn_like(x_1_latent).float().to(device)
            x_t_latent = (x_t_latent / x_t_latent.std()) * std_ref
            bruit_initial = x_t_latent.clone()

            for j in range(step):
                velocity_ref = x_1_latent_copy - bruit_initial
                t = (
                    torch.round(
                        torch.full(
                            (x_t_latent.shape[0],), j * 1.0 / step, device=device
                        )
                        * step
                    )
                    / step
                )
                pred_uncond = model(input=x_t_latent, t=t, text_input=None)
                pred_cond = model(input=x_t_latent, t=t, text_input=feat)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                x_t_latent = backbone.euler(x_t_latent, pred, 1.0 / step)

                velocity_ref = velocity_ref.detach().cpu().numpy().squeeze()
                pred = pred.detach().cpu().numpy().squeeze()
                tab_v_intermediaire_ref_batch.append(velocity_ref)
                tab_vt_intermediaire_gen_batch.append(pred)

            x_t, after = pretrained_model.decoder(x_t_latent, length=x_1.shape[-1])

            tab_vt_intermediaire_gen.append(tab_vt_intermediaire_gen_batch)
            tab_v_intermediaire_ref.append(tab_v_intermediaire_ref_batch)

            if batch == num_of_batch:
                break

    tab_vt_intermediaire_gen = np.array(tab_vt_intermediaire_gen)
    tab_v_intermediaire_ref = np.array(tab_v_intermediaire_ref)

    print(tab_vt_intermediaire_gen.shape)
    print(tab_v_intermediaire_ref.shape)

    ############################################################
    #    Erreur MSE moyenne par step (donc par valeur de t)    #
    ############################################################
    # On calcule la MSE par step t
    num_steps = tab_vt_intermediaire_gen.shape[1]
    mse_per_step = []

    for step in range(num_steps):
        pred_for_t = torch.from_numpy(
            tab_vt_intermediaire_gen[:, step]
        )  # (num_batches, batch_size, 64, 30)
        ref_for_t = torch.from_numpy(
            tab_v_intermediaire_ref[:, step]
        )  # (num_batches, batch_size, 64, 30)
        mse = F.mse_loss(
            pred_for_t, ref_for_t, reduction="none"
        )  # shape: (num_batches, batch_size, 64, 30)
        mse = mse.mean(dim=[0, 1, 2, 3])  # moyenne sur batchs et spatial
        mse_per_step.append(mse.item())

    # Optionnel : normaliser t entre 0 et 1
    t_values = torch.linspace(0, 1, steps=num_steps)

    # Plot
    plt.plot(t_values, mse_per_step)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.title("Erreur MSE moyenne par t")
    plt.grid(True)
    plt.show()

    if False:
        t_giver = TGiverFromError(t_values, mse_per_step, alpha=1.0)
        t_giver.save("t_giver/t_giver_epoch15.pt")


############################################
#              Test t_giver                #
############################################
if False:
    # Charger t_giver
    t_giver = TGiverFromError.load("t_giver/t_giver_epoch15.pt")

    # Définir device
    device = "cuda"  # ou "cuda" si tu veux tester sur GPU
    t_giver.device = device

    # Générer des échantillons
    N = 100000
    samples = t_giver(N).cpu().numpy()

    # Histogramme des échantillons
    num_bins = len(t_giver.t_vals)
    hist, bin_edges = np.histogram(samples, bins=num_bins, range=(0, 1), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Courbe théorique (la proba de chaque bin)
    target_distribution = 100 * t_giver.probs.cpu().numpy()

    # Affichage
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_giver.t_vals.numpy(),
        target_distribution,
        label="Distribution cible",
        linewidth=2,
    )
    plt.plot(bin_centers, hist, label="Distribution échantillonnée", linestyle="--")
    plt.xlabel("t")
    plt.ylabel("Densité")
    plt.title("Vérification du t_giver : échantillons vs cible")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


############################################
#   Norme moyenne du gradient par step t   #
############################################

if False:
    num_steps = tab_vt_intermediaire_gen.shape[1]
    gradient_norms_per_step = []

    for step in range(num_steps):
        pred = (
            torch.from_numpy(tab_vt_intermediaire_gen[:, step])
            .detach()
            .clone()
            .requires_grad_(True)
        )
        ref = torch.from_numpy(tab_v_intermediaire_ref[:, step])

        loss = F.mse_loss(pred, ref)
        loss.backward()

        # Norme du gradient
        grad = pred.grad  # shape: (num_batches, batch_size, 64, 30)
        grad_norm = grad.pow(2).sum(dim=[-1, -2]).sqrt()  # L2 par vecteur
        grad_norm = grad_norm.mean().item()  # moyenne sur tout
        gradient_norms_per_step.append(grad_norm)

    # Optionnel : normaliser t entre 0 et 1
    t_values = torch.linspace(0, 1, steps=num_steps)

    # Plot
    plt.plot(t_values, gradient_norms_per_step)
    plt.xlabel("t")
    plt.ylabel("Norme moyenne du gradient")
    plt.title("Force du gradient de la loss par t")
    plt.grid(True)
    plt.show()


############################################
#           Seeing beta function           #
############################################
if False:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    # Beta parameters
    alpha = 1.0
    beta_param = 0.95

    x = np.linspace(0, 1, 100)
    pdf = beta.pdf(x, alpha, beta_param)

    plt.plot(x, pdf, label=f"Beta({alpha}, {beta_param})")
    plt.xlabel("t")
    plt.ylabel("Densité")
    plt.title("Loi de probabilité pour échantillonnage de t")
    plt.grid(True)
    plt.legend()
    plt.show()


############################################
#          Seeing power function           #
############################################
if False:
    for gamma in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        t_sched = 1 - torch.linspace(0, 1, 100) ** gamma
        plt.plot(t_sched.cpu(), label=f"γ={gamma}")

    plt.xlabel("Step index")
    plt.ylabel("t")
    plt.title("Schedules non-uniformes de t")
    plt.grid(True)
    plt.legend()
    plt.show()
