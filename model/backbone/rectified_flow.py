import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import matplotlib.pyplot as plt
import numpy as np


class RectifiedFlow:
    def __init__(self, device):
        std_ref_np = np.load(
            "Data/dataset_50_run_05s_downsample/std_latent_data_train_encode.npy"
        )
        self.std_ref = torch.from_numpy(std_ref_np).to(device)

    def euler(self, x_t, v, dt):
        x_t = x_t + v * dt
        return x_t

    def heun(self, model, x_t, t, t_next, feat, cfg_scale, dt):
        # Step 1 — première prédiction (à t)
        v_uncond_1 = model(input=x_t, t=t, text_input=None)
        v_cond_1 = model(input=x_t, t=t, text_input=feat)
        v1 = v_uncond_1 + cfg_scale * (v_cond_1 - v_uncond_1)

        # Estimation d'Euler
        x_euler = x_t + dt * v1

        # Step 2 — seconde prédiction (à t_next)
        v_uncond_2 = model(input=x_euler, t=t_next, text_input=None)
        v_cond_2 = model(input=x_euler, t=t_next, text_input=feat)
        v2 = v_uncond_2 + cfg_scale * (v_cond_2 - v_uncond_2)

        # Heun step final
        x_next = x_t + 0.5 * dt * (v1 + v2)
        return x_next

    def create_flow(self, x_1, t, dt=0, noise=None):
        if noise == None:
            x_0 = torch.randn_like(x_1).to(x_1.device)
            x_0 = (x_0 / x_0.std()) * self.std_ref
            t = t[:, None, None]  # [B, 1, 1, 1]
            x_t = t * x_1 + (1 - t) * x_0
            x_t_next = (t + dt) * x_1 + (1 - t - dt) * x_0
            return x_t, x_0, x_t_next
        else:
            x_0 = noise
            t = t[:, None, None]  # [B, 1, 1, 1]
            x_t = t * x_1 + (1 - t) * x_0
            return x_t, x_0

    def loss(
        self,
        v,
        noise_gt,
        v_plus=0,
        v_minus=0,
        x_t_next_decode=0,
        x_t_next=0,
        x_t_next_ref=0,
        ecart_type_ref=0,
        ecart_type_gen=0,
        t=0,
        λ_ecart=0.05,
        λ_norm=0.05,
        λ_cos=1.0,
        λ_curv=0.12,
        λ_smooth=0.01,
        kernel=None,
        numtaps=0,
        train=True,
    ):
        # LOSS 3 VITESSES
        # if train:
        #     # noise_gt : x_1 - x_0
        # loss_mse = F.mse_loss(v, noise_gt)

        #     # Vector norm regularization
        #     pred_norm = v.norm(dim=1)
        #     target_norm = noise_gt.norm(dim=1)
        #     loss_norm = F.mse_loss(pred_norm, target_norm)

        #     # Direction vector regularization
        #     cos_sim = F.cosine_similarity(v, noise_gt, dim=1)  # ∈ [-1, 1]
        #     loss_cos = 1 - cos_sim.mean()

        #     loss_curvature = F.mse_loss(v_plus + v_minus, 2 * v)

        #     return loss_mse + λ_var * loss_norm + λ_cos * loss_cos + λ_curv * loss_curvature
        # else:
        #     return F.mse_loss(v, noise_gt)

        # LOSS SUR LA NORM DE VITESSE
        if train:
            # noise_gt : x_1 - x_0
            loss_mse = F.mse_loss(v, noise_gt)

            #
            #

            # loss_mse_vector = F.mse_loss(x_t_next, x_t_next_ref)

            #
            #

            #
            #

            # Loss sur les ecart type
            # loss_ecart = F.mse_loss(ecart_type_ref, ecart_type_gen)

            #
            #

            # # Vector norm regularization
            # pred_norm = v.norm(dim=0)
            # target_norm = noise_gt.norm(dim=0)

            # Calculer les normes de Frobenius pour chaque matrice dans le batch
            # pred_norm = torch.norm(v, p="fro", dim=(1, 2))
            # target_norm = torch.norm(noise_gt, p="fro", dim=(1, 2))

            # loss_norm = F.mse_loss(pred_norm, target_norm)

            #
            #

            # Direction vector regularization
            # cos_sim = F.cosine_similarity(v, noise_gt, dim=1)  # ∈ [-1, 1]
            # loss_cos = 1 - cos_sim.mean()

            #
            #

            # eviter d'avoir de trop grosse variation entre deux pas
            # norms = v.norm(dim=1)
            # loss_curv = norms.var()

            #
            #

            # LOSS SMOOTH WITH 1st DER
            # tenter de lisser le signal avec l'approx première
            # dx = x_t_next_decode[:, 1:] - x_t_next_decode[:, :-1]
            # loss_smooth = dx.pow(2).mean()

            # losses_smooth_per_series = dx.pow(2).mean(dim=1)
            # mask = (t > 0.6).float()  # Seuil de lissage tardif
            # loss_smooth = (losses_smooth_per_series * mask).sum() / (
            #     mask.sum() + 1e-8
            # )  # Appliquer la régularisation uniquement aux séries où tᵢ > t_ref

            #
            #

            # # LOSS SMOOTH
            # x_t_next_decode = x_t_next_decode.unsqueeze(1)
            # smoothed = F.conv1d(x_t_next_decode, kernel, padding=numtaps // 2)

            # loss_smooth = F.mse_loss(x_t_next_decode, smoothed)

            # # LOSS SMOOTH WITH FILTRE
            # if x_t_next_decode.dim() == 3:
            #     x_t_next_decode = x_t_next_decode.squeeze(1)
            #     smoothed = smoothed.squeeze(1)

            # # MSE par série : moyenne sur la dimension temporelle (L)
            # losses_smooth_per_series = ((x_t_next_decode - smoothed) ** 2).mean(dim=1)
            # mask = (t > 0.0).float()  # Seuil de lissage tardif
            # loss_smooth = (losses_smooth_per_series * mask).sum() / (
            #     mask.sum() + 1e-10
            # )  # Appliquer la régularisation uniquement aux séries où tᵢ > t_ref

            #
            #

            # tenter de lisser le signal avec l'approx seconde
            # x_prev = x_t_next_decode[:, :-2]
            # x_curr = x_t_next_decode[:, 1:-1]
            # x_next = x_t_next_decode[:, 2:]
            # d2x = x_next - 2 * x_curr + x_prev
            # loss_smooth = d2x.pow(2).mean()

            return (
                loss_mse
                # loss_mse_vector
                # + λ_ecart * loss_ecart
                # + λ_norm * loss_norm
                # + λ_cos * loss_cos
                # + λ_curv * loss_curv
                # + λ_smooth * loss_smooth
            )
        else:
            return F.mse_loss(v, noise_gt)

    # def loss(self, v, noise_gt, λ_var=0.1):
    #     # noise_gt : x_1 - x_0
    #     loss_mse = F.mse_loss(v, noise_gt)

    #     # Vector norm regularization
    #     pred_norm = v.norm(dim=1)
    #     target_norm = noise_gt.norm(dim=1)
    #     loss_norm = F.mse_loss(pred_norm, target_norm)

    #     return loss_mse + λ_var * loss_norm


if __name__ == "__main__":
    rf = RectifiedFlow()
    t = torch.tensor([0.999])
    x_t = rf.create_flow(
        torch.ones(
            1,
            24,
            1,
        ).float(),
        t,
    )
    plt.plot(x_t[0].detach().cpu().numpy().squeeze())
    plt.plot(x_t[1].detach().cpu().numpy().squeeze())
    plt.show()

    print(x_t)
