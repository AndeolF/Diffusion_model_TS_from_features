import argparse
import os
import torch  # type: ignore
from torch.optim import AdamW, lr_scheduler  # type: ignore
from datafactory.dataloader import loader_provider
from datafactory.dataloader_perso import load_scaled_dataloaders
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from model.denoiser.transformer import Transformer
from model.denoiser.mlp import MLP
import time
from torch.distributions import Beta  # type: ignore
from scipy.signal import firwin
import torch.nn.functional as F  # type: ignore
import numpy as np
from t_giver import TGiverFromError


def delete_other_model(path_floder):
    files = os.listdir(path_floder)
    for file in files:
        if file.endswith(".pth"):
            chemin_fichier = os.path.join(path_floder, file)
            os.remove(chemin_fichier)


def check_previous_val_loss(save_path, current_save_dict):

    loss_values = []

    for filename in os.listdir(save_path):
        if filename.endswith(".pth"):
            filepath = os.path.join(save_path, filename)

            saved_dict = torch.load(filepath)

            loss_values.append(saved_dict["loss_val"][-1])

    if loss_values:
        min_loss = min(loss_values)
        if current_save_dict["loss_val"][-1] < min_loss:
            for filename in os.listdir(save_path):
                if filename.endswith(".pth"):
                    os.remove(os.path.join(save_path, filename))
            return True
        else:
            return False
    else:
        return True


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


def train(args):
    print(
        f"Training config::\tepoch: {args.epochs}\tsave_path: {args.save_path}\tdevice: {args.device}"
    )
    os.makedirs(args.save_path, exist_ok=True)
    #
    # dataset, dataloader = loader_provider(args, period="train")
    train_loader, val_loader, test_loader, feat_scaler, ts_scaler = (
        load_scaled_dataloaders(
            dataset_path="./Data/dataset_50_run_05s_downsample",
            batch_size=args.batch_size,
            scale_features=True,
            scale_series=True,
        )
    )
    print("\n\n")
    #
    model = {"DiT": Transformer, "MLP": MLP}.get(args.denoiser)
    if model:
        model = model().to(args.device)
    else:
        raise ValueError(f"No denoiser found")
    pretrained_model = torch.load(
        args.pretrained_model_path, map_location=torch.device(args.device)
    )
    pretrained_model.float().to(args.device)

    for param in pretrained_model.parameters():
        param.requires_grad = False

    backbone = {
        "flowmatching": RectifiedFlow(args.device),
        "ddpm": DDPM(args.total_step, args.device),
    }.get(args.backbone)
    if backbone:
        pass
    else:
        raise ValueError(f"No backbone found")

    model.encoder = pretrained_model.encoder
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = not args.usepretrainedvae
    print(
        f"Total learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(
        f"VAE learnable parameters: {sum(p.numel() for p in pretrained_model.encoder.parameters() if p.requires_grad)}"
    )
    print(f"Total diffusion parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Total VAE parameters: {sum(p.numel() for p in pretrained_model.parameters())}"
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if args.use__lr_scheduler:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-4, total_steps=len(train_loader) * (args.epochs + 1)
        )
    t_giver = TGiverFromError.load("t_giver/t_giver_epoch15.pt")
    t_giver.device = args.device
    loss_list = []
    loss_val = []
    start_epoch = 0

    f_s = 2400  # Hz, fréquence d’échantillonnage
    f_c = 150  # Hz, fréquence de coupure
    numtaps = 31  # taille du noyau (doit être impair pour symétrie)
    window_choose = "hamming"
    # f_s = 1200  # Hz, fréquence d’échantillonnage
    # f_c = 150  # Hz, fréquence de coupure
    # numtaps = 31  # taille du noyau (doit être impair pour symétrie)
    # Noyau FIR passe-bas
    kernel_np = firwin(numtaps=numtaps, cutoff=f_c, fs=f_s, window=window_choose)
    kernel = torch.tensor(kernel_np, dtype=torch.float32).view(1, 1, -1).to(args.device)

    #############################################
    #               Re use model                #
    #############################################
    if not args.reuse_previous_model == 0:
        number_model_save = 0
        for filename in os.listdir(args.save_path):
            if filename.endswith(".pth"):
                number_model_save += 1
                filepath = os.path.join(args.save_path, filename)

                saved_dict = torch.load(
                    filepath, map_location=torch.device(args.device)
                )
        if number_model_save == 1:
            print("re use model")
            model.load_state_dict(saved_dict["model"])
            optimizer.load_state_dict(saved_dict["optimizer"])
            if args.use__lr_scheduler:
                scheduler.load_state_dict(saved_dict["scheduler"])
            start_epoch = saved_dict["epoch"] + 1
            loss_list = saved_dict["loss_list"]
            loss_val = saved_dict["loss_val"]
        elif number_model_save > 1:
            print("Too many model")
        else:
            print("No model available")
    else:
        print("No model re use")

    #############################################
    #                    Train                  #
    #############################################
    print("Training")
    print(f"Total batch train : {len(train_loader)}")
    print(f"Total batch val : {len(val_loader)}\n")
    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        epoch_val_loss = 0
        mean_loss_big_batch = 0
        compteur_batch_display = 0
        print(loss_val)

        # TRAIN
        model.train()

        for batch, data in enumerate(train_loader):
            train_ts, train_feat = data
            train_feat = train_feat.float().to(args.device)
            x_1_ts = train_ts.float().to(args.device)

            #
            #

            # ecart_type_ref = torch.std(x_1_ts, dim=1)

            #
            #

            # print(f"\nx_1_ts.shape : {x_1_ts.shape}\n")
            x_1_latent, before = model.encoder(
                x_1_ts
            )  # TS data ==>VAE==> clear TS embedding

            if args.backbone == "flowmatching":
                # t = (
                #     torch.round(
                #         torch.rand(x_1_latent.size(0), device=args.device)
                #         * args.total_step
                #     )
                #     / args.total_step
                # )
                if (
                    (epoch) < args.total_curriculum_epochs
                    and args.total_curriculum_epochs != 0
                ):
                    if batch == 0:
                        print("incremental noise")

                    # INCREMENTAL CROISSANT
                    max_t = min(
                        1.0, (epoch + 1) / (args.total_curriculum_epochs + 1)
                    )  # max_t progresse vers 1.0
                    t = torch.rand(x_1_latent.size(0), device=args.device) * max_t
                    t = torch.round(t * args.total_step) / args.total_step

                    # INCREMENTAL DECROISSANT
                    # max_t = 1.0
                    # min_t = 1.0 - min(
                    #     1.0, (epoch + 1) / (args.total_curriculum_epochs + 1)
                    # )
                    # t = min_t + torch.rand(x_1_latent.size(0), device=args.device) * (
                    #     max_t - min_t
                    # )
                    # t = torch.round(t * args.total_step) / args.total_step
                else:
                    if batch == 0:
                        print("regular noise (no curriculum)")

                    # WITH BETA
                    t = Beta(1.0, 0.8).sample([x_1_latent.size(0)]).to(args.device)

                    # WITH UNIFORM
                    # t = (
                    #     torch.round(
                    #         torch.rand(x_1_latent.size(0), device=args.device)
                    #         * args.total_step
                    #     )
                    #     / args.total_step
                    # )

                    # WITH T_GIVER
                    # t = t_giver(x_1_latent.size(0))

                # t = Beta(2, 2).sample([x_1.size(0)]).to(args.device)
                if batch == 0:
                    print(f"t: {torch.mean(t):.4f}")
                    if args.use__lr_scheduler:
                        print(f"lr : {scheduler.get_last_lr()[0]}")
                if args.noise_type == "pink":
                    print("pink")
                    pink_noise = generate_pink_noise_nd(x_1_ts.shape).to(args.device)
                    pink_noise_latent, _ = model.encoder(pink_noise)
                    x_t, x_0 = backbone.create_flow(
                        x_1_latent, t, noise=pink_noise_latent
                    )  # x_t: dirty TS embedding, x_0：pure noise
                elif args.noise_type == "filtred":
                    print("filtred")
                    x_0_ts = torch.randn_like(x_1_ts).to(x_1_ts.device)
                    x_0_ts = x_0_ts.unsqueeze(1)
                    x_0_filtred_ts = F.conv1d(x_0_ts, kernel, padding=numtaps // 2)
                    x_0_filtred_ts = x_0_filtred_ts.squeeze(1)
                    x_0_filtred_latent, _ = model.encoder(x_0_filtred_ts)
                    x_t, x_0 = backbone.create_flow(
                        x_1_latent, t, noise=x_0_filtred_latent
                    )  # x_t: dirty TS embedding, x_0：pure noise
                else:
                    x_t, x_0, x_t_next_ref = backbone.create_flow(
                        x_1_latent, t, dt=1.0 / args.total_step
                    )  # x_t: dirty TS embedding, x_0：pure noise
                noise_gt = x_1_latent - x_0
            elif args.backbone == "ddpm":
                t = torch.floor(
                    torch.rand(x_1_latent.size(0)).to(args.device) * args.total_step
                ).long()
                noise_gt = torch.randn_like(x_1_latent).float().to(args.device)
                x_t, n_xt = backbone.q_sample(x_1_latent, t, noise_gt)
            else:
                raise ValueError(f"Unsupported backbone type: {args.backbone}")

            optimizer.zero_grad()
            decide = torch.rand(1) < 0.3  #  for classifier free guidance
            if decide:
                train_feat = None

            pred = model(input=x_t, t=t, text_input=train_feat)

            # # Loss sur le signal reconstruit :
            # x_t_next = backbone.euler(x_t, pred, 1.0 / args.total_step)
            # x_t_next_decode, _ = pretrained_model.decoder(
            #     x_t_next, length=x_1_ts.shape[-1]
            # )
            # x_t_next_ref_decode, _ = pretrained_model.decoder(
            #     x_t_next_ref, length=x_1_ts.shape[-1]
            # )
            # ecart_type_gen = torch.std(x_t_next_decode, dim=1)

            #

            loss = backbone.loss(
                v=pred,
                noise_gt=noise_gt,
                # x_t_next_decode=x_t_next_decode,
                # x_t_next=x_t_next,
                # x_t_next_ref=x_t_next_ref,
                # ecart_type_ref=ecart_type_ref,
                # ecart_type_gen=ecart_type_gen,
                t=t,
                # numtaps=numtaps,
                # kernel=kernel,
            )
            loss.backward()

            loss_list.append(loss.item())
            mean_loss_big_batch += loss
            compteur_batch_display += 1
            optimizer.step()
            if batch % 8000 == 0:
                if batch != 0:
                    mean_loss_big_batch = mean_loss_big_batch / compteur_batch_display
                    print(
                        f"[Epoch {epoch}] [batch {batch}] loss: {mean_loss_big_batch}"
                    )
                    mean_loss_big_batch = 0
                    compteur_batch_display = 0

            if args.use__lr_scheduler:
                scheduler.step()

        #############################################
        #               Validation                  #
        #############################################
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                for batch, data in enumerate(val_loader):
                    val_ts, val_feat = data
                    val_feat = val_feat.float().to(args.device)
                    x_1_ts = val_ts.float().to(args.device)

                    #

                    # ecart_type_ref = torch.std(x_1_ts, dim=1)

                    #

                    x_1_latent, before = model.encoder(
                        x_1_ts
                    )  # TS data ==>VAE==> clear TS embedding

                    if args.backbone == "flowmatching":
                        t = (
                            torch.round(
                                torch.rand(x_1_latent.size(0), device=args.device)
                                * args.total_step
                            )
                            / args.total_step
                        )
                        if args.noise_type == "pink":
                            pink_noise = generate_pink_noise_nd(x_1_ts.shape).to(
                                args.device
                            )
                            pink_noise_latent, _ = model.encoder(pink_noise)
                            x_t, x_0 = backbone.create_flow(
                                x_1_latent, t, noise=pink_noise_latent
                            )  # x_t: dirty TS embedding, x_0：pure noise
                        elif args.noise_type == "filtred":
                            x_0_ts = torch.randn_like(x_1_ts).to(x_1_ts.device)
                            x_0_ts = x_0_ts.unsqueeze(1)
                            x_0_filtred_ts = F.conv1d(
                                x_0_ts, kernel, padding=numtaps // 2
                            )
                            x_0_filtred_ts = x_0_filtred_ts.squeeze(1)
                            x_0_filtred_latent, _ = model.encoder(x_0_filtred_ts)
                            x_t, x_0 = backbone.create_flow(
                                x_1_latent, t, noise=x_0_filtred_latent
                            )  # x_t: dirty TS embedding, x_0：pure noise
                        else:
                            x_t, x_0, x_t_next_ref = backbone.create_flow(
                                x_1_latent, t, dt=1.0 / args.total_step
                            )  # x_t: dirty TS embedding, x_0：pure noise
                        noise_gt = x_1_latent - x_0
                    elif args.backbone == "ddpm":
                        t = torch.floor(
                            torch.rand(x_1_latent.size(0)).to(args.device)
                            * args.total_step
                        ).long()
                        noise_gt = torch.randn_like(x_1_latent).float().to(args.device)
                        x_t, n_xt = backbone.q_sample(x_1_latent, t, noise_gt)
                    else:
                        raise ValueError(f"Unsupported backbone type: {args.backbone}")

                    pred = model(input=x_t, t=t, text_input=val_feat)

                    #

                    # x_t_next = backbone.euler(x_t, pred, 1.0 / args.total_step)
                    # x_t_next_decode, _ = pretrained_model.decoder(
                    #     x_t_next, length=x_1_ts.shape[-1]
                    # )
                    # ecart_type_gen = torch.std(x_t_next_decode, dim=1)

                    #

                    v_loss = backbone.loss(
                        v=pred,
                        noise_gt=noise_gt,
                        # x_t_next_decode=x_t_next_decode,
                        # x_t_next=x_t_next,
                        # x_t_next_ref=x_t_next_ref,
                        # ecart_type_ref=ecart_type_ref,
                        # ecart_type_gen=ecart_type_gen,
                        t=t,
                        # numtaps=numtaps,
                        # kernel=kernel,
                    )

                    epoch_val_loss += v_loss

                epoch_val_loss = epoch_val_loss / len(val_loader)

                loss_val.append(epoch_val_loss)
                print(f"VALIDATION : [Epoch {epoch}]  val_loss: {epoch_val_loss}")

                # Because of the fact that the noise is usual in validation and not in trainnig (cause of the noise schedular)
                # so we have to wait til all the validation loss that we wil look are due to the model after his 'noise load' time

                #############################################
                #              Save and Stop                #
                #############################################
                # SAVE
                if epoch_val_loss == min(loss_val):
                    # if True:
                    os.makedirs(args.save_path, exist_ok=True)

                    save_dict = dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        # scheduler=scheduler.state_dict(),
                        epoch=epoch,
                        loss_list=loss_list,
                        loss_val=loss_val,
                    )
                    bool_save = check_previous_val_loss(args.save_path, save_dict)
                    if bool_save:
                        # if True:
                        torch.save(
                            save_dict,
                            os.path.join(args.save_path, f"model_{epoch}.pth"),
                        )
                        print(f"Saved Model from epoch: {epoch}\n")

                # CHECK CALLBACK
                epsilon = 1e-9
                if (epoch + 1) > args.total_curriculum_epochs + args.time_checkback:
                    if len(loss_val) > args.time_checkback:
                        list_checkback = []
                        for i in range(args.time_checkback + 1):
                            list_checkback.append(loss_val[-i - 1])
                        if min(list_checkback) + epsilon > list_checkback[-1]:
                            print("callback reach")
                            return


def get_args():

    parser = argparse.ArgumentParser(description="Train Features2S model")

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
    parser.add_argument(
        "--use__lr_scheduler",
        type=int,
        default=0,
        help="0 no use, other use",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument(
        "--time_checkback", type=int, default=10, help="for_the_call_back"
    )
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument(
        "--noise_type",
        type=str,
        default="white",
        help="color of the noise for sampling",
    )

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
    parser.add_argument(
        "--total_curriculum_epochs",
        type=int,
        default=5,
        help="progressive noise",
    )
    parser.add_argument("--denoiser", type=str, default="DiT", help="DiT or MLP")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.pretrained_model_path = (
        f"checkpoint/dataset_50_run_05s_downsample/final_VAEpretrain_model.pth"
    )

    args.checkpoint_path = os.path.join(
        args.checkpoint_path,
        "{}".format(args.dataset_name),
    )
    args.save_path = os.path.join(
        args.checkpoint_path,
        "{}_{}".format(args.backbone, args.denoiser),
    )
    return args


if __name__ == "__main__":
    args = get_args()
    stime = time.time()
    train(args)
    etime = time.time()
    print(etime - stime)
