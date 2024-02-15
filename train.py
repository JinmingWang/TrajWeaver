# Interpolation Prior Diffusion Model

from Dataset.XianDataset import XianTrajectoryDataset, collectFunc
from Models.TrajWeaver import TrajWeaver
from Models.MultiModalEmbedding import MultimodalEmbedding
from Utils import MovingAverage, saveModel, loadModel
from DiffusionManager import DiffusionManager

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def makeInterpNoise(diff_manager, traj: torch.Tensor, erase_mask: torch.Tensor, t: torch.Tensor):
    """
    :param traj: (B, 3, L) lng, lat, time
    :param erase_mask: (B, 1, L) 1 for erased, 0 for not erased
    :param t: (B)

    usual noise is just a random normal distribution
    but for interpolation, we do have a rough estimation of where the target point is
    for example, to insert a point pi to p1 and p2, with time ti, t1, t2
    the location of pi is likely to be around the linear interpolation of p1 and p2
    denote the linear interpolation of p1 and p2 at time ti as pi'
    then our noise can be a scaled normal distribution with mean at pi' and some feasible std
    """

    B, C, L = traj.shape

    mask = erase_mask.repeat(1, 3, 1).to(torch.bool)
    erased_subtraj = traj[mask].view(B, 3, -1)  # (B, 3, L_erased)
    remain_subtraj = traj[~mask].view(B, 3, -1)  # (B, 3, L_remain)

    L_remain = remain_subtraj.shape[-1]
    L_erased = erased_subtraj.shape[-1]

    time_interp = erased_subtraj[:, 2, :]  # (B, L_erased)
    time_remain = remain_subtraj[:, 2, :]  # (B, L_remain)
    ids_right = torch.stack([torch.searchsorted(time_remain[i], time_interp[i]) for i in range(B)]).to(torch.long).view(
        -1)  # (B * L_erased)
    ids_left = ids_right - 1  # (B * L_erased)

    ids_left = torch.clamp(ids_left, 0, L_remain - 1)
    ids_right = torch.clamp(ids_right, 0, L_remain - 1)

    ids_batch = torch.arange(B).view(B, 1).repeat(1, L_erased).view(-1)  # (B * L_erased

    lng_left = remain_subtraj[ids_batch, torch.zeros_like(ids_batch), ids_left]  # (B * L_erased)
    lng_right = remain_subtraj[ids_batch, torch.zeros_like(ids_batch), ids_right]  # (B * L_erased)
    lat_left = remain_subtraj[ids_batch, torch.ones_like(ids_batch), ids_left]  # (B * L_erased)
    lat_right = remain_subtraj[ids_batch, torch.ones_like(ids_batch), ids_right]  # (B * L_erased)
    time_left = remain_subtraj[ids_batch, torch.ones_like(ids_batch) * 2, ids_left]  # (B * L_erased)
    time_right = remain_subtraj[ids_batch, torch.ones_like(ids_batch) * 2, ids_right]  # (B * L_erased)

    ratio = (time_interp.reshape(-1) - time_left) / (time_right - time_left)  # (B * L_erased)
    lng_interp = lng_left * (1 - ratio) + lng_right * ratio  # (B * L_erased)
    lat_interp = lat_left * (1 - ratio) + lat_right * ratio  # (B * L_erased)

    lnglat_guess = torch.stack([lng_interp, lat_interp], dim=1).view(B, L_erased, 2).transpose(1, 2)  # (B, 2, L_erased)

    interp_lnglat_0 = erased_subtraj[:, :2]
    noise = torch.randn_like(interp_lnglat_0)
    interp_lnglat_t = diff_manager.diffusionForward(interp_lnglat_0, t, noise)
    mask[:, 2] = False
    traj_t = traj.clone()
    traj_t[mask] = interp_lnglat_t.reshape(-1)
    traj_guess = traj.clone()
    traj_guess[mask] = lnglat_guess.reshape(-1)
    return traj_t, traj_guess, mask, noise


def train(
        init_lr: float = 1e-4,
        lr_reduce_factor: float = 0.5,
        lr_reduce_patience: int = 5,
        batch_size: int = 16,
        epochs: int = 60,
        log_interval: int = 40,
        mov_avg_interval: int = 200,
        checkpoint_unet: str = None,
        checkpoint_embedder: str = None,
):
    dataset_args = {
        "dataset_root": "/home/jimmy/Data/Didi/Xian/nov",
        "traj_length": 512,
        "feature_mean": [21599.4980, 108.950773428688, 34.24354179925547],  # time lon lat
        "feature_std": [12470.9102, 0.02129110045580343, 0.019358855648211895],
    }
    dataset = XianTrajectoryDataset(**dataset_args)
    dataset.loadCache("Dataset/Xian_nov_cache.pth")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collectFunc)

    diff_steps = 500

    model = TrajWeaver(
        in_c=6,  # input trajectory encoding channels
        out_c=2,
        down_schedule=[1, 2, 2, 2],  # downsample schedule, first element is step stride
        diffusion_steps=diff_steps,  # maximum diffusion steps
        channel_schedule=[64, 128, 128, 256],  # channel schedule of stages, first element is stem output channels
        n_res_blocks=[2, 2, 2],  # number of resblocks in each stage
        embed_c=64,  # channels of time embeddings
        num_heads=4,  # number of heads for attention
        dropout=0.0,  # dropout
        max_length=4096,  # maximum input encoding length
        self_attn=True
    ).cuda()

    if checkpoint_unet is not None:
        loadModel(model, checkpoint_unet)
    model.train()

    diffusion_args = {
        "min_beta": 0.0001,
        "max_beta": 0.05,
        "max_diffusion_step": diff_steps,
        "scale_mode": "quadratic",
    }

    # --- Prepare ---
    diff_manager = DiffusionManager(**diffusion_args)

    mse_func = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=lr_reduce_factor,
                                  patience=int(lr_reduce_patience),
                                  verbose=True)

    global_it = 0

    mov_avg_loss = MovingAverage(mov_avg_interval)

    log_dir = f"Runs/{model.__class__.__name__}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    writer = SummaryWriter(log_dir)

    with open(log_dir + "info.txt", "w") as file:
        file.write(f"Training {model.__class__.__name__} on Xian dataset\n")
        file.write("Model:\n")
        file.write(str(model))



    sample_lengths = [i for i in range(64, 512)]

    # --- Training ---
    saveModel(model, log_dir + f"{model.__class__.__name__}.pth")
    for e in range(epochs):
        dataloader.dataset.resetSampleLength(random.choice(sample_lengths))
        dataloader.dataset.resetEraseRate(random.uniform(0.2, 0.9))
        pbar = tqdm(dataloader, desc=f'Epoch {e}', ncols=120)
        for traj_0, _, n_erased, erase_mask, num_attr, cat_attr, _ in pbar:
            dataloader.dataset.resetSampleLength(random.choice(sample_lengths))
            dataloader.dataset.resetEraseRate(random.uniform(0.2, 0.9))
            B, _, L = traj_0.shape
            # traj_0: (B, 3, L)

            optimizer.zero_grad()

            # -- Diffusion forward, we use half original trajectory and half equalized trajectory ---
            t = torch.randint(0, diff_steps, (B,)).cuda()

            # traj_t & traj_guess: (B, 3, L)
            # mask: (B, 3, L)
            # epsilon: (B, 2, n_interp)
            traj_t, traj_guess, mask, epsilon = makeInterpNoise(diff_manager, traj_0, erase_mask, t)

            # --- Construct input and predict epsilon ---
            traj_input = torch.cat([traj_t, traj_guess[:, :2, :], erase_mask], dim=1)

            output = model(traj_input, t)  # epsilon_pred: (B, 2, L)
            epsilon_pred = output[mask[:, :2, :]].view(B, 2, -1)  # epsilon_pred: (B, 2, n_erased)

            loss = mse_func(epsilon_pred, epsilon)

            if torch.any(torch.isnan(loss)):
                global_it += 1
                continue

            loss.backward()

            optimizer.step()

            global_it += 1
            mov_avg_loss << loss.item()

            pbar.set_postfix_str(f"loss={float(mov_avg_loss):.5f} | "
                                 f"lr={optimizer.param_groups[0]['lr']:.7f} | "
                                 f"n_erased={int(n_erased[0])}")

            if global_it % log_interval == 0:
                writer.add_scalar("Loss", float(mov_avg_loss), global_it)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_it)

        saveModel(model, log_dir + f"{model.__class__.__name__}.pth")
        print(f"Model saved to {log_dir + model.__class__.__name__}.pth")
        scheduler.step(float(mov_avg_loss))


