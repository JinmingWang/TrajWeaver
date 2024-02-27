# Interpolation Prior Diffusion Model

from Dataset.XianDataset import XianTrajectoryDataset, collectFunc
from Models.TrajWeaver import TrajWeaver
from Models.MultiModalEmbedding import MultimodalEmbedding
from Utils import MovingAverage, saveModel, loadModel, makeInterpNoise
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


def train(
        init_lr: float =            1e-4,
        lr_reduce_factor: float =   0.5,
        lr_reduce_patience: int =   5,
        batch_size: int =           2,
        epochs: int =               60,
        log_interval: int =         40,
        mov_avg_interval: int =     200,
        checkpoint_unet: str =      None,
):

    # IMPORTANT: This implementation does not use the multimodal embedding model

    dataset_args = {
        "dataset_root": "nov",  # assume nov dataset is already cached
        "traj_length": 512,
        "feature_mean": [21599.4980, 108.950773428688, 34.24354179925547],  # time lon lat
        "feature_std": [12470.9102, 0.02129110045580343, 0.019358855648211895],
    }
    dataset = XianTrajectoryDataset(**dataset_args)
    dataset.loadCache("Dataset/Xian_nov_cache.pth")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collectFunc)

    diff_steps = 500

    model = TrajWeaver(
        in_c=6,  # input trajectory + encoding channels
        out_c=2, # output trajectory channels
        down_schedule=[1, 2, 2, 2],  # downsample schedule, first element is step stride
        diffusion_steps=diff_steps,  # maximum diffusion steps
        channel_schedule=[64, 128, 128, 256],  # channel schedule of stages, first element is stem output channels
        n_res_blocks=[2, 2, 2],  # number of resblocks in each stage
        embed_c=64,  # channels of time embeddings
        num_heads=4,  # number of heads for attention
        dropout=0.0,  # dropout
        max_length=4096,  # maximum input encoding length
        self_attn=True  # True then use self-attention, False then only use channel attention with conv
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

    # all possible trajectory lengths
    sample_lengths = [i for i in range(64, 512)]

    # --- Training ---
    saveModel(model, log_dir + f"{model.__class__.__name__}.pth")
    for e in range(epochs):
        dataloader.dataset.resetSampleLength(random.choice(sample_lengths))
        dataloader.dataset.resetEraseRate(random.uniform(0.2, 0.9))
        pbar = tqdm(dataloader, desc=f'Epoch {e}', ncols=120)
        for traj_0, _, n_erased, erase_mask, num_attr, cat_attr, _ in pbar:
            # randomize sample length and erase rate
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


if __name__ == '__main__':
    train()


