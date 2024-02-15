import torch
import torch.nn as nn
from typing import *
import ctypes
import numpy as np
import matplotlib.pyplot as plt

from Dataset.XianDataset import XianTrajectoryDataset, collectFunc
from torch.utils.data import DataLoader

cur_dir = __file__[0:__file__.rfind("/")+1]
eval_c_lib = ctypes.cdll.LoadLibrary(f"{cur_dir}loops_in_cpp.so")
eval_c_lib.NDTW.restype = ctypes.c_float

def XianDataGetter(batch_size: int, traj_length: int, erase_rate: float):
    """
    Get a batch of data from DidiDataset
    :param batch_size:
    :param traj_length: the length of the trajectory
    :param erase_rate: the erase rate of the trajectory
    :param city: the city of the dataset
    :return:
    """
    dataset_args = {
        "dataset_root": "nov",
        "traj_length": 512,
        "feature_mean": [21599.4980, 108.950773428688, 34.24354179925547],  # time lon lat
        "feature_std": [12470.9102, 0.02129110045580343, 0.019358855648211895],
    }
    dataset = XianTrajectoryDataset(**dataset_args)
    dataset.loadCache("Dataset/Xian_512_cache.pth")

    dataset.resetSampleLength(traj_length)
    dataset.resetEraseRate(erase_rate)

    # --- Prepare ---
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collectFunc)

    uids = torch.zeros(batch_size, dtype=torch.long).cuda()

    for data_tuple in dataloader:
        traj_0, traj_eq, n_erased, erase_mask, num_attrs, cat_attrs, insertion_nums,  = data_tuple
        yield traj_0, erase_mask, uids



def getInferFunction(model: nn.Module,
                     traj_0: torch.Tensor,
                     mask: torch.Tensor,
                     interp_mask: torch.Tensor,
                     global_embed: torch.Tensor,
                     traj_guess: torch.Tensor
                     ):
    """
    This function returns a function that can be used to infer the noise of a trajectory
    :param model: The diffusion model
    :param traj_0: The original trajectory (B, 3, L)
    :param mask: The mask of the original trajectory (B, 3, L)
    :param interp_mask: The mask of the interpolated trajectory (B, 1, L)
    :return:
    """

    B = traj_0.shape[0]

    lnglat_guess = traj_guess[:, :2, :]

    def infer(interp_lnglat_t: torch.Tensor, t: torch.Tensor):
        # interp_lnglat_t: (B, 2, n_interp)
        # t: (B,)
        traj_t = traj_0.clone()  # (B, 3, L)
        # print(traj_t[mask].shape, interp_lnglat_t.shape)
        traj_t[mask] = interp_lnglat_t.reshape(-1)  # fill in the generated / interpolated points
        traj_input = torch.cat([traj_t, lnglat_guess, interp_mask, global_embed], dim=1)  # construct the input
        output = model(traj_input, t)
        epsilon_pred = output[mask[:, :2]].view(B, 2, -1)  # extract the predicted part / interpolated part
        return epsilon_pred

    return infer



def getUpsampleInput(traj_0: torch.Tensor, scale: int):
    """
    Generate the upsampled trajectory and the corresponding mask that are used to perform diffusion
    :param traj_0: The original trajectory (B, 3, L)
    :param scale: The scale of upsampling
    :return: traj_up (B, 3, scale * L), interp_mask_up (B, 1, scale * L)
    """
    B, _, L = traj_0.shape
    n_remain = L
    n_interp = L * (scale - 1)

    interp_mask_up = torch.cat([torch.zeros(B, 1, n_remain), torch.ones(B, 1, n_interp)], dim=0)
    remain_subtraj = traj_0
    interp_subtraj = torch.zeros(B, 3, n_interp).cuda()

    # the times of the interpolated points should distribute evenly during the time of the original trajectory
    min_times = remain_subtraj[:, 2, :].min(dim=1, keepdim=True)  # (B, 1)
    max_times = remain_subtraj[:, 2, :].max(dim=1, keepdim=True)  # (B, 1)
    times_ranges = [torch.linspace(min_times[b], max_times[b], L * 7 + 2)[1:-1] for b in range(B)]  # B * (n_interp,)
    interp_subtraj[:, 2, :] = torch.stack(times_ranges, dim=0)  # (B, n_interp)

    # merge remain_subtraj and interp_subtraj according to time order
    traj_up = torch.cat([remain_subtraj, interp_subtraj], dim=2)  # (B, 3, L + n_interp)
    for b in range(B):
        sort_idx = torch.argsort(traj_up[b, 2])
        interp_mask_up[b] = interp_mask_up[b][:, sort_idx]
        traj_up[b] = traj_up[b][:, sort_idx]

    return traj_up, interp_mask_up


def JSD(P: torch.Tensor, Q: torch.Tensor) -> float:
    """ Compute the Jensen-Shannon divergence between two distributions.

    :param P: Batch of original distributions. (B, ...)
    :param Q: Batch of generated distributions. (B, ...)
    :return: JSD score
    """
    # Compute KL divergence between P and the average distribution of P and Q
    P_avg = 0.5 * (P + Q)
    kl_divergence_P = torch.nn.functional.kl_div(P.log(), P_avg, reduction='batchmean')

    # Compute KL divergence between Q and the average distribution of P and Q
    Q_avg = 0.5 * (P + Q)
    kl_divergence_Q = torch.nn.functional.kl_div(Q.log(), Q_avg, reduction='batchmean')

    # Compute Jensen-Shannon Divergence
    jsd_score = 0.5 * (kl_divergence_P + kl_divergence_Q)

    return jsd_score.item()


def getPointDistribution(traj_dataset: torch.Tensor, n_grids: int = 16, normalize: bool = True):
    # traj_dataset: (B, 2, L)
    # Get the distribution of all points in the dataset
    # return: (B, 2, L)

    traj_dataset = traj_dataset.transpose(1, 2).reshape(-1, 2).contiguous()  # (B*L, 2)

    # STEP 1. get min and max
    min_lon = traj_dataset[:, 0].min().item()
    max_lon = traj_dataset[:, 0].max().item()
    min_lat = traj_dataset[:, 1].min().item()
    max_lat = traj_dataset[:, 1].max().item()

    # STEP 2. split city into 16x16 grid
    lng_interval = (max_lon - min_lon) / n_grids
    lat_interval = (max_lat - min_lat) / n_grids

    # STEP 3. count points in each grid
    point_count = torch.zeros((n_grids, n_grids), device=traj_dataset.device)
    lng_indices = torch.clip((traj_dataset[:, 0] - min_lon) // lng_interval, 0, n_grids - 1).to(torch.long)
    lat_indices = torch.clip((traj_dataset[:, 1] - min_lat) // lat_interval, 0, n_grids - 1).to(torch.long)

    eval_c_lib.accumulateCount.restype = ctypes.POINTER(ctypes.c_float * n_grids * n_grids)
    lng_indices = lng_indices.cpu().numpy().astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    lat_indices = lat_indices.cpu().numpy().astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # The C function eval_c_lib.accumulateCount does this:
    # point_count = np.zeros((n_grids, n_grids))
    # for r, c in zip(lng_indices, lat_indices):
    #     point_count[r, c] += 1
    # return point_count
    # Python is insanely slow, so we use C to do this
    point_count_ptr = eval_c_lib.accumulateCount(lng_indices, lat_indices, n_grids, traj_dataset.shape[0])

    point_count = torch.tensor(np.frombuffer(point_count_ptr.contents, dtype=np.int32).reshape(n_grids, n_grids)).cuda().to(torch.float32)

    # STEP 4. normalize
    if normalize:
        point_count = (point_count + 1) / point_count.sum()
    return point_count


def NDTW(target_traj, compare_traj):
    """
    This function calculates the Dynamic Time Warping (DTW) distance between two trajectories.
    :param target_traj: trajectory 1 (3, N)
    :param compare_traj: trajectory 2 (3, M)
    :return: DTW distance
    """
    n = target_traj.shape[1]
    m = compare_traj.shape[1]
    dtw = torch.zeros((n + 1, m + 1))
    dtw[1:, 0] = torch.inf
    dtw[0, 1:] = torch.inf
    dtw[0, 0] = 0

    lng_lat_A = target_traj[:2, :].unsqueeze(2)
    lng_lat_B = compare_traj[:2, :].unsqueeze(1)
    squared_dist = torch.sum((lng_lat_A - lng_lat_B) ** 2, dim=0)
    dist_mat = torch.sqrt(squared_dist)

    dist_mat_ptr = dist_mat.cpu().numpy().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dtw_ptr = dtw.cpu().numpy().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    return eval_c_lib.NDTW(dist_mat_ptr, dtw_ptr, n, m)


def plotResults(means: Dict[str, np.ndarray], stds: Dict[str, np.ndarray]):
    """
    Plot the results
    :param means: The means of the results
    :param stds: The stds of the results
    :return:
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (name, mean) in enumerate(means.items()):
        ax = axes[i]
        ax.set_title(name)
        ax.set_xlabel("Erase Rate")
        ax.set_ylabel(name)
        ax.grid(True, linewidth=1, alpha=0.1)
        ax.errorbar([0.3, 0.5, 0.7, 0.9, 0.95], mean, stds[name], linestyle='dotted', marker='.', elinewidth=1, capsize=3)
    plt.tight_layout()
    plt.show()
