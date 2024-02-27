import matplotlib.pyplot as plt
import numpy as np
import torch


def saveModel(model:torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def loadModel(model:torch.nn.Module, path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(path))
    return model


def copyModel(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict())


def exportONNX(model: torch.nn.Module, sample_inputs: list[torch.Tensor], path: str) -> None:
    model.eval()
    torch.onnx.export(model, sample_inputs, path, verbose=True, opset_version=13, 
                      input_names=['input_traj', "time", "attr"], output_names=['output_traj'])
    

def visualizeTraj(traj: torch.Tensor) -> None:
    """ draw trajectory

    :param traj: (3, N)
    :return: None
    """
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.plot(traj[0, :].cpu(), traj[1, :].cpu(), color='#101010', linewidth=0.1)
    plt.scatter(traj[0, :].cpu(), traj[1, :].cpu(), c=traj[2, :].cpu(), cmap='rainbow', s=0.5)


def visualizeEncoding(encoding: torch.Tensor) -> None:
    # encoding: (N, L)
    n_rows = encoding.shape[0] // 2
    n_cols = 2
    for i in range(n_rows):
        for j in range(n_cols):
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            plt.plot(encoding[i * n_cols + j, :].cpu(), linewidth=0.1)


def renderTrajRecovery(good_traj: torch.Tensor,
                       noise_traj: torch.Tensor,
                       recover_traj: torch.Tensor,
                       broken_traj: torch.Tensor) -> plt.figure:
    # traj: (2, traj_length)
    # recover_traj: (2, traj_length)

    # draw original trajectory
    plt.subplot(2, 2, 1)
    plt.title("original")
    visualizeTraj(good_traj.detach())

    # draw recovered trajectory
    plt.subplot(2, 2, 2)
    plt.title("noise traj")
    visualizeTraj(noise_traj.detach())

    plt.subplot(2, 2, 3)
    plt.title("recovered")
    visualizeTraj(recover_traj.detach())

    plt.subplot(2, 2, 4)
    plt.title("broken")
    visualizeTraj(broken_traj.detach())

    # render the figure and return the image as numpy array
    plt.tight_layout()

    return plt.gcf()


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


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.avg = 0
        self.size = 0

    def __lshift__(self, number: float) -> None:
        moving_sum = (self.avg * self.size - self.avg + number)
        self.size = min(self.size + 1, self.window_size)
        self.avg = moving_sum / self.size

    def __float__(self) -> float:
        return self.avg

    def __str__(self) -> str:
        return str(self.avg)

    def __repr__(self) -> str:
        return str(self.avg)

    def __format__(self, format_spec: str) -> str:
        return self.avg.__format__(format_spec)



if __name__ =="__main__":
    ema = MovingAverage(100)

    for i in range(1000):
        ema << np.random.randn()
        print(ema)