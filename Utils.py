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