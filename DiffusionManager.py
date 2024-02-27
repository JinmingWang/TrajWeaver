import torch
from typing import *
from tqdm import tqdm
from math import log


# import cv2

class DiffusionManager:
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear",
                 shifted: bool = False):

        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        self.shifted = shifted
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step
        self.b = betas.view(-1, 1, 1)  # (T, 1, 1)
        self.a = alphas.view(-1, 1, 1)  # (T, 1, 1)
        self.abar = alpha_bars.view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_abar = torch.sqrt(alpha_bars).view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_1_m_abar = torch.sqrt(1 - alpha_bars).view(-1, 1, 1)  # (T, 1, 1)


    def diffusionForward(self, x_0, t, epsilon):
        """
        Forward Diffusion Process
        :param x_0: input (B, C, L)
        :param t: time steps (B, )
        :param epsilon: noise (B, C, L)
        :return: x_t: output (B, C, L)
        """
        x_t = self.sqrt_abar[t] * x_0 + self.sqrt_1_m_abar[t] * epsilon
        return x_t

    def diffusionBackwardStep(self, x_t: torch.Tensor, t: int, epsilon_pred: torch.Tensor):
        """
        Backward Diffusion Process
        :param x_t: input images (B, C, L)
        :param t: time steps
        :param epsilon_pred: predicted noise (B, C, L)
        :param scaling_factor: scaling factor of noise
        :return: x_t-1: output images (B, C, L)
        """
        beta = self.b[t].view(-1, 1, 1)
        alpha = self.a[t].view(-1, 1, 1)
        sqrt_1_minus_alpha_bar = self.sqrt_1_m_abar[t].view(-1, 1, 1)

        mu = (x_t - beta / sqrt_1_minus_alpha_bar * epsilon_pred) / torch.sqrt(alpha)

        if t == 0:
            return mu
        else:
            stds = torch.sqrt((1 - self.abar[t - 1]) / (1 - self.abar[t]) * beta) * torch.randn_like(x_t) * 0.5
            return mu + stds

    @torch.no_grad()
    def diffusionBackward(self, x_T: torch.Tensor, pred_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Backward Diffusion Process
        :param x_T: input (B, C, L)
        :param pred_func: the function that takes x_t and t and returns epsilon_pred
        :return: x_0: output (B, C, L)
        """
        B = x_T.shape[0]
        x_t = x_T
        tensor_t = torch.arange(self.T, dtype=torch.long, device=x_t.device).repeat(B, 1)  # (B, T)
        for t in tqdm(range(self.T - 1, -1, -1), "Diffusion Backward"):
            epsilon_pred = pred_func(x_t, tensor_t[:, t])  # epsilon_pred: (B, C, L)
            x_t = self.diffusionBackwardStep(x_t, t, epsilon_pred)
        return x_t



