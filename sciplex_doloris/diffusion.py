"""Gaussian diffusion as DOLORIS runs it (guided-diffusion's GaussianDiffusion + SpacedDiffusion).

Ported from the released code (github.com/ChangxiChi/Doloris: train.py, diffusion.py,
respace.py), only the parts the sci-Plex pipeline reaches:

  * linear beta schedule over `steps` (500) base steps, beta in [1e-4, 2e-2] x 1000/steps
  * the process is *respaced* to `respacing` (50) steps with DDIM striding ("ddim50"). The
    released code builds the SpacedDiffusion with timestep_respacing="ddim50" for training as
    well as for sampling, so training samples t only from the 50 retained steps; the network
    receives the original step index (0, 10, ..., 490; rescale_timesteps=False).
  * the network predicts x0 (predict_xstart=True); training loss is MSE over the entries where
    x0 > 0 only, normalised by their count over the whole batch ("Unlasting code" in
    diffusion.training_losses)
  * deterministic DDIM sampling (eta = 0) and DDIM inversion, with x0 clipped to [-1, 1]
    (clip_denoised=True in test_util.predict for molecular data)
"""
from __future__ import annotations

import numpy as np
import torch


def linear_betas(steps: int) -> np.ndarray:
    scale = 1000 / steps
    return np.linspace(scale * 1e-4, scale * 2e-2, steps, dtype=np.float64)


def ddim_stride(steps: int, count: int) -> list[int]:
    """space_timesteps(steps, f"ddim{count}"): the first integer stride giving exactly `count` steps."""
    for stride in range(1, steps):
        if len(range(0, steps, stride)) == count:
            return list(range(0, steps, stride))
    raise ValueError(f"cannot create exactly {count} steps from {steps} with an integer stride")


class SpacedDiffusion:
    def __init__(self, steps: int = 500, respacing: int = 50, clip_denoised: bool = True):
        base_acp = np.cumprod(1.0 - linear_betas(steps))
        self.timestep_map = ddim_stride(steps, respacing)
        # SpacedDiffusion: new betas from the retained cumulative products, so acp[k] = base_acp[map[k]]
        self.acp = base_acp[self.timestep_map]
        self.acp_prev = np.append(1.0, self.acp[:-1])
        self.acp_next = np.append(self.acp[1:], 0.0)
        self.num_timesteps = len(self.timestep_map)
        self.clip_denoised = clip_denoised

    # coefficients are float64 in numpy and cast to float32 per step, as _extract_into_tensor does
    def _c(self, arr: np.ndarray, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(arr, device=x.device)[t].float().view(-1, *([1] * (x.dim() - 1)))

    def model_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.timestep_map, device=t.device, dtype=t.dtype)[t]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._c(np.sqrt(self.acp), t, x0) * x0 + self._c(np.sqrt(1.0 - self.acp), t, x0) * noise

    def eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (self._c(np.sqrt(1.0 / self.acp), t, x_t) * x_t - x0) / self._c(np.sqrt(1.0 / self.acp - 1), t, x_t)

    def training_loss(self, model, x0: torch.Tensor, model_kwargs: dict, generator=None, t=None, noise=None) -> torch.Tensor:
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device, generator=generator)
        if noise is None:
            noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        out = model(self.q_sample(x0, t, noise), self.model_t(t), **model_kwargs)
        mask = (x0 > 0).float()
        return ((x0 - out) ** 2 * mask).sum() / mask.sum()

    def pred_x0(self, model, x: torch.Tensor, t: torch.Tensor, model_kwargs: dict) -> torch.Tensor:
        out = model(x, self.model_t(t), **model_kwargs)
        return out.clamp(-1, 1) if self.clip_denoised else out

    @torch.no_grad()
    def ddim_sample(self, model, noise: torch.Tensor, model_kwargs: dict) -> torch.Tensor:
        """ddim_sample_loop(noise=..., eta=0): from the latent at the last retained step to x0."""
        x = noise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            x0 = self.pred_x0(model, x, t, model_kwargs)
            eps = self.eps_from_x0(x, t, x0)
            a_prev = self._c(self.acp_prev, t, x)
            x = x0 * torch.sqrt(a_prev) + torch.sqrt(1 - a_prev) * eps
        return x

    @torch.no_grad()
    def ddim_reverse(self, model, x: torch.Tensor, model_kwargs: dict) -> torch.Tensor:
        """ddim_reverse_sample_loop: encode data into the shared latent with the deterministic ODE."""
        for i in range(self.num_timesteps):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            x0 = self.pred_x0(model, x, t, model_kwargs)
            eps = self.eps_from_x0(x, t, x0)
            a_next = self._c(self.acp_next, t, x)
            x = x0 * torch.sqrt(a_next) + torch.sqrt(1 - a_next) * eps
        return x
