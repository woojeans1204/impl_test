import torch
import torch.nn.functional as F
import numpy as np

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 샘플링용
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
            forward: x_0에 t 시점 노이즈
            x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start) # eps ~ N(0, I)

        # 인덱싱 및 차원 확장
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # sqrt(alpha) * x + sqrt(1-alpha) * eps
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_noise = model(x_noisy, t)

        return F.mse_loss(noise, predicted_noise)

    def _extract(self, a, t, x_shape):
        """배열 a에서 t 인덱스 값을 추출하여 x_shape과 차원을 맞춤"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        # [Batch, 1, 1, ..., 1]
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def p_sample(self, model, x, t, t_index):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(variance_t) * noise