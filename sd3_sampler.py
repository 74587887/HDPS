from typing import List, Tuple, Optional
import math
import torch
import torch.nn.functional as F
from torch.fft import fftn, ifftn, fftshift, ifftshift

from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
from diffusers import AutoencoderTiny
from torch.utils.checkpoint import checkpoint
from piq import LPIPS
import matplotlib.pyplot as plt
import numpy as np

from utils import diffpir_util as sr

# =======================================================================
# Factory
# =======================================================================

__SOLVER__ = {}

def register_solver(name:str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name:str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

# =======================================================================


class StableDiffusion3Base():
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        pipe.vae = AutoencoderTiny.from_pretrained("pretrained_models/taesd3", torch_dtype=torch.float16)

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae=pipe.vae
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        # self.transformer.requires_grad_(True)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)-1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        del pipe

        self.lpips_loss = LPIPS(replace_pooling=True, reduction="mean")

    def encode_prompt(self, prompt: List[str], batch_size:int=1) -> List[torch.Tensor]:
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        # CLIP encode (used for modulation of adaLN-zero)
        # now, we have two CLIPs
        text_clip1_ids = self.tokenizer_1(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(self.text_enc_1.device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=self.text_enc_1.device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=self.text_enc_1.device)

        text_clip2_ids = self.tokenizer_2(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(self.text_enc_2.device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=self.text_enc_2.device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=self.text_enc_2.device)

        # T5 encode (used for text condition)
        text_t5_ids = self.tokenizer_3(prompt,
                                       padding="max_length",
                                       max_length=77,
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_tensors='pt').input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(self.text_enc_3.device))[0]
        text_t5_emb = text_t5_emb.to(dtype=self.dtype, device=self.text_enc_3.device)


        # Merge
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(
            clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1])
        )
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb


    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):
        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        # z = self.vae.encode(image).latent_dist.sample()
        
        z_output = self.vae.encode(image)
        # 检查返回的是否是 AutoencoderTinyOutput
        if hasattr(z_output, "latents"):
            z = z_output.latents
        else:
            # 兼容原有的 VAE 逻辑
            z = z_output.latent_dist.sample() if hasattr(z_output, "latent_dist") else z_output

        z = (z-self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z/self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
        return v

class SD3Euler(StableDiffusion3Base):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda'):
        super().__init__(model_key=model_key, device=device)

    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1,
                  prompt_emb:Optional[List[torch.Tensor]]=None,
                  null_emb:Optional[List[torch.Tensor]]=None):

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""])
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        # initialize latent
        src_img = src_img.to(device=self.vae.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img).to(self.transformer.device)

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.transformer.device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.transformer.device)])
        timesteps = reversed(timesteps)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps[:-1], total=NFE, desc='SD3 Euler Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.transformer.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1]

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        return z

    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               **kwargs):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@register_solver("flowdps")
class SD3FlowDPS(SD3Euler):
    def data_consistency(self, z0t, operator, measurement, task, stepsize: float=30.0):
        z0t = z0t.requires_grad_(True)
        num_iters = 16  # 16  # 3
        for _ in range(num_iters):
            x0t = self.decode(z0t).float()
            if "sr" in task:
                loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
            else:
                loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
            grad = torch.autograd.grad(loss, z0t)[0]
            z0t = z0t - stepsize*grad

        return z0t.detach()

    def get_src(self, operator, measurement, task, batch_size, imgH, imgW):
        if "sr" in task:
            src0 = operator.A_pinv(measurement)
        else:
            src0 = operator.At(measurement)
        return src0.view(batch_size, 3, imgH, imgW)

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=30.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               **kwargs):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        
        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3-FlowDPS')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            # denoising
            z0t = z - sigma * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            z1t = z + (1-sigma) * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            delta = sigma - sigma_next

            if i < NFE:

                z0y = self.data_consistency(z0t, operator, measurement, task=task, stepsize=step_size)

                z0y = (1-sigma) * z0t + sigma * z0y

            else:
                z0y = z0t

            # renoising
            noise = math.sqrt(sigma_next) * z1t + math.sqrt(1-sigma_next) * torch.randn_like(z1t)
            # noise = z1t
            z = z0y + (sigma-delta) * (noise - z0y)

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@register_solver("flowchef")
class SD3FlowChef(SD3Euler):
    def data_consistency(self, z0t, operator, measurement, task):
        z0t = z0t.requires_grad_(True)
        x0t = self.decode(z0t).float()
        if "sr" in task:
            loss = torch.linalg.norm((operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1))
        else:
            loss = torch.linalg.norm((operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1))
        grad = torch.autograd.grad(loss, z0t)[0]
        return grad.detach()


    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=0.5,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               **kwargs):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3-FlowChef')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            # denoising
            z0t = z - sigma * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            z1t = z + (1-sigma) * (pred_null_v + cfg_scale * (pred_v-pred_null_v))
            delta = sigma - sigma_next

            if i < NFE:
                grad = self.data_consistency(z0t, operator, measurement, task=task)
            else:
                grad = 0

            # renoising
            z = z0t + (sigma-delta) * (z1t - z0t) - step_size*grad

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img

@register_solver("resample")
class SD3ReSample(SD3Euler):
    def data_consistency(self, z0t, operator, measurement, task, lr=1):
        if "sr" in task:
            lr = lr * 10
        z0t = z0t.requires_grad_(True)
        num_iters = 30
        for _ in range(num_iters):
            x0t = self.decode(z0t).float()
            loss = torch.linalg.norm((operator.A(x0t) - measurement).view(1, -1))
            grad = torch.autograd.grad(loss, z0t)[0]
            z0t = z0t - lr*grad
        return z0t.detach()

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE: int,
               img_shape: Optional[Tuple[int]] = None,
               cfg_scale: float = 1.0, batch_size: int = 1,
               step_size: float = 40.0, # 外部传入的参数
               latent: Optional[List[torch.Tensor]] = None,
               prompt_emb: Optional[List[torch.Tensor]] = None,
               null_emb: Optional[List[torch.Tensor]] = None,
               **kwargs):

        device = self.device
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(range(len(timesteps) - 1), desc='SD3-ReSample')
        for i in pbar:
            t_curr = timesteps[i]
            
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            alpha_next = (1-sigma_next) ** 2
            
            with torch.no_grad():
                v_t = self.predict_vector(z, t_curr.expand(batch_size), prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    v_null = self.predict_vector(z, t_curr.expand(batch_size), null_prompt_emb, null_pooled_emb)
                    v_t = v_null + cfg_scale * (v_t - v_null)
                
            z0t = z - sigma * v_t
            z_prime = z + (sigma_next - sigma) * v_t

            if i < NFE:
                z0y = self.data_consistency(
                    z0t, operator, measurement, task
                )
            else:
                z0y = z0t

            # renoising
            gamma = step_size
            noise_new = torch.randn_like(z0y) * (gamma*(1-alpha_next) / (gamma + (1-alpha_next)))
            mean = (gamma*math.sqrt(alpha_next)*z0y + (1-alpha_next)*z_prime) / (gamma + (1-alpha_next))
            z = mean + noise_new

        with torch.no_grad():
            img = self.decode(z)
            
        return img
    
@register_solver("flair")
class SD3FLAIR(SD3Euler):
    def find_closest_t(self, t, regularizer_weight):
        ts = torch.linspace(1, 0.0, regularizer_weight.shape[0], device=t.device, dtype=t.dtype)
        return torch.argmin(torch.abs(ts - t))
    
    def compute_regularization_term(self, noise, t, z0y, pred_v, weight):
        x_t = (1-t)*z0y + t*noise
        lambda_t_der = -2 * (1/(1-t) + 1/t)
        u_t = - 1 / (1-t) * x_t - t * lambda_t_der / 2 * noise
        reg_term = -(u_t - pred_v).reshape(noise.shape[0], -1)

        return reg_term * weight

    def get_src(self, operator, measurement, task, batch_size, imgH, imgW):
        if "sr" in task:
            src0 = operator.A_pinv(measurement)
        else:
            src0 = operator.At(measurement)
        return src0.view(batch_size, 3, imgH, imgW)

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=30.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               **kwargs):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # Encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # Initialize latent
        if latent is None:
            z1t = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z1t = latent

        with torch.no_grad():
            x0y = self.get_src(operator, measurement, task, batch_size, imgH, imgW)
            z0y = self.encode(x0y.half())
        
        z0y = z0y.detach().clone().requires_grad_(True)
        optimizer_x = torch.optim.SGD([z0y], lr=step_size)
        optimizer_z = torch.optim.SGD([z0y], lr=1)

        sigmas = torch.linspace(1, 0.18, NFE+2, device=self.device, dtype=torch.float32)
        sigmas = sigmas[1:-1]  

        reg_weight = np.load("../../FLAIR/FLAIR-main/LCFM/SD3_loss_v_MSE_DIV2k_neg_prompt.npy")
        reg_weight = 1 / (reg_weight + 1e-7)
        reg_weight = reg_weight / np.nansum(reg_weight) * reg_weight.shape[0]
        regularizer_weight = np.clip(reg_weight, 0, None) * 0.5

        # Solve ODE
        pbar = tqdm(sigmas, total=NFE, desc='SD3-FLAIR')
        for i, t in enumerate(pbar):
            timestep = (t*self.scheduler.config.num_train_timesteps).expand(z1t.shape[0]).to(self.device)
            
            sigma = sigmas[i]
            reg_idx = self.find_closest_t(sigma, regularizer_weight)
            weight = regularizer_weight[reg_idx]

            # 1. Renoising
            noise = (1-sigma) * z1t + math.sqrt(1-(1-sigma)**2) * torch.randn_like(z1t)
            z = (1-sigma)*z0y + sigma*noise

            # 2. Predict
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
            pred_v = (pred_null_v + cfg_scale * (pred_v - pred_null_v)).detach()

            # 3. Denoising Logs
            z1t = z + (1-sigma) * pred_v

            # 4. Prior Update (Reg Term)
            reg_term_val = self.compute_regularization_term(noise, sigma, z0y, pred_v, weight)
            with torch.enable_grad():
                loss_z = (reg_term_val.detach() * z0y.view(reg_term_val.shape[0], -1)).sum()
            
            loss_z.backward()
            optimizer_z.step()
            optimizer_z.zero_grad()
            del loss_z

            # 5. Data Consistency (DC-X)
            meas_detach = measurement.detach()
            for _ in range(15):
                with torch.enable_grad():
                    x0y_rec = self.decode(z0y).float()
                    loss_x = torch.nn.MSELoss(reduction='sum')(operator.A(x0y_rec), meas_detach)
                    
                    if loss_x < 1e-4 * meas_detach.numel():
                        z0y.grad = None
                        break
                    
                    loss_x = loss_x.sum() * weight
                
                loss_x.backward()
                optimizer_x.step()
                optimizer_x.zero_grad()

                del loss_x

        # Final decode
        with torch.no_grad():
            img = self.decode(z0y)
        return img


# z daps
@register_solver("latentdaps")
class SD3LatentDAPS(SD3Euler):
    def data_consistency_z(self, z0t, y, operator, sigma, num_iters, step_size, tau=1e-4):
        y = y.detach().clone()
        z0_anchor = z0t.detach().clone()
        z0t = z0t.detach().clone().requires_grad_(True)

        for _ in range(num_iters):
            x0 = self.decode(z0t).float()
            loss_data  = ((y - operator.A(x0))**2).mean()
            data_fitting_grad = torch.autograd.grad(loss_data, z0t)[0]
            data_term = - data_fitting_grad / tau ** 2

            prior_term = (z0_anchor - z0t) / sigma ** 2

            noise = torch.randn_like(z0t)
            z0t = z0t + step_size * (data_term+prior_term) + math.sqrt(2 * step_size) * noise 

        return z0t.detach()
    
    def get_pinv(self, operator, measurement, task, batch_size, imgH, imgW):
        if "sr" in task:
            src0 = operator.A_pinv(measurement)
        else:
            src0 = operator.At(measurement)
        return src0.view(batch_size, 3, imgH, imgW)
    
    def calculate_v(self, z, timestep, prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale):
        with torch.no_grad():
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0
        pred_v = pred_null_v + cfg_scale * (pred_v-pred_null_v)

        return pred_v 

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=30.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               **kwargs):

        mask_type = kwargs['mask_type']
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialization preparation
        x0y = self.get_pinv(operator, measurement, task, batch_size, imgH, imgW)
        z0y = self.encode(x0y.half())

        z = latent
        if mask_type == "box":
            # initialize latent
            if latent is None:
                z = self.initialize_latent((imgH, imgW), batch_size)
            # timesteps (default option. You can make your custom here.)
            self.scheduler.config.shift = 4.0
            self.scheduler.set_timesteps(NFE, device=self.device)
            timesteps = self.scheduler.timesteps
            sigmas = timesteps / self.scheduler.config.num_train_timesteps
        else:
            # initialize latent
            if latent is None:
                z = self.initialize_latent((imgH, imgW), batch_size)
                z = 0.8 * z + (1-0.8) * z0y
            # timesteps (default option. You can make your custom here.)
            self.scheduler.config.shift = 4.0
            self.scheduler.set_timesteps(NFE*2, device=self.device)
            timesteps = self.scheduler.timesteps[NFE:]
            sigmas = timesteps / self.scheduler.config.num_train_timesteps
        
        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3-LatentDAPS')
        for i, t in enumerate(pbar):

            timestep = t.expand(z.shape[0]).to(self.device)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
            delta = sigma - sigma_next

            # denoising
            pred_v = self.calculate_v(z, timestep, prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale)
            z0t = z - sigma * pred_v
            z1t = z + (1-sigma) * pred_v

            # task specific params
            if "sr" in task:
                if i > 43:
                    break
            elif "motion" in task:
                if i > 44:
                    break
            elif "gauss" in task:
                if i > 45:
                    break
            elif "inpainting" in task:
                if i > 44:
                    break

            # optimization
            z0y = self.data_consistency_z(z0t, measurement, operator, sigma, num_iters=15, step_size=1e-4)

            # renoising
            noise = (1-sigma) * z1t + math.sqrt(1-(1-sigma)**2) * torch.randn_like(z1t)
            z = z0y + (sigma-delta) * (noise - z0y)

        # decode
        with torch.no_grad():
            img = self.decode(z0y)
        return img

@register_solver("hdps")
class SD3HDPS(SD3Euler):
    def data_consistency_z(self, z0t, x0y, sigma, num_iters, step_size):
        x_anchor = x0y.detach().clone()
        z0t = z0t.detach().clone().requires_grad_(True)

        for _ in range(num_iters):
            x0 = self.decode(z0t).float()
            loss = torch.linalg.norm((x_anchor - x0).view(1, -1))

            loss = loss * sigma
            grad = torch.autograd.grad(loss, z0t)[0]
            z0t = z0t - step_size*grad    

        return z0t.detach()
    
    def data_consistency_x(self, x0t, operator, measurement, sigma, task, num_iters, step_size, tau=1e-5):
        measurement = measurement.detach()
        x0_anchor = x0t.detach().clone()
        x0t = x0t.detach().clone().requires_grad_(True)

        for _ in range(num_iters):
            loss_data  = ((measurement - operator.A(x0t))**2).mean()
            data_fitting_grad = torch.autograd.grad(loss_data, x0t)[0]
            data_term = - data_fitting_grad / tau ** 2

            prior_term = (x0_anchor - x0t) / sigma ** 2

            noise = torch.randn_like(x0t)
            x0t = x0t + step_size * (data_term+prior_term) + math.sqrt(2 * step_size) * noise

        x0t = torch.clamp(x0t, -1.0, 1.0)

        return x0t.detach()

    def get_pinv(self, operator, measurement, task, batch_size, imgH, imgW):
        if "sr" in task:
            src0 = operator.A_pinv(measurement)
        else:
            src0 = operator.At(measurement)
        return src0.view(batch_size, 3, imgH, imgW)
    
    def calculate_v(self, z, timestep, prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale):
        with torch.no_grad():
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0
        pred_v = pred_null_v + cfg_scale * (pred_v-pred_null_v)

        return pred_v 

    def sample(self, measurement, operator, task,
               prompts: List[str], NFE:int,
               img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               step_size: float=30.0,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None,
               **kwargs):

        mask_type = kwargs['mask_type']
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]

            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]

            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # task specific params
        if "sr" in task:
            num_iters_x = 15
            lr_x = 1e-4
            num_iters_z = 15
            lr_z = 3
        elif "motion" in task:
            num_iters_x = 30
            lr_x = 1e-5
            num_iters_z = 15
            lr_z = 30
        elif "gauss" in task:
            num_iters_x = 20
            lr_x = 1e-4
            num_iters_z = 15
            lr_z = 15
        elif "inpainting" in task:
            if mask_type == "random":
                num_iters_x = 20
                lr_x = 1e-5
                num_iters_z = 15
                lr_z = 40

        # initialization preparation
        x0y = self.get_pinv(operator, measurement, task, batch_size, imgH, imgW)
        z0y = self.encode(x0y.half())

        ratio = 0.8
        ratios = {"0.5": 5.0, "0.6": 3.7, "0.7": 2.7,
                  "0.8": 2.0, "0.9": 1.4, "1.0": 1.0}

        z = latent
        if mask_type == "box":
            # initialize latent
            if latent is None:
                z = self.initialize_latent((imgH, imgW), batch_size)
            # timesteps (default option. You can make your custom here.)
            self.scheduler.config.shift = 4.0
            self.scheduler.set_timesteps(NFE, device=self.device)
            timesteps = self.scheduler.timesteps
            sigmas = timesteps / self.scheduler.config.num_train_timesteps
        else:
            # initialize latent
            if latent is None:
                z = self.initialize_latent((imgH, imgW), batch_size)
                z = ratio * z + (1-ratio) * z0y
            # timesteps (default option. You can make your custom here.)
            self.scheduler.config.shift = 4.0
            self.scheduler.set_timesteps(int(NFE*ratios[str(ratio)]), device=self.device)
            timesteps = self.scheduler.timesteps[-NFE:]
            sigmas = timesteps / self.scheduler.config.num_train_timesteps
        
        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3-HDPS')
        for i, t in enumerate(pbar):

            timestep = t.expand(z.shape[0]).to(self.device)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
            delta = sigma - sigma_next

            # denoising
            pred_v = self.calculate_v(z, timestep, prompt_emb, pooled_emb, null_prompt_emb, null_pooled_emb, cfg_scale)
            z0t = z - sigma * pred_v
            z1t = z + (1-sigma) * pred_v

            # task specific params
            if "sr" in task:
                if i > 42:
                    break

            # optimization
            x0t = self.decode(z0t).float()
            x0y = self.data_consistency_x(x0t, operator, measurement, sigma, task, num_iters=num_iters_x, step_size=lr_x)

            z0y = self.data_consistency_z(z0t, x0y, sigma, num_iters=num_iters_z, step_size=lr_z)

            # renoising (optional)
            noise = (1-sigma) * z1t + math.sqrt(1-(1-sigma)**2) * torch.randn_like(z1t)
            # noise = math.sqrt(1-sigma**2) * z1t + sigma * torch.randn_like(z1t)
            z = z0y + (sigma-delta) * (noise - z0y)

        # decode
        with torch.no_grad():
            img = self.decode(z0y)
        return img  
    
