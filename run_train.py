import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from diffusers import (
    DDPMScheduler,
    UniDiffuserModel,
    AutoencoderKL,
)
from transformers import ViTForImageClassification, ViTImageProcessor
import torchvision.transforms as T

from utils.dataset_strategy import get_loader
from utils.utils import AvgMeter, adjust_lr
from utils.image_util import cutmix
from torch.cuda import amp
from safetensors.torch import save_model

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from mutual_cross_domain_attention import MutualCrossDomainAttention

import numpy as np
from PIL import Image

from val_utils import compute_validation_metrics
from datetime import datetime

# ---------------------------------------------------------
# Custom Network Modifiers
# ---------------------------------------------------------

def replace_dit_in_dim(dit):
    """Replaces the input projection layer of UniDiffuser to accept 8 channels instead of 4."""
    old_proj = dit.vae_img_in.proj
    _weight = old_proj.weight.clone() 
    _bias = old_proj.bias.clone() if old_proj.bias is not None else None
    
    # Duplicate along the channel dimension
    _weight = _weight.repeat((1, 2, 1, 1))  
    _weight *= 0.5
    
    _n_convin_out_channel = old_proj.out_channels
    _new_conv_in = nn.Conv2d(
        8, _n_convin_out_channel, 
        kernel_size=old_proj.kernel_size, 
        stride=old_proj.stride, 
        padding=old_proj.padding,
        bias=(old_proj.bias is not None)
    )
    _new_conv_in.weight = nn.Parameter(_weight)
    if _bias is not None:
        _new_conv_in.bias = nn.Parameter(_bias)
        
    dit.vae_img_in.proj = _new_conv_in
    dit.config["in_channels"] = 8
    return dit

class DBIATransformerBlockWrapper(nn.Module):
    """Wraps a transformer block to inject Mutual Cross-Domain Attention."""
    def __init__(self, original_block, embed_dim, num_heads):
        super().__init__()
        self.original_block = original_block
        self.dbia = MutualCrossDomainAttention(embed_dim * num_heads, num_heads)
        self.norm = nn.LayerNorm(embed_dim * num_heads)

    def forward(self, hidden_states, *args, **kwargs):
        # Forward through original block
        block_output = self.original_block(hidden_states, *args, **kwargs)
        h = block_output[0] if isinstance(block_output, tuple) else block_output
        
        # Apply DBIA and Residual
        dbia_h = self.dbia(self.norm(h))
        h = h + dbia_h
        
        if isinstance(block_output, tuple):
            return (h,) + block_output[1:]
        return h

# ---------------------------------------------------------
# Arguments & Setup
# ---------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_vae', type=float, default=1e-4, help='VAE learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument("--pretrained_model_name_or_path", type=str, default='/path/to/sd-turbo/')
parser.add_argument("--unidiffuser_model_path", type=str, default="thu-ml/unidiffuser-v1")
parser.add_argument("--dataset_path", type=str, default='/path/to/DIS5K/')

opt = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float32

writer = SummaryWriter(f'./runs/DiffDIS/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

test_datasets = {
    'DIS-VD': "DIS5K/DIS-VD",
}

# ---------------------------------------------------------
# Build Models
# ---------------------------------------------------------

# 1. VAE Loader
vae = AutoencoderKL.from_pretrained(opt.unidiffuser_model_path, subfolder='vae')
# Train only VAE decoder
vae.encoder.requires_grad_(False)
vae.quant_conv.requires_grad_(False)
vae.decoder.requires_grad_(True)
vae.post_quant_conv.requires_grad_(True)
vae.encoder.eval()
vae.quant_conv.eval()
vae.to(device, dtype=weight_dtype)

# 2. UniDiffuser Loader & Injection
dit = UniDiffuserModel.from_pretrained(opt.unidiffuser_model_path, subfolder="unet")
dit = replace_dit_in_dim(dit)

# Inject DBIA into the middle block
embed_dim = dit.config.attention_head_dim
num_heads = dit.config.num_attention_heads
original_middle_block = dit.transformer.transformer_mid_block

dbia_wrapper = DBIATransformerBlockWrapper(
    original_block=original_middle_block,
    embed_dim=embed_dim,
    num_heads=num_heads
)
dit.transformer.transformer_mid_block = dbia_wrapper
dit.train().to(device, dtype=weight_dtype)

# 3. ViT Conditioning
vit_model_name = 'google/vit-base-patch16-224'
vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
vit_preprocessor = ViTImageProcessor.from_pretrained(vit_model_name)
vit_model.train()
vit_model.requires_grad_(False)
vit_model.to(device, dtype=weight_dtype)

# Projection layers (Trainable)
vit_projection = nn.Linear(vit_model.config.hidden_size, 64).to(device, dtype=weight_dtype)
bde_proj_layer = nn.Linear(4, dit.config["clip_img_dim"]).to(device, dtype=weight_dtype)
vit_projection.train()
bde_proj_layer.train()

# 4. Scheduler
noise_scheduler = DDPMScheduler.from_pretrained(opt.unidiffuser_model_path, subfolder='scheduler')
# noise_scheduler.set_timesteps(1, device=device)
noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

masks_log_dir = './masks_logs'
os.makedirs(masks_log_dir, exist_ok=True)

# ---------------------------------------------------------
# Optimizers & Loss
# ---------------------------------------------------------

params_to_optimize = (
    list(dit.parameters()) + 
    list(vit_projection.parameters()) + 
    list(bde_proj_layer.parameters()) +
    list(vit_model.parameters())
)

vae_decoder_optimizer = torch.optim.Adam(list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters()), lr=opt.lr_vae)

generator_optimizer = torch.optim.Adam([
    {"params": params_to_optimize, "lr": opt.lr_gen}
])

mse_loss = torch.nn.MSELoss(reduction="mean")
scaler = amp.GradScaler(enabled=True)

# ---------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------

train_image_root = f'{opt.dataset_path}/DIS-TR/im/'
train_gt_root = f'{opt.dataset_path}/DIS-TR/gt/'
train_edge_root = f'{opt.dataset_path}/DIS-TR/contour/'

val_image_root = f'{opt.dataset_path}/DIS-VD/im/'
val_gt_root = f'{opt.dataset_path}/DIS-VD/gt/'

train_loader = get_loader(train_image_root, train_gt_root, train_edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
print(f'Total steps per epoch: {len(train_loader)}')

rgb_latent_scale_factor = 0.18215

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------

for epoch in range(1, opt.epoch + 1):
    dit.train()
    vit_projection.train()
    bde_proj_layer.train()
    vae.decoder.train()
    vae.post_quant_conv.train()
    
    loss_record = AvgMeter()
    print(f"Generator Learning Rate: {generator_optimizer.param_groups[0]['lr']}")
    
    for i, pack in enumerate(train_loader, start=1):
        generator_optimizer.zero_grad()
        vae_decoder_optimizer.zero_grad()

        rgb, label, edge, box = pack
        rgb = rgb.cuda().to(weight_dtype) 
        label = label.unsqueeze(1).repeat(1, 3, 1, 1).cuda().to(weight_dtype) 
        edge = edge.unsqueeze(1).repeat(1, 3, 1, 1).cuda().to(weight_dtype) 
        box = box.unsqueeze(1).repeat(1, 3, 1, 1).cuda().to(weight_dtype) 

        bsz = rgb.shape[0]
        assert bsz % 2 == 0, "Batch size must be even"

        # ---------------- CutMix Logic ----------------
        rgb_chunks = rgb.chunk(bsz // 2, dim=0)       
        label_chunks = label.chunk(bsz // 2, dim=0)    
        edge_chunks = edge.chunk(bsz // 2, dim=0)      
        box_chunks = box.chunk(bsz // 2, dim=0) 

        rgbs, labels, edges = [], [], []
        for rgb_, label_, edge_, box_ in zip(rgb_chunks, label_chunks, edge_chunks, box_chunks):
            mixed_rgb, mixed_label, mixed_edge = cutmix(rgb_, label_, edge_, box_)
            rgbs.append(mixed_rgb)
            labels.append(mixed_label)
            edges.append(mixed_edge)
            
        rgb_mix = torch.cat(rgbs, dim=0)
        label_mix = torch.cat(labels, dim=0)
        edge_mix = torch.cat(edges, dim=0)

        # ---------------- VAE Encoding ----------------
        with torch.no_grad():
            h_batch = vae.encoder(torch.cat((rgb_mix, label_mix, edge_mix), dim=0))
            moments_batch = vae.quant_conv(h_batch)
            mean_batch, _ = torch.chunk(moments_batch, 2, dim=1)
            batch_latents = mean_batch * rgb_latent_scale_factor
            
            rgb_latents, mask_latents, edge_latents = torch.chunk(batch_latents, 3, dim=0)

        # ---------------- Diffusion Setup ----------------
        unified_latents = torch.cat((mask_latents, edge_latents), dim=0)
        noise = torch.randn_like(unified_latents)
        
        timesteps = torch.tensor([999], device=device).long()
        noisy_unified_latents = noise_scheduler.add_noise(unified_latents, noise, timesteps.repeat(bsz * 2))

        # Re-pack input to [2B, 8, H, W]
        dit_input = torch.cat([rgb_latents.repeat(2, 1, 1, 1), noisy_unified_latents], dim=1)  
        
        # ---------------- Conditions Setup ----------------
        # 1. ViT Image Embeddings Setup
        with torch.no_grad():
            vit_inputs = vit_preprocessor(rgb_mix, return_tensors='pt')
            pixel_values = vit_inputs['pixel_values'].to(device=device, dtype=weight_dtype)
            vit_outputs = vit_model.vit(pixel_values=pixel_values)
            
        vit_embeds = vit_projection(vit_outputs.last_hidden_state.to(device=device, dtype=weight_dtype))
        # Duplicate for Mask and Edge batches
        prompt_embeds_doubled = torch.cat([vit_embeds[:, :77], vit_embeds[:, :77]], dim=0)

        # 2. Discriminative Labels (BDE) Setup
        discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device=device)
        BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)
        bde_embeddings = bde_proj_layer(BDE).unsqueeze(1)

        # ---------------- Forward Pass ----------------
        with amp.autocast(enabled=True):
            noise_pred, _, _ = dit(
                latent_image_embeds=dit_input,
                image_embeds=bde_embeddings,
                prompt_embeds=prompt_embeds_doubled,
                timestep_img=timesteps.repeat(bsz * 2),
                timestep_text=timesteps.repeat(bsz * 2),
                encoder_hidden_states=None
            )
            
            # Split the added noise and the predicted noise
            noise_mask, noise_edge = torch.chunk(noise, 2, dim=0)
            noise_pred_mask, noise_pred_edge = torch.chunk(noise_pred, 2, dim=0)

            # Calculate loss directly on the noise
            loss1 = mse_loss(noise_pred_mask, noise_mask)
            loss2 = mse_loss(noise_pred_edge, noise_edge)
            loss = loss1 + loss2


        # ---------------- Backward & Step ----------------
        scaler.scale(loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()
        
        loss_record.update(loss.data, opt.batchsize)

        # ----- VAE loss and step -----
        x_denoised = noise_scheduler.step(noise_pred, timesteps, noisy_unified_latents, return_dict=True).pred_original_sample
        mask_latent, edge_latent = torch.chunk(x_denoised, 2, dim=0)
        vae_decoded_img = vae.decode(mask_latent.detach() / rgb_latent_scale_factor).sample
        vae_loss = mse_loss(vae_decoded_img, label_mix)  # VAE reconstruction loss on the mask part

        vae_loss.backward()
        vae_decoder_optimizer.step()

        # ---------------- Logging ----------------
        writer.add_scalar('mask_loss', loss1.item(), epoch * len(train_loader) + i)
        writer.add_scalar('edge_loss', loss2.item(), epoch * len(train_loader) + i)
        writer.add_scalar('total_loss', loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('vae_loss', vae_loss.item(), epoch * len(train_loader) + i)


        if i % 100 == 0:
            with torch.no_grad():
                x_denoised = noise_scheduler.step(noise_pred, timesteps, noisy_unified_latents, return_dict=True).pred_original_sample
                    
                # split mask and edge latents along batch dimension 
                mask_latent, edge_latent = torch.chunk(x_denoised, 2, dim=0)
                
                pred_mask = vae.decode(mask_latent / rgb_latent_scale_factor).sample
                pred_mask_np = pred_mask[0].permute(1, 2, 0).cpu().numpy()
                pred_mask_np = (pred_mask_np + 1) / 2 * 255
                pred_mask_pil = Image.fromarray(pred_mask_np.astype(np.uint8)).convert('L')
                pred_mask_pil.save(os.path.join(masks_log_dir, f'epoch{epoch:03d}_step{i:04d}.png'))
                print("Image saved to:", os.path.join(masks_log_dir, f'epoch{epoch:03d}_step{i:04d}.png'))

        if i % 10 == 0 or i == len(train_loader):
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}, mask loss:{:.4f}, edge loss:{:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, len(train_loader), loss_record.show(), loss1.item(), loss2.item()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    compute_validation_metrics(dit_model=dit,
                               vit_model=vit_model,
                               vit_preprocessor=vit_preprocessor,
                               vit_proj=vit_projection,
                               bde_proj=bde_proj_layer,
                               vae=vae,
                               noise_scheduler=noise_scheduler,
                               test_datasets_dict=test_datasets,
                               rgb_latent_scale_factor=rgb_latent_scale_factor,
                               weight_dtype=weight_dtype,
                               epoch=epoch,
                               writer=writer,
                               )

    # ---------------- Save Checkpoints ----------------
    if epoch % 10 == 0: 
        save_path = f'../saved_model/DiffDIS/Model_{epoch}/'
        os.makedirs(f'{save_path}dit/', exist_ok=True)
        
        save_model(dit, f'{save_path}dit/diffusion_pytorch_model.safetensors')
        torch.save(vit_projection.state_dict(), f'{save_path}vit_projection.pth')
        torch.save(bde_proj_layer.state_dict(), f'{save_path}bde_proj_layer.pth')
        torch.save(generator_optimizer.state_dict(), f'{save_path}generator_optimizer.pth')