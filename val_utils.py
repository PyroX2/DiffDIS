import torch
import os
import glob
import numpy as np
from utils.saliency_metric import (
    cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, 
    cal_dice, cal_iou, cal_ber, cal_acc, HCEMeasure
)
from PIL import Image
from tqdm import tqdm
from utils.image_util import resize_res
import cv2


@torch.no_grad()
def compute_validation_metrics(dit_model, vit_model, vit_preprocessor, vit_proj, bde_proj, vae, noise_scheduler, test_datasets_dict, 
                               rgb_latent_scale_factor, weight_dtype, epoch, writer, imsize=512):
    """
    Generate predictions using the model and compute validation metrics, then log to tensorboard
    """
    dit_model.eval()
    vit_model.eval()
    vit_proj.eval()
    vae.eval()
    device = next(dit_model.parameters()).device

    mse_loss = torch.nn.MSELoss(reduction="mean")

    
    with torch.no_grad():
        for dataset_name, gt_root_base in test_datasets_dict.items():
            # Get image and ground truth paths
            rgb_root = os.path.join(gt_root_base, 'im')
            gt_root = os.path.join(gt_root_base, 'gt')
            
            if not os.path.exists(rgb_root):
                print(f"Image directory {rgb_root} not found, skipping {dataset_name}")
                continue
                
            if not os.path.exists(gt_root):
                print(f"GT directory {gt_root} not found, skipping {dataset_name}")
                continue
            
            # Get list of image files
            EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
            rgb_filename_list = glob.glob(os.path.join(rgb_root, "*"))
            rgb_filename_list = [
                f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
            ]
            rgb_filename_list = sorted(rgb_filename_list)
            
            if len(rgb_filename_list) == 0:
                print(f"No images found in {rgb_root}")
                continue
            
            # Process validation images
            # num_samples = min(len(rgb_filename_list), 5)
            num_samples = len(rgb_filename_list)

            # Prepare metric calculators
            mae = cal_mae()
            fm = cal_fm(num_samples)
            sm = cal_sm()
            em = cal_em()
            wfm = cal_wfm()
            m_dice = cal_dice()
            m_iou = cal_iou()
            ber = cal_ber()
            acc = cal_acc()
            
            total_mask_loss = 0.0
            total_vae_loss = 0.0
            total_instances = 0

            for input_image_path in tqdm(rgb_filename_list[:num_samples], 
                                         desc=f"Validating {dataset_name}"):
                # Load and process image similar to run_inference.py
                input_image_pil = Image.open(input_image_path)
                w_, h_ = input_image_pil.size
                img_resize = resize_res(input_image_pil, resolution=imsize)
                
                input_image = img_resize.convert("RGB")
                image = np.array(input_image)
                rgb = np.transpose(image, (2, 0, 1))
                rgb_norm = rgb / 255.0 * 2.0 - 1.0
                rgb_tensor = torch.from_numpy(rgb_norm).to(device)
                rgb_tensor = rgb_tensor.float().unsqueeze(0)
                
                bsz = rgb_tensor.shape[0]
                
                # ---------------- VAE Encoding ----------------
                h_batch = vae.encoder(rgb_tensor)
                moments_batch = vae.quant_conv(h_batch)
                mean_batch, _ = torch.chunk(moments_batch, 2, dim=1)
                rgb_latents = mean_batch * rgb_latent_scale_factor
                
                # Create input for model (use zero mask and edge as input)
                mask_latents = torch.zeros_like(rgb_latents)
                edge_latents = torch.zeros_like(rgb_latents)
                
                # Concatenate for model input
                unified_latents = torch.cat([mask_latents, edge_latents], dim=0)
                
                timesteps = torch.tensor([999], device=device).long()
                noise = torch.randn_like(unified_latents)
                noisy_unified_latents = noise_scheduler.add_noise(unified_latents, noise, timesteps.repeat(2))

                dit_input = torch.cat([rgb_latents.repeat(2, 1, 1, 1), noisy_unified_latents], dim=1)  

                # Get ViT embeddings: use the processor to return tensors
                # and move pixel values to the same device/dtype as the models.
                vit_inputs = vit_preprocessor(input_image, return_tensors='pt')
                pixel_values = vit_inputs['pixel_values'].to(device=device, dtype=weight_dtype)
                vit_outputs = vit_model.vit(pixel_values=pixel_values)

                vit_embeds = vit_proj(vit_outputs.last_hidden_state.to(device=device, dtype=weight_dtype))
                prompt_embeds_doubled = torch.cat([vit_embeds[:, :77], vit_embeds[:, :77]], dim=0)
                
                # 2. Discriminative Labels (BDE) Setup
                discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device=device)
                BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)
                bde_embeddings = bde_proj(BDE).unsqueeze(1)
                
                noise_pred, _, _ = dit_model(
                    latent_image_embeds=dit_input,
                    image_embeds=bde_embeddings,
                    prompt_embeds=prompt_embeds_doubled,
                    timestep_img=timesteps.repeat(bsz * 2),
                    timestep_text=timesteps.repeat(bsz * 2),
                    encoder_hidden_states=None
                )

                noise_mask, noise_edge = torch.chunk(noise, 2, dim=0)
                noise_pred_mask, noise_pred_edge = torch.chunk(noise_pred, 2, dim=0)

                # Calculate loss directly on the noise
                mask_loss = mse_loss(noise_pred_mask, noise_mask)
                
                # Denoise one step
                x_denoised = noise_scheduler.step(noise_pred, timesteps, noisy_unified_latents, 
                                                 return_dict=True).pred_original_sample
                
                # Split into mask and edge
                mask_latent, edge_latent = torch.chunk(x_denoised, 2, dim=0)

                vae_decoded_img = vae.decode(mask_latent.detach() / rgb_latent_scale_factor).sample
                
                # Decode from latent space
                pred_mask = vae.decode(mask_latent / rgb_latent_scale_factor).sample
                pred_edge = vae.decode(edge_latent / rgb_latent_scale_factor).sample
                
                # Convert to numpy and resize
                pred_mask_np = pred_mask[0].permute(1, 2, 0).cpu().numpy()
                pred_mask_np = (pred_mask_np + 1) / 2 * 255  # Denormalize
                pred_mask_pil = Image.fromarray(pred_mask_np.astype(np.uint8)).convert('L')
                pred_mask_pil = pred_mask_pil.resize((w_, h_), Image.BILINEAR)
                
                pred_mask_resized = np.array(pred_mask_pil)
                
                # Load ground truth
                gt_filename = os.path.basename(input_image_path)
                gt_path = os.path.join(gt_root, gt_filename.replace('.jpg', '.png').replace('.jpeg', '.png'))
                
                if not os.path.exists(gt_path):
                    continue
                
                gt_pil = Image.open(gt_path).convert('L')
                gt_np = np.array(gt_pil, dtype=np.float64)
                gt_np /= (gt_np.max() + 1e-8)
                gt_np[gt_np > 0.5] = 1
                gt_np[gt_np != 1] = 0
                
                # Normalize prediction
                res = np.array(pred_mask_resized, dtype=np.float64)
                if res.max() == res.min():
                    res = res / 255
                else:
                    res = (res - res.min()) / (res.max() - res.min())
                
                # Binarize
                res[res > 0.5] = 1
                res[res != 1] = 0

                if gt_np.shape[:2] != res.shape[:2]:
                    res = res.T
                
                # Update metrics
                mae.update(res, gt_np)
                sm.update(res, gt_np)
                fm.update(res, gt_np)
                em.update(res, gt_np)
                wfm.update(res, gt_np)
                m_dice.update(res, gt_np)
                m_iou.update(res, gt_np)
                ber.update(res, gt_np)
                acc.update(res, gt_np)

                gt_tensor = torch.from_numpy(cv2.resize(gt_np, (512, 512))).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
                vae_loss = mse_loss(vae_decoded_img, gt_tensor)  # VAE reconstruction loss on the mask part

                total_mask_loss += mask_loss.item()
                total_vae_loss += vae_loss.item()
                total_instances += 1
            
            # Get results and log
            MAE = mae.show()
            maxf, meanf, _, _ = fm.show()
            sm_val = sm.show()
            em_val = em.show()
            wfm_val = wfm.show()
            m_dice_val = m_dice.show()
            m_iou_val = m_iou.show()
            ber_val = ber.show()
            acc_val = acc.show()
            
            mean_mask_loss = total_mask_loss / total_instances
            mean_vae_loss = total_vae_loss / total_instances

            # Log to tensorboard
            metric_prefix = f"val/{dataset_name}"
            writer.add_scalar(f"{metric_prefix}/MAE", MAE, epoch)
            writer.add_scalar(f"{metric_prefix}/maxF", maxf, epoch)
            writer.add_scalar(f"{metric_prefix}/meanF", meanf, epoch)
            writer.add_scalar(f"{metric_prefix}/wFm", wfm_val, epoch)
            writer.add_scalar(f"{metric_prefix}/Sm", sm_val, epoch)
            writer.add_scalar(f"{metric_prefix}/adpEm", em_val, epoch)
            writer.add_scalar(f"{metric_prefix}/Dice", m_dice_val, epoch)
            writer.add_scalar(f"{metric_prefix}/IoU", m_iou_val, epoch)
            writer.add_scalar(f"{metric_prefix}/Ber", ber_val, epoch)
            writer.add_scalar(f"{metric_prefix}/Acc", acc_val, epoch)
            writer.add_scalar(f"{metric_prefix}/Mask_loss", mean_mask_loss, epoch)
            writer.add_scalar(f"{metric_prefix}/VAE_loss", mean_vae_loss, epoch)

            print(f"Epoch {epoch} - {dataset_name}: MAE={MAE:.4f}, maxF={maxf:.4f}, meanF={meanf:.4f}, "
                  f"Sm={sm_val:.4f}")