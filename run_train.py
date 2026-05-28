import torch
from torch import nn
import torch.nn.functional as F
import  os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
import numpy as np
import cv2
import PIL
from PIL import Image
import glob
from diffusers import (
    DDPMScheduler,
    UniDiffuserModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer, ViTImageProcessor, ViTForImageClassification
from utils.dataset_strategy import get_loader
from utils.test_data import test_dataset
from utils.saliency_metric import (
    cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, 
    cal_dice, cal_iou, cal_ber, cal_acc, HCEMeasure
)
from utils.config import diste1, diste2, diste3, diste4, disvd
from utils.utils import *
from utils.image_util import cutmix, resize_res
from torch.cuda import amp
from safetensors.torch import save_model
from skimage.morphology import skeletonize
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# Create timestamped log directory
log_base_dir = './logs/DiT'
os.makedirs(log_base_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(log_base_dir, timestamp)
writer = SummaryWriter(log_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=90, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument("--pretrained_model_name_or_path", type=str, default='/path/to/sd-turbo/')
parser.add_argument("--dataset_path", type=str, default='/path/to/DIS5K/')


opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
print(f'TensorBoard logs will be saved to: {log_dir}')
print(f'View logs with: tensorboard --logdir=./logs\n')

# Test datasets for validation
test_datasets = {
    'DIS-VD': "DIS5K/DIS-VD",
}

def compute_validation_metrics(dit_model, vit_model, vit_proj, test_datasets_dict, dataset_path, vae, 
                               text_encoder, tokenizer, noise_scheduler,
                               rgb_latent_scale_factor, weight_dtype, epoch, writer, opt):
    """
    Generate predictions using the model and compute validation metrics, then log to tensorboard
    """
    dit_model.eval()
    vit_model.eval()
    vit_proj.eval()
    device = next(dit_model.parameters()).device
    
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
            # hce = HCEMeasure()

            for input_image_path in tqdm(rgb_filename_list[:num_samples], 
                                         desc=f"Validating {dataset_name}"):
                # Load and process image similar to run_inference.py
                input_image_pil = Image.open(input_image_path)
                w_, h_ = input_image_pil.size
                img_resize = resize_res(input_image_pil, resolution=1024)
                
                input_image = img_resize.convert("RGB")
                image = np.array(input_image)
                rgb = np.transpose(image, (2, 0, 1))
                rgb_norm = rgb / 255.0 * 2.0 - 1.0
                rgb_tensor = torch.from_numpy(rgb_norm).to(device)
                rgb_tensor = rgb_tensor.float().unsqueeze(0)
                
                # Encode to latent space
                with torch.no_grad():
                    h = vae.encoder(rgb_tensor.to(weight_dtype))
                    moments = vae.quant_conv(h)
                    mean, logvar = torch.chunk(moments, 2, dim=1)
                    rgb_latents = mean * rgb_latent_scale_factor
                
                # Create input for model (use zero mask and edge as input)
                mask_latents = torch.zeros_like(rgb_latents)
                edge_latents = torch.zeros_like(rgb_latents)
                
                # Multi-scale conditions
                rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents = \
                    generate_multi_scale_latents(rgb_tensor, rgb_latent_scale_factor, vae, weight_dtype, opt)
                
                # Concatenate for model input
                unified_latents = torch.cat([mask_latents, edge_latents], dim=0)
                
                # Create noise and add it
                noise = pyramid_noise_like(unified_latents, discount=0.8)
                timesteps = torch.tensor([999], device=device).long()
                noisy_unified_latents = noise_scheduler.add_noise(unified_latents, noise, timesteps.repeat(2))
                
                # Encode text embedding for empty prompt
                prompt = ""
                text_inputs = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device)
                empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)
                batch_empty_text_embed = empty_text_embed.repeat((2, 1, 1))

                # Extract ViT image features for encoder_hidden_states
                vit_inputs = vit_processor(images=img_resize, return_tensors="pt").to(device)
                vit_outputs = vit_model.vit(**vit_inputs)
                vit_embeds = vit_proj(vit_outputs.last_hidden_state.to(weight_dtype))
                batch_vit_embeds = vit_embeds.repeat(2, 1, 1)

                # Batch discriminative embedding
                discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device=device)
                BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1)
                dit_input = torch.cat([rgb_latents.repeat(2, 1, 1, 1), noisy_unified_latents], dim=1)
                
                bsz = rgb_tensor.shape[0]

                dummy_image_embeds = torch.zeros((bsz*2, 1, 512), device=text_encoder.device)
                dummy_prompt_embeds = torch.zeros((bsz*2, 77, 64), device=text_encoder.device)
                
                noise_pred, img_clip_out, text_out = dit_model(
                                                    latent_image_embeds=dit_input,
                                                    image_embeds=dummy_image_embeds,
                                                    prompt_embeds=dummy_prompt_embeds,
                                                    timestep_img=timesteps.repeat(bsz*2),
                                                    timestep_text=timesteps.repeat(bsz*2),
                                                    encoder_hidden_states=batch_vit_embeds,
                                                )
                
                # Denoise one step
                x_denoised = noise_scheduler.step(noise_pred, timesteps, noisy_unified_latents, 
                                                 return_dict=True).prev_sample
                
                # Split into mask and edge
                mask_latent, edge_latent = torch.chunk(x_denoised, 2, dim=0)
                
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
                
                # Load or compute skeleton
                # ske_path = gt_path.replace("/gt/", "/ske/")
                # if os.path.exists(ske_path):
                #     ske_ary = cv2.imread(ske_path, cv2.IMREAD_GRAYSCALE)
                #     ske_ary = ske_ary > 128
                # else:
                #     ske_ary = skeletonize(gt_np > 0.5)
                
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
                # hce.step(pred=res, gt=gt_np, gt_ske=ske_ary)
            
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
            # hce_val = hce.get_results()["hce"]

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
            # writer.add_scalar(f"{metric_prefix}/HCE", hce_val, epoch)

            print(f"Epoch {epoch} - {dataset_name}: MAE={MAE:.4f}, maxF={maxf:.4f}, meanF={meanf:.4f}, "
                  f"Sm={sm_val:.4f}")

# build models
text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder='text_encoder')
vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder='vae')
dit = UniDiffuserModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="unet")

vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to('cuda')
vit_model.train()



text_encoder.requires_grad_(False)
vae.requires_grad_(False)
# unet = replace_unet_conv_in(unet)
# unet = update_att_weights(unet) 
dit.train().cuda()

vit_projection = nn.Linear(vit_model.config.hidden_size, 64).to('cuda')


noise_scheduler = DDPMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder='scheduler')##{'clip_sample_range', 'rescale_betas_zero_snr', 'sample_max_value', 'timestep_spacing', 'thresholding', 'dynamic_thresholding_ratio'} 
noise_scheduler.set_timesteps(1, device="cuda")
noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()
tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path,subfolder='clip_tokenizer')


# params, params_class_embedding = [], []
# for name, param in unet.named_parameters():
#     if 'class_embedding' in name:
#         params_class_embedding.append(param)
#     else:
#         params.append(param)        
generator_optimizer = torch.optim.Adam(list(dit.parameters()) + list(vit_model.parameters()) + list(vit_projection.parameters()), lr=opt.lr_gen)

# load data
image_root = f'{opt.dataset_path}/DIS-TR/im/'
gt_root = f'{opt.dataset_path}/DIS-TR/gt/'
edge_root = f'{opt.dataset_path}/DIS-TR/contour/'

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
print(total_step)

rgb_latent_scale_factor = 0.18215
weight_dtype = torch.float32
text_encoder.to('cuda', dtype=weight_dtype)
vae.to('cuda', dtype=weight_dtype)

mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1] 
scaler = amp.GradScaler(enabled=True)


for epoch in range(1, opt.epoch+1):
    dit.train()
    vit_projection.train()
    vit_model.train()
    loss_record = AvgMeter()

    epoch_loss1 = 0.0
    epoch_loss2 = 0.0
    epoch_loss = 0.0

    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()

            rgb, label, edge, box = pack

            rgb = rgb.cuda().to(weight_dtype) 
            label=label.unsqueeze(1).repeat(1,3,1,1).cuda().to(weight_dtype)    # Conversion from grayscale to rgb, out: [4, 3, 1024, 1024] 
            edge=edge.unsqueeze(1).repeat(1,3,1,1).cuda().to(weight_dtype) 
            box=box.unsqueeze(1).repeat(1,3,1,1).cuda().to(weight_dtype) 

            bsz = rgb.shape[0]
            assert bsz % 2 == 0, "Batch size must be even"

            rgb_chunks = rgb.chunk(bsz // 2, dim=0)     # Splits into bsz // 2 number of chunks (each of size 2) for cutmix     
            label_chunks = label.chunk(bsz // 2, dim=0)    
            edge_chunks = edge.chunk(bsz // 2, dim=0)      
            box_chunks = box.chunk(bsz // 2, dim=0) 

            # apply cutmix within each chunk
            rgbs, labels, edges = [], [], []
            for rgb_, label_, edge_, box_ in zip(rgb_chunks, label_chunks, edge_chunks, box_chunks):
                mixed_rgb, mixed_label, mixed_edge = cutmix(rgb_, label_, edge_, box_)
                rgbs.append(mixed_rgb)
                labels.append(mixed_label)
                edges.append(mixed_edge)
            
            rgb_mix = torch.cat(rgbs, dim=0)    # Convert back to the original batch size by concatenation of cutmixed images
            label_mix = torch.cat(labels, dim=0)
            edge_mix = torch.cat(edges, dim=0)

            # map pixels into latent space
            h_batch = vae.encoder(torch.cat((rgb_mix, label_mix, edge_mix), dim=0).to(weight_dtype))    # Processes rgb_mix, label_mix and edge_mix as a single batch. Concat is done on dim=0 so the output will be [bsz*3, 8, 128, 128]

            moments_batch = vae.quant_conv(h_batch) # Also of shape [12, 8, 128, 128], this is just pointwise convolution
            mean_batch, logvar_batch = torch.chunk(moments_batch, 2, dim=1)     # Split of latent space vector into means and logvars
            batch_latents = mean_batch * rgb_latent_scale_factor
            rgb_latents, mask_latents, edge_latents = torch.chunk(batch_latents, 3, dim=0)  # Unbatch the rgb, mask and edge latents into separate tensors
            
            # generate multi-scale conditions
            rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents = generate_multi_scale_latents(rgb_mix, rgb_latent_scale_factor, vae, weight_dtype, opt)   # Generates rgb latents with different sizes ([64x64, 32x32, 16x16] respectively)
            
            # concat mask and edge latents along batch dimension 
            unified_latents = torch.cat((mask_latents,edge_latents), dim=0) # Batch concat of mask and edge latents

            # create multi-resolution noise
            noise = pyramid_noise_like(unified_latents, discount=0.8)   # Add noise 

            # set timestep to T
            timesteps = torch.tensor([999], device="cuda").long()
            
            # add noise 
            noisy_unified_latents = noise_scheduler.add_noise(unified_latents, noise, timesteps.repeat(bsz*2))
 
            # encode text embedding for empty prompt
            prompt = ""
            text_inputs =tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)   # After decoding would be two tokens: '<|startoftext|><|endoftext|>' 
            empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype) # Convert tokens to embeddings using text encoder
            batch_empty_text_embed = empty_text_embed.repeat((noisy_unified_latents.shape[0], 1, 1))    # Repeat text embedding so that it matches batch size  

            # Extract ViT image features for encoder_hidden_states
            vit_images = []
            for j in range(rgb_mix.shape[0]):
                img = rgb_mix[j].detach().cpu()
                img = ((img + 1) / 2).clamp(0, 1)
                img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                vit_images.append(Image.fromarray(img))
            vit_inputs = vit_processor(images=vit_images, return_tensors="pt").to('cuda')
            vit_outputs = vit_model.vit(**vit_inputs)
            vit_embeds = vit_projection(vit_outputs.last_hidden_state.to(weight_dtype))
            batch_vit_embeds = vit_embeds.repeat_interleave(2, dim=0)

            # batch discriminative embedding
            discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device='cuda')
            BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)
            # dit_input = torch.cat([rgb_latents.repeat(2,1,1,1),noisy_unified_latents], dim=1)  # Expands rgb latents so that it matches mask and edge latents batch dim and concats everything in channel dimension, out: [8, 8, 128, 128]  
            dit_input = noisy_unified_latents
            
            # predict the noise 
            # noise_pred = unet(unet_input, timesteps.repeat(bsz*2), encoder_hidden_states=batch_empty_text_embed, class_labels = BDE,\
            #                    rgb_token=[rgb_latents.repeat(2,1,1,1) , rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents],\
            #     ).sample 
            dummy_image_embeds = torch.zeros((bsz*2, 1, 512), device=text_encoder.device)
            dummy_prompt_embeds = torch.zeros((bsz*2, 77, 64), device=text_encoder.device)

            noise_pred, img_clip_out, text_out = dit(
                                                    latent_image_embeds=dit_input,
                                                    image_embeds=dummy_image_embeds,
                                                    prompt_embeds=dummy_prompt_embeds,
                                                    timestep_img=timesteps.repeat(bsz*2),
                                                    timestep_text=timesteps.repeat(bsz*2),
                                                    encoder_hidden_states=batch_vit_embeds,
                                                )
            
            # one-step denoising process
            x_denoised = noise_scheduler.step(noise_pred, timesteps, noisy_unified_latents, return_dict=True).prev_sample
            
            # split mask and edge latents along batch dimension 
            mask_latent, edge_latent = torch.chunk(x_denoised, 2, dim=0)

            loss1 = F.mse_loss(mask_latent.cuda().to(weight_dtype),mask_latents.cuda().to(weight_dtype), reduction="mean")
            loss2 = F.mse_loss(edge_latent.cuda().to(weight_dtype),edge_latents.cuda().to(weight_dtype), reduction="mean")
            loss = loss1 + loss2

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss += loss.item()


            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}, mask loss:{:.4f}, edge loss:{:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), loss1, loss2))
                    
    mean_epoch_loss1 = epoch_loss1 / loss_record.num
    mean_epoch_loss2 = epoch_loss2 / loss_record.num
    mean_epoch_loss = epoch_loss / loss_record.num
    
    writer.add_scalar('mask_loss', mean_epoch_loss1, epoch)
    writer.add_scalar('edge_loss', mean_epoch_loss2, epoch)
    writer.add_scalar('total_loss', mean_epoch_loss, epoch)

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    # Compute validation metrics every 5 epochs
    if epoch % 1 == 0:
        print(f"\nComputing validation metrics for epoch {epoch}...")
        compute_validation_metrics(dit, vit_model, vit_projection, test_datasets, opt.dataset_path, vae, 
                                   text_encoder, tokenizer, noise_scheduler,
                                   rgb_latent_scale_factor, weight_dtype, epoch, writer, opt)
        print(f"Validation metrics logged for epoch {epoch}\n")

    # save checkpoints every 10 epochs
    if epoch % 5 == 0: 
        save_path = f'../saved_model/DiffDIS_DiT/Model_{epoch}/unet/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_model(dit, f'{save_path}diffusion_pytorch_model.safetensors')
        optimizer_state = generator_optimizer.state_dict()
        torch.save(optimizer_state, f'../saved_model/DiffDIS_DiT/Model_{epoch}/generator_optimizer.pth')
