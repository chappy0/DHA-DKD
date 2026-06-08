"""
fcdiffusion_test.py
Evaluation and benchmarking script for the original Teacher diffusion model.
Measures generation quality, inference latency, and peak GPU memory allocation.
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from fcdiffusion.dataset import TestDataset
from ldm.util import instantiate_from_config

# Disable CURL CA bundle to prevent potential SSL issues during local testing
os.environ['CURL_CA_BUNDLE'] = ''


def resize_batch_tensors(batch, target_size=(1024, 1024)):
    """
    Dynamically intercepts DataLoader outputs and forces the image tensors 
    to be interpolated to the target resolution.
    """
    for key in ["image", "jpg", "hint", "control"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            tensor = batch[key]
            if tensor.ndim == 4:
                if tensor.shape[-1] == 3:  # Format: [B, H, W, C]
                    tensor = tensor.permute(0, 3, 1, 2) 
                    tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
                    tensor = tensor.permute(0, 2, 3, 1) 
                else:  # Format: [B, C, H, W]
                    tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
            batch[key] = tensor
    return batch


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    """Instantiate the teacher model from config and load the checkpoint."""
    print(f"Loading Teacher model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    if verbose:
        if len(m) > 0: print("Missing keys:", m)
        if len(u) > 0: print("Unexpected keys:", u)
        
    model.to(device)
    model.eval()
    return model


def traverse_images_and_texts(directory):  
    """Match source images with corresponding text prompts in a directory."""
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg')  
    image_files, text_contents = [], []
    for root, dirs, files in os.walk(directory):  
        for file in sorted(files):  
            if file.lower().endswith(IMAGE_EXTENSIONS):  
                image_files.append(os.path.join(root, file))  
            elif file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)  
                with open(file_path, 'r', encoding='utf-8') as f:  
                    text_contents.append(f.read().strip())  
    return image_files, text_contents


def main(args):
    target_resolution = (args.resolution, args.resolution)
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Configuration and Model
    config = OmegaConf.load(args.config_path)
    model = load_model_from_config(config, args.ckpt_path, device)
    
    image_files, text_contents = traverse_images_and_texts(args.test_dir)
    print(f"Found {len(image_files)} test samples. Target Resolution: {target_resolution}")

    latencies = []
    peak_memories = []

    for idx, (img_path, prompt) in enumerate(zip(image_files, text_contents)):
        print(f"\n--- Processing Sample {idx+1}/{len(image_files)} ---")
        _, file_name = os.path.split(img_path)
        output_img_path = os.path.join(args.output_dir, "t_" + file_name)

        dataset = TestDataset(img_path, prompt, res_num=1)
        dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)
        batch = next(iter(dataloader))
        
        batch = resize_batch_tensors(batch, target_size=target_resolution)

        # Clear environment and reset peak memory stats for accurate measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats() 
        torch.cuda.synchronize()
        start_time = time.time()

        try:
            with torch.no_grad():
                # Force N=1 to prevent underlying redundant memory allocation
                log = model.log_images(batch, N=1, ddim_steps=args.steps)

            # Stop timer and record VRAM
            torch.cuda.synchronize()
            latency = time.time() - start_time
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
            latencies.append(latency)
            peak_memories.append(peak_vram_gb)
            
            print(f"✅ Success! Latency: {latency:.2f}s | Peak VRAM: {peak_vram_gb:.2f} GB")

            # Save generated image
            sample = log['samples'].squeeze().permute(1, 2, 0)
            sample = torch.clamp(sample, -1, 1)
            img_np = ((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)
            Image.fromarray(img_np).save(output_img_path)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM (Out of Memory) at resolution {target_resolution}! Teacher model failed.")
                torch.cuda.empty_cache()
            else:
                raise e

    # Generate Performance Report
    print("\n" + "="*50)
    print(f"📊 TEACHER PERFORMANCE REPORT FOR {target_resolution[0]}x{target_resolution[1]} (NFE: {args.steps})")
    print("="*50)
    if len(latencies) > 0:
        # Exclude the first run to avoid cold-start bias
        valid_latencies = latencies[1:] if len(latencies) > 1 else latencies
        avg_latency = np.mean(valid_latencies)
        avg_vram = np.mean(peak_memories)
        print(f"Successful Runs  : {len(latencies)} / {len(image_files)}")
        print(f"Average Latency  : {avg_latency:.2f} seconds/image")
        print(f"Peak VRAM Usage  : {avg_vram:.2f} GB")
    else:
        print("🚨 ALL RUNS FAILED DUE TO OOM. The Teacher model cannot run this resolution.")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for the Teacher Diffusion Model.")
    parser.add_argument("--config_path", type=str, default="configs/model_config.yaml", help="Path to teacher config.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to teacher checkpoint.")
    parser.add_argument("--test_dir", type=str, default="./datasets/test_sub", help="Directory containing test images and prompts.")
    parser.add_argument("--output_dir", type=str, default="./datasets/teacher_test_res", help="Directory to save generated images.")
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution (e.g., 512, 768, 1024).")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps (NFE).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (Must be 1 for accurate latency/VRAM benchmarking).")
    
    args = parser.parse_args()
    main(args)