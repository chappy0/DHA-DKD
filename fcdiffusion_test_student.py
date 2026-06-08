import os
import random
import argparse
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fcdiffusion.dataset import TestDataset
from ldm.util import instantiate_from_config


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


def load_student_model_from_distill_ckpt(config, ckpt_path, device=torch.device("cuda"), verbose=True):
    """
    Specifically designed for loading the student model from checkpoints saved by the DecoupledDistiller.
    It automatically handles the "student_model." prefix.
    """
    print(f"Loading distilled student model from: {ckpt_path}")
    
    # 1. load checkpoint
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    
    # get state_dict
    full_state_dict = pl_sd["state_dict"]
    
    # 2. remove prefix
    student_state_dict = OrderedDict()
    for k, v in full_state_dict.items():
        if k.startswith("student_model."):
            new_key = k[len("student_model."):]
            student_state_dict[new_key] = v
    
    if not student_state_dict:
        raise KeyError("Could not find weights with 'student_model.' prefix in the checkpoint. Please check the checkpoint file.")

    # 3. Instantiate and load the student model
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(student_state_dict, strict=False)
    
    if len(m) > 0 and verbose:
        print("Missing keys in student model:")
        print(m)
    if len(u) > 0 and verbose:
        print("Unexpected keys in student model:")
        print(u)
        
    model.to(device)
    model.eval()
    print("Student model loaded successfully.")
    return model


def is_image_file(filename):  
    """Determine if the file is a image file"""  
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']  
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)  
  

def is_text_file(filename):  
    """Determine if the file is a text file"""  
    return filename.lower().endswith('.txt')  
  

def traverse_images_and_texts(directory):  
    """Traverse the image files"""  
    image_files = []  
    text_contents = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if is_image_file(file):  
                image_files.append(os.path.join(root, file))  
            elif is_text_file(file):
                file_path = os.path.join(root, file)  
                with open(file_path, 'r', encoding='utf-8') as f:  
                    content = f.read()  
                    text_contents.append((content))  
    return image_files, text_contents


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference script for DKD Student Model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the distillation checkpoint")
    parser.add_argument("--routing", type=str, default="dynamic", choices=["dynamic", "random", "static"], help="DHA routing strategy")
    parser.add_argument("--sparsity", type=float, default=0.5, help="DHA sparsity ratio")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--out_dir", type=str, default="datasets/user_study_ours/", help="Output directory for generated images")
    parser.add_argument("--data_dir", type=str, default="datasets/user_study_30", help="Directory containing test images and prompts")
    parser.add_argument("--yaml_path", type=str, default="configs/student_model_config.yaml", help="Path to student config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility") 
    args = parser.parse_args()


    set_seed(args.seed)

    os.environ["DHA_ROUTING"] = args.routing
    os.environ["DHA_SPARSITY"] = str(args.sparsity)

    print("\n" + "="*40)
    print("--- Inference Configuration ---")
    print(f"Routing Strategy : {os.environ['DHA_ROUTING']}")
    print(f"Sparsity Ratio   : {os.environ['DHA_SPARSITY']}")
    print(f"DDIM Steps       : {args.ddim_steps}")
    print(f"Checkpoint       : {args.ckpt_path}")
    print(f"Output Directory : {args.out_dir}")
    print(f"Random Seed      : {args.seed}") 
    print("="*40 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(args.yaml_path)
    model = load_student_model_from_distill_ckpt(config, args.ckpt_path, device)
    model.eval()

    image_files, text_contents = traverse_images_and_texts(args.data_dir)  
    if not image_files:
        print(f"Warning: No images found in {args.data_dir}")
        exit()

    test_res_num = 1
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for image_file, text_content in zip(image_files, text_contents):  
        test_img_path, target_prompt = image_file, text_content
        _, img_basename = os.path.split(test_img_path) 
        output_img_path = os.path.join(args.out_dir, "e_" + img_basename)

        dataset = TestDataset(test_img_path, target_prompt, test_res_num)
        dataloader = DataLoader(dataset, num_workers=0, batch_size=16, shuffle=False)
        
        for step, batch in enumerate(dataloader):          
            with torch.no_grad():
                log = model.log_images(batch, ddim_steps=args.ddim_steps)
                
            if step == 0 and 'reconstruction' in log:
                reconstruction = log['reconstruction'].squeeze()
                reconstruction = reconstruction.permute(1, 2, 0)
                reconstruction = torch.clamp(reconstruction, -1, 1)

            if 'samples' in log:
                sample = log['samples'].squeeze()
                sample = sample.permute(1, 2, 0)
                sample = torch.clamp(sample, -1, 1)
                Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(output_img_path)
                print(f"Saved: {output_img_path}")