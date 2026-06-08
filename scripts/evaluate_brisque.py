import os
import argparse
import torch
import pyiqa
from tqdm import tqdm

def main(args):
    """
    Calculate the average BRISQUE score for all images in a given directory.
    Lower BRISQUE scores indicate better perceptual image quality.
    """
    # 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found at: {args.input_dir}")
        return

    # 3. Initialize BRISQUE evaluator
    try:
        brisque_metric = pyiqa.create_metric('brisque', device=device)
        print("Successfully created BRISQUE metric. Score range is [0, 100]. Lower is better.")
    except Exception as e:
        print(f"Error creating IQA metric: {e}")
        return

    # 4. Gather image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_filenames = sorted([
        f for f in os.listdir(args.input_dir) 
        if f.lower().endswith(valid_extensions)
    ])
    
    if not image_filenames:
        print(f"ERROR: No image files found in '{args.input_dir}'.")
        return
        
    print(f"Found {len(image_filenames)} images to evaluate.")

    # 5. Compute BRISQUE score for each image
    total_brisque_score = 0.0
    processed_count = 0

    for filename in tqdm(image_filenames, desc="Calculating BRISQUE scores"):
        image_path = os.path.join(args.input_dir, filename)
        
        try:
            score = brisque_metric(image_path)
            total_brisque_score += score.item()
            processed_count += 1
        except Exception as e:
            print(f"\nWARNING: Could not process file '{filename}'. Error: {e}")
            continue
    
    # 6. Compute and display average score
    if processed_count > 0:
        average_score = total_brisque_score / processed_count
        print("\n" + "="*40)
        print("        BRISQUE Evaluation Summary")
        print("="*40)
        print(f"Evaluated {processed_count} / {len(image_filenames)} images successfully.")
        print(f"Average BRISQUE Score: {average_score:.4f}")
        print("(Note: Lower score indicates better quality)")
        print("="*40)
    else:
        print("No images were successfully processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the average BRISQUE score for a folder of images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing images.")
    
    args = parser.parse_args()
    main(args)