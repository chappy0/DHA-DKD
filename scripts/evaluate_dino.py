import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import ViTFeatureExtractor, AutoModel

class DinoVitExtractor:
    def __init__(self, model_name="facebook/dino-vits16", device="cpu"):
        """
        Initialize the DINO-ViT feature extractor.
        Default model_name uses HuggingFace hub. Adjust if using local weights.
        """
        print(f"Loading DINO model: {model_name}...")
        self.device = device
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, image):
        """Extract DINO features for a single image."""
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state


def compute_structure_distance(features1, features2):
    """Compute structure distance between two sets of feature maps."""
    features1 = features1.mean(dim=1)  # Average over tokens
    features2 = features2.mean(dim=1)  # Average over tokens
    similarity = torch.cosine_similarity(features1, features2)
    return 1.0 - similarity.item()


def compute_structure_similarity(image1, image2, dino_extractor):
    """Compute structure similarity between two images using DINO-ViT."""
    features1 = dino_extractor.extract_features(image1)
    features2 = dino_extractor.extract_features(image2)
    distance = compute_structure_distance(features1, features2)
    return 1.0 - distance


def clean_filename(filename):
    """
    Helper function to clean prefixes/suffixes for exact matching.
    Modify this function if your generated images contain specific tags (e.g., 're_', 'out_').
    """
    base = os.path.splitext(filename)[0]
    # Example: If generated images are named 'out_001.jpg' but source is '001.jpg'
    # base = base.replace('out_', '')
    return base


def compute_folder_similarity(source_dir, generated_dir, dino_extractor):
    valid_ext = ('.jpg', '.png', '.jpeg')
    source_images = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(valid_ext)])
    generated_images = sorted([f for f in os.listdir(generated_dir) if f.lower().endswith(valid_ext)])

    results = []
    
    for src_filename in tqdm(source_images, desc="Computing DINO Similarity"):
        src_base = clean_filename(src_filename)
        found_match = False
        
        for gen_filename in generated_images:
            gen_base = clean_filename(gen_filename)
            
            if src_base == gen_base:
                img1_path = os.path.join(source_dir, src_filename)
                img2_path = os.path.join(generated_dir, gen_filename)
                
                try:
                    image1 = Image.open(img1_path).convert("RGB")
                    image2 = Image.open(img2_path).convert("RGB")
                    similarity = compute_structure_similarity(image1, image2, dino_extractor)
                    results.append((src_filename, gen_filename, similarity))
                except Exception as e:
                    print(f"Error processing {src_filename} and {gen_filename}: {e}")
                
                found_match = True
                break
        
        if not found_match:
            print(f"WARNING: No matching generated image found for {src_filename}")

    average_similarity = np.mean([x[2] for x in results]) if results else 0.0
    return results, average_similarity


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dino_extractor = DinoVitExtractor(model_name=args.model_name, device=device)

    results, average_similarity = compute_folder_similarity(
        args.source_dir, 
        args.generated_dir, 
        dino_extractor
    )

    if not results:
        print("ERROR: No valid image pairs were successfully processed.")
        return

    # Save results to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("Source_Image,Generated_Image,DINO_Similarity\n")
        for img1, img2, similarity in results:
            f.write(f"{img1},{img2},{similarity:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Similarity: {average_similarity:.4f}\n")

    print(f"\nEvaluation Complete. Average DINO Similarity: {average_similarity:.4f}")
    print(f"Detailed results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average DINO-ViT structural similarity between two folders.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the original source images.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory containing the generated images.")
    parser.add_argument("--output_file", type=str, default="dino_similarity_results.csv", help="Path to save the output CSV.")
    parser.add_argument("--model_name", type=str, default="facebook/dino-vits16", help="HuggingFace model name or local path for DINO.")
    
    args = parser.parse_args()
    main(args)