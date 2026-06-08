import os
import argparse
import torch
import clip
from PIL import Image
from tqdm import tqdm

# CLIP's maximum token limit
MAX_CONTEXT_LENGTH = 77

def truncate_text(text):
    """
    Truncate raw text to fit within the maximum context length for CLIP.
    """
    tokens = clip.tokenize([text])
    if tokens.shape[1] > MAX_CONTEXT_LENGTH:
        while tokens.shape[1] > MAX_CONTEXT_LENGTH:
            text = text[:-1]  # Iteratively truncate the text
            tokens = clip.tokenize([text])
    return text

def load_images_and_texts(image_dir, text_dir, preprocess):
    """
    Match images with their corresponding text prompts based on filenames.
    """
    image_text_pairs = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    
    for filename in os.listdir(image_dir):
        base, ext = os.path.splitext(filename)
        if ext.lower() in valid_extensions:
            image_path = os.path.join(image_dir, filename)
            
            # NOTE: Modify this matching logic if your text files have different naming conventions.
            # Example: If image is "001_output.jpg" and text is "001.txt", use base.replace("_output", "")
            text_filename = f"{base}.txt"
            text_path = os.path.join(text_dir, text_filename)
            
            if os.path.exists(text_path):
                try:
                    # Load and preprocess image
                    image = Image.open(image_path).convert("RGB")
                    image = preprocess(image)
                    
                    # Load and truncate text
                    with open(text_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        text = truncate_text(text)
                        image_text_pairs.append((image, text, filename))
                except Exception as e:
                    print(f"Error processing pair ({filename}, {text_path}): {e}")
            else:
                print(f"Warning: Text file not found for {filename}")
                
    return image_text_pairs

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print("Loading CLIP ViT-B/32 model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load matched data
    print("Pairing images and text prompts...")
    image_text_pairs = load_images_and_texts(args.image_dir, args.text_dir, preprocess)

    if not image_text_pairs:
        print("ERROR: No valid image-text pairs found. Check your directories and filename formats.")
        return

    print(f"Successfully loaded {len(image_text_pairs)} pairs. Calculating similarities...")
    
    cosine_similarities = []
    
    # Calculate similarities
    with torch.no_grad():
        for image, text, filename in tqdm(image_text_pairs, desc="Computing CLIP Score"):
            image = image.unsqueeze(0).to(device)
            
            # Extract and normalize image features
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Extract and normalize text features
            text_tokens = clip.tokenize([text]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).item()
            cosine_similarities.append((filename, text, similarity))

    # Compute average similarity
    average_similarity = sum(sim for _, _, sim in cosine_similarities) / len(cosine_similarities)

    # Save results to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("Pairwise Cosine Similarities:\n")
        f.write("-" * 50 + "\n")
        for filename, text, sim in cosine_similarities:
            f.write(f"Image: {filename} | Prompt: '{text}' -> Cosine Similarity: {sim:.4f}\n")
        
        f.write("-" * 50 + "\n")
        f.write(f"Average Cosine Similarity: {average_similarity:.4f}\n")
        
    print(f"\nEvaluation Complete. Average Cosine Similarity: {average_similarity:.4f}")
    print(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average CLIP text-image similarity.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing generated images.")
    parser.add_argument("--text_dir", type=str, required=True, help="Directory containing original text prompts (.txt).")
    parser.add_argument("--output_file", type=str, default="clip_similarity_results.txt", help="Path to save output results.")
    
    args = parser.parse_args()
    main(args)