Hybrid Attention Mechanism with Decoupled Distillation


Note: This is the official implementation of the manuscript "Hybrid Attention Mechanism with Decoupled Distillation: Enhancing Efficiency and Fidelity in Diffusion Models", currently submitted to The Visual Computer.

📖 Introduction

In the realm of image-to-image translation, diffusion models often face a trade-off between generation quality and computational efficiency. To address this, we introduce:

Dynamic Hybrid Attention (DHA): A mechanism that synergizes the high-fidelity local modeling of Self-Attention with the efficient global context aggregation of External Attention.

Decoupled Knowledge Distillation (DKD): A training framework designed to effectively train this heterogeneous architecture.

Our approach achieves a 1.3x speedup and reduces peak GPU memory by 8% during inference, while maintaining nearly indistinguishable quality and full controllability compared to the original model.

🚀 Quick Start Guide

1. Environment Setup

We recommend creating and activating an isolated virtual environment using Conda.

# 1. Create and activate the Conda environment
conda create -n DKD python=3.8
conda activate DKD

# 2. Install PyTorch and related dependencies (adjust according to your CUDA version)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# 3. Install all other required libraries
pip install -r requirements.txt


2. Data & Pre-trained Model Preparation

Dataset

Image-text pairs are required for training. You can prepare a dataset like LAION Aesthetics 6.5+ and organize it using the following structure:

DKD/
└── datasets/
    └── training_data/
        ├── image/       # Source images
        │   ├── 000000000.jpg
        │   └── ...
        └── prompt/      # Corresponding text prompts
            ├── 000000000.txt
            └── ...


Pre-trained Models

The following two pre-trained models are required for distillation training:

Stable Diffusion v2-1-base:

Download: HuggingFace Link

Target Path: ./models/v2-1_512-ema-pruned.ckpt

OpenCLIP:

Download: HuggingFace Link

Target Path: ./models/CLIP-ViT-H-14/open_clip_pytorch_model.bin

Ensure your ./models directory structure is as follows:

DKD/
└── models/
    ├── v2-1_512-ema-pruned.ckpt
    └── CLIP-ViT-H-14/
        └── open_clip_pytorch_model.bin


3. Model Distillation Training (DKD)

To train the efficient student model using our Decoupled Knowledge Distillation framework:

Configuration: Open fcdffusion_distill_final.py and update the following paths:

teacher_model_path: Path to the teacher model's checkpoint.

dataset_path: Root directory of your dataset.

output_path: Directory to save student weights.

Run Training:

python fcdffusion_distill_final.py


4. Model Testing & Validation

Use fcdiffusion_student_test.py to evaluate the trained DHA-based student model.

Configuration: Open fcdiffusion_student_test.py:

student_model_path: Path to the trained student weights.

input_image_path: Source image for testing.

output_dir: Directory for results.

Run Inference:

python fcdiffusion_student_test.py


📌 Citation

If you find this code useful for your research, please consider citing our paper:

@article{Zhu2025Hybrid,
  title={Hybrid Attention Mechanism with Decoupled Distillation: Enhancing Efficiency and Fidelity in Diffusion Models},
  author={Zhu, Aihua and Su, Rui and Zhao, Qingling and Feng, Li},
  journal={The Visual Computer (Under Review)},
  year={2025}
}
