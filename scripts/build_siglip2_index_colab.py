"""
Google Colab Fast Index Builder for Official Google SigLIP 2 Model
Model: google/siglip2-base-patch16-224

Usage in Google Colab:
1. Set runtime to GPU (T4 / A100)
2. Run:
   !pip install -q transformers torch pillow huggingface_hub tqdm
   !python scripts/build_siglip2_index_colab.py
"""

import os
import io
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

MODEL_NAME = "google/siglip2-base-patch16-224"
DATA_DIR = "./data"
KEYFRAMES_DIR = "./keyframes"
BATCH_SIZE = 128

def build_siglip2_index():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing SigLIP 2 Index Builder on device: {device}...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Indexing {len(metadata)} keyframe images...")

    embeddings = []
    batch_images = []
    
    with torch.inference_mode():
        for i, item in enumerate(tqdm(metadata, desc="Extracting SigLIP 2 Embeddings")):
            img_path = os.path.join(KEYFRAMES_DIR, item["video_id"], item["frame_filename"])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                # Fallback blank image if missing
                img = Image.new("RGB", (224, 224), color=0)
            
            batch_images.append(img)

            if len(batch_images) == BATCH_SIZE or i == len(metadata) - 1:
                inputs = processor(images=batch_images, return_tensors="pt").to(device)
                outputs = model.get_image_features(**inputs)
                vecs = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
                vecs = vecs / vecs.norm(dim=-1, keepdim=True)
                embeddings.append(vecs.cpu().numpy().astype(np.float32))
                batch_images = []

    final_matrix = np.vstack(embeddings)
    output_path = os.path.join(DATA_DIR, "embeddings_siglip.npy")
    np.save(output_path, final_matrix)
    print(f"✅ SigLIP 2 Index Build Complete! Saved matrix {final_matrix.shape} to {output_path}")

if __name__ == "__main__":
    build_siglip2_index()
