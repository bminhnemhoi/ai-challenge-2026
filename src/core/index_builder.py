import os
import json
import time
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class KeyframeIndexBuilder:
    def __init__(self, keyframes_dir: str, output_dir: str, model_name: str = "openai/clip-vit-base-patch32"):
        self.keyframes_dir = keyframes_dir
        self.output_dir = output_dir
        self.model_name = model_name
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        os.makedirs(self.output_dir, exist_ok=True)

    def scan_keyframes(self) -> List[Dict[str, Any]]:
        """
        Scans keyframes_dir for video subfolders and image files.
        """
        metadata = []
        if not os.path.exists(self.keyframes_dir):
            print(f"Error: Keyframes directory {self.keyframes_dir} does not exist!")
            return metadata

        videos = sorted([d for d in os.listdir(self.keyframes_dir) if os.path.isdir(os.path.join(self.keyframes_dir, d))])
        
        for video_id in videos:
            video_path = os.path.join(self.keyframes_dir, video_id)
            frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            for f in frame_files:
                # Extract frame index from filename, e.g., "001.jpg" -> 1
                base_name = os.path.splitext(f)[0]
                try:
                    frame_idx = int(base_name)
                except ValueError:
                    frame_idx = 0
                
                rel_path = os.path.join(video_id, f)
                abs_path = os.path.join(video_path, f)
                
                metadata.append({
                    "video_id": video_id,
                    "frame_filename": f,
                    "frame_idx": frame_idx,
                    "rel_path": rel_path,
                    "abs_path": abs_path
                })
                
        print(f"Scanned {len(videos)} videos, found {len(metadata)} total keyframes.")
        return metadata

    def build_index(self, batch_size: int = 64) -> None:
        """
        Extracts CLIP features for all scanned keyframes and saves index files.
        """
        metadata = self.scan_keyframes()
        if not metadata:
            print("No keyframes found to index.")
            return

        print(f"Loading CLIP model '{self.model_name}' on device '{self.device}'...")
        model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(self.model_name)
        model.eval()

        embeddings_list = []
        start_time = time.time()

        print("Extracting CLIP embeddings in batches...")
        for i in range(0, len(metadata), batch_size):
            batch_meta = metadata[i : i + batch_size]
            images = []
            valid_indices = []

            for idx, item in enumerate(batch_meta):
                try:
                    img = Image.open(item["abs_path"]).convert("RGB")
                    images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Warning: Failed to open image {item['abs_path']}: {e}")

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                if hasattr(outputs, "pooler_output"):
                    image_features = outputs.pooler_output
                elif hasattr(outputs, "image_embeds"):
                    image_features = outputs.image_embeds
                elif isinstance(outputs, torch.Tensor):
                    image_features = outputs
                else:
                    image_features = outputs[0]
                
                # Normalize L2 norm
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings_list.append(image_features.cpu().numpy())

            if (i // batch_size) % 10 == 0 or i + batch_size >= len(metadata):
                processed = min(i + batch_size, len(metadata))
                print(f"  Processed {processed}/{len(metadata)} frames...", flush=True)

        all_embeddings = np.vstack(embeddings_list).astype(np.float32)
        elapsed = time.time() - start_time
        print(f"Finished feature extraction in {elapsed:.2f} seconds. Output shape: {all_embeddings.shape}", flush=True)

        # Save metadata and embeddings
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        embeddings_path = os.path.join(self.output_dir, "embeddings.npy")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        np.save(embeddings_path, all_embeddings)
        print(f"Saved metadata to '{metadata_path}' and embeddings to '{embeddings_path}'.", flush=True)

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    keyframes_dir = os.path.join(base_dir, "keyframes")
    output_dir = os.path.join(base_dir, "data")
    
    builder = KeyframeIndexBuilder(keyframes_dir=keyframes_dir, output_dir=output_dir)
    builder.build_index()
