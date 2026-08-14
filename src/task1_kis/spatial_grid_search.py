import torch
import numpy as np
from typing import List, Dict, Any, Optional

class SpatialGridSearchEngine:
    """
    Level 4 Google SOTA Spatial Grid Attention Search Engine.
    Evaluates 49-patch (7x7) spatial visual tokens per keyframe to match localized small objects
    (logos, license plates, signs at corners) directly at Stage 1 retrieval.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def compute_spatial_grid_scores(
        self,
        siglip_model: Any,
        siglip_processor: Any,
        text_query: str,
        patch_embeds_batch: torch.Tensor
    ) -> np.ndarray:
        """
        Computes Max-Patch Spatial Grid Similarity for a batch of keyframes.
        
        Args:
            siglip_model: Loaded SigLIP 2 model instance
            siglip_processor: Loaded SigLIP 2 processor instance
            text_query: Search prompt string
            patch_embeds_batch: Tensor of shape (B_images, N_patches, dim)
        
        Returns:
            np.ndarray of shape (B_images,) containing spatial patch max similarity scores.
        """
        if siglip_model is None or siglip_processor is None or patch_embeds_batch is None:
            return np.zeros(patch_embeds_batch.shape[0] if patch_embeds_batch is not None else 0, dtype=np.float32)

        try:
            inputs_txt = siglip_processor(
                text=[f"a photo of {text_query}"],
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True
            ).to(self.device)

            with torch.inference_mode():
                text_out = siglip_model.text_model(**inputs_txt)
                text_tokens = text_out.last_hidden_state  # (1, 64, dim)
                text_norm = text_tokens / text_tokens.norm(dim=-1, keepdim=True)

                # Normalize patch embeddings
                patch_norm = patch_embeds_batch / patch_embeds_batch.norm(dim=-1, keepdim=True)
                mask = (inputs_txt["input_ids"] > 0).float()

                # Calculate Token-to-Patch Cross Attention: (B_images, 64, N_patches)
                # MaxSim: For each text token, find the maximum matching spatial patch in the image
                scale = float(torch.exp(siglip_model.logit_scale.detach().cpu()).item()) if hasattr(siglip_model, "logit_scale") else 90.0
                bias = float(torch.exp(siglip_model.logit_bias.detach().cpu()).item()) if hasattr(siglip_model, "logit_bias") else -10.0

                spatial_scores = []
                for i in range(patch_norm.shape[0]):
                    p_i = patch_norm[i]  # (N_patches, dim)
                    sim_mat = torch.matmul(text_norm, p_i.T)  # (1, 64, N_patches)
                    maxsim_per_tok = sim_mat.max(dim=-1).values  # (1, 64)
                    raw_maxsim = (maxsim_per_tok * mask).sum(dim=-1) / mask.sum(dim=-1)
                    score = float(torch.sigmoid(scale * raw_maxsim + bias).item())
                    spatial_scores.append(score)

                return np.array(spatial_scores, dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Spatial Grid Search warning: {e}")
            return np.zeros(patch_embeds_batch.shape[0], dtype=np.float32)
