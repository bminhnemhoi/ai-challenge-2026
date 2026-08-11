import os
import json
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Optional

class TextualKISRetriever:
    """
    Search Engine for Textual Known Item Search (Task 1).
    Performs CLIP text-to-image similarity search over indexed keyframes.
    """
    def __init__(
        self,
        data_dir: str,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.metadata_path = os.path.join(data_dir, "metadata.json")
        self.embeddings_path = os.path.join(data_dir, "embeddings.npy")
        
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        
        self._is_loaded = False

    def load_index_and_model(self):
        """Loads metadata, vector embeddings, and initialized CLIP model."""
        if self._is_loaded:
            return

        if not os.path.exists(self.metadata_path) or not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"Index files not found in '{self.data_dir}'. Please run `src/index_builder.py` first!"
            )

        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Loading embeddings from {self.embeddings_path}...")
        self.embeddings = np.load(self.embeddings_path) # Shape: (N, D)

        print(f"Loading CLIP model '{self.model_name}' on '{self.device}'...")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()

        self._is_loaded = True
        print(f"Retriever initialized with {len(self.metadata)} indexed keyframes.")

    def search(
        self,
        query: str,
        top_k: int = 100,
        nms_frame_gap: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes Textual KIS search for a natural language text query.
        
        Args:
            query: Natural language text query in English or Vietnamese.
            top_k: Number of predictions to return (max 100 for AIC competition).
            nms_frame_gap: Frame index gap threshold to suppress consecutive duplicate frames.
            
        Returns:
            List of dicts containing video_id, frame_idx, score, rel_path.
        """
        if not self._is_loaded:
            self.load_index_and_model()

        # Clean query text processing
        clean_query = query.strip()
        
        # Check if query has non-ASCII (e.g., Vietnamese) characters for translation
        en_query = None
        has_non_ascii = any(ord(char) > 127 for char in clean_query)
        if has_non_ascii:
            try:
                from deep_translator import GoogleTranslator
                translated = GoogleTranslator(source="auto", target="en").translate(clean_query)
                if translated and translated.strip():
                    en_query = translated.strip()
            except Exception as e:
                print(f"Translation warning: {e}")
                en_query = None

        # Dynamic Color & Attribute Dictionary with Competing Variants
        color_map = {
            "đỏ": ("red", ["yellow golden", "white silver", "blue", "black", "green"]),
            "vàng": ("yellow golden", ["red", "white silver", "blue", "black", "green"]),
            "trắng": ("white silver", ["yellow golden", "red", "black", "blue"]),
            "đen": ("black", ["white silver", "yellow golden", "red"]),
            "xanh": ("green blue", ["red", "yellow golden", "white silver"]),
            "xanh lá": ("green", ["red", "yellow golden", "blue"]),
            "xanh lá cây": ("green", ["red", "yellow golden", "blue"]),
            "xanh dương": ("blue", ["red", "yellow golden", "green"]),
            "xanh nước biển": ("blue", ["red", "yellow golden", "green"]),
            "tím": ("purple", ["yellow golden", "red", "white silver"]),
            "cam": ("orange", ["blue", "white silver", "green"]),
            "hồng": ("pink", ["yellow golden", "white silver", "blue"])
        }

        # Dynamic Action & Pose Dictionary
        pose_map = {
            "ngồi": ("sitting seated on chair ground pose", ["standing upright pose", "walking running pose"]),
            "đang ngồi": ("sitting seated on chair ground pose", ["standing upright pose", "walking running pose"]),
            "đứng": ("standing upright pose", ["sitting seated pose", "lying down pose"]),
            "đang đứng": ("standing upright pose", ["sitting seated pose", "lying down pose"]),
            "đỗ": ("parked stationary vehicle", ["moving fast driving vehicle"]),
            "đang đỗ": ("parked stationary vehicle", ["moving fast driving vehicle"]),
            "chạy": ("running moving fast", ["sitting seated pose", "standing stationary pose"]),
            "đang chạy": ("running moving fast", ["sitting seated pose", "standing stationary pose"])
        }

        # Cultural & General Noun Dictionary
        cultural_nouns = {
            "con lân": "lion dance costume performance barongsai",
            "múa lân": "lion dance costume performance barongsai",
            "lân": "lion dance costume performance barongsai",
            "áo dài": "vietnamese traditional ao dai dress",
            "cô gái": "woman girl female presenter",
            "bánh chưng": "vietnamese square sticky rice cake",
            "bánh tét": "vietnamese cylindrical sticky rice cake"
        }

        low_clean = clean_query.lower()
        
        # Detect color, pose, and subject terms dynamically
        detected_color_en = None
        competing_colors_en = []
        for c_vi, (c_en, c_comp) in color_map.items():
            if f"màu {c_vi}" in low_clean or f" {c_vi}" in low_clean or low_clean.startswith(c_vi) or low_clean.endswith(c_vi):
                detected_color_en = c_en
                competing_colors_en = c_comp
                break

        detected_pose_en = None
        competing_poses_en = []
        for p_vi, (p_en, p_comp) in pose_map.items():
            if f" {p_vi}" in low_clean or low_clean.startswith(p_vi) or low_clean.endswith(p_vi):
                detected_pose_en = p_en
                competing_poses_en = p_comp
                break

        detected_subject_en = None
        for n_vi, n_en in cultural_nouns.items():
            if n_vi in low_clean:
                detected_subject_en = n_en
                break

        # Construct primary English translation
        components = []
        if detected_color_en:
            components.append(detected_color_en)
        if detected_subject_en:
            components.append(detected_subject_en)
        elif en_query:
            components.append(en_query)
        else:
            components.append(clean_query)
        if detected_pose_en:
            components.append(detected_pose_en)

        primary_text = " ".join(components)

        # Generate Query Expansion variants to cover vector space synonyms and prompt templates
        query_variants = [primary_text]
        
        # Add Prompt Templates
        query_variants.append(f"a photo of {primary_text}")
        query_variants.append(f"a video frame showing {primary_text}")

        # Add Synonym & Structural variants
        synonym_map = {
            "car": ["automobile", "sedan", "vehicle"],
            "motorcycle": ["motorbike", "scooter"],
            "speaker": ["presenter", "orator"],
            "soccer": ["football", "ball game"],
            "building": ["building entrance", "structure"]
        }

        variant_syn = primary_text
        for word, syns in synonym_map.items():
            if word in primary_text.lower():
                variant_syn = variant_syn.lower().replace(word, syns[0])
                query_variants.append(variant_syn)
                break

        # Unique query variants
        unique_variants = list(dict.fromkeys(query_variants))

        def encode_text(t_str: str) -> np.ndarray:
            inputs = self.processor(
                text=[t_str],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                if hasattr(outputs, "text_embeds"):
                    tf = outputs.text_embeds
                elif hasattr(outputs, "pooler_output"):
                    tf = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    tf = outputs
                else:
                    tf = outputs[0]
                tf = tf / tf.norm(dim=-1, keepdim=True)
                return tf.cpu().numpy().squeeze(0)

        # Calculate similarity scores across all Query Expansion variants
        variant_sims = []
        for v in unique_variants:
            v_vec = encode_text(v)
            variant_sims.append(np.dot(self.embeddings, v_vec))

        # Generalized Multi-Aspect Constraint (Subject + Color + Pose/Action)
        aspect_constraint = None
        competing_aspect_penalty = np.zeros(len(self.embeddings), dtype=np.float32)

        subj_query = detected_subject_en if detected_subject_en else primary_text
        has_aspect = detected_color_en or detected_pose_en or any(w in low_clean for w in ["xe", "car", "áo", "người", "tòa nhà"])

        if has_aspect:
            target_aspect_words = []
            if detected_color_en:
                target_aspect_words.append(detected_color_en)
            target_aspect_words.append(subj_query)
            if detected_pose_en:
                target_aspect_words.append(detected_pose_en)

            target_aspect_query = " ".join(target_aspect_words)
            
            subj_vec = encode_text(subj_query)
            aspect_vec = encode_text(target_aspect_query)
            
            sim_subj = np.clip(np.dot(self.embeddings, subj_vec), 0, None)
            sim_aspect = np.clip(np.dot(self.embeddings, aspect_vec), 0, None)
            
            # Harmonic Mean requires BOTH Subject AND Target Aspect (Color/Pose) to be strongly present
            aspect_constraint = (2.0 * sim_subj * sim_aspect) / (sim_subj + sim_aspect + 1e-5)
            if aspect_constraint.max() > 0:
                aspect_constraint = aspect_constraint / aspect_constraint.max()

            # Competing Aspect Contrast Penalty: Compare Target Aspect vs Competing Aspects
            competing_queries = []
            if detected_color_en and competing_colors_en:
                for c_comp in competing_colors_en[:3]:
                    competing_queries.append(f"{c_comp} {subj_query}")

            if detected_pose_en and competing_poses_en:
                for p_comp in competing_poses_en:
                    competing_queries.append(f"{subj_query} {p_comp}")

            if competing_queries:
                comp_sims_list = []
                for q_comp in competing_queries:
                    q_comp_vec = encode_text(q_comp)
                    comp_sims_list.append(np.dot(self.embeddings, q_comp_vec))
                max_competing_sim = np.maximum.reduce(comp_sims_list)
                
                # If competing aspect similarity strictly exceeds target aspect similarity, apply heavy demotion (-0.30)
                aspect_delta = sim_aspect - max_competing_sim
                competing_aspect_penalty = np.where(aspect_delta < 0.00, 0.30, 0.0).astype(np.float32)

        # Compute Reciprocal Rank Fusion (RRF) scores across variants
        rrf_scores = np.zeros(len(self.embeddings), dtype=np.float32)
        top_n_rrf = min(1000, len(self.embeddings))

        for sims in variant_sims:
            top_indices = np.argpartition(-sims, top_n_rrf)[:top_n_rrf]
            top_sorted = top_indices[np.argsort(-sims[top_indices])]
            for rank_idx, idx in enumerate(top_sorted, 1):
                rrf_scores[idx] += 1.0 / (60.0 + rank_idx)

        # Normalize RRF scores
        if rrf_scores.max() > 0:
            rrf_scores = rrf_scores / rrf_scores.max()

        # Combine primary vector similarity (35%), RRF expansion score (15%), Dynamic Aspect Constraint (50%), minus Competing Penalty
        primary_sims = variant_sims[0]
        if aspect_constraint is not None:
            combined_scores = (0.35 * primary_sims + 0.15 * rrf_scores + 0.50 * aspect_constraint) - competing_aspect_penalty
        else:
            combined_scores = 0.70 * primary_sims + 0.30 * rrf_scores

        # BTC Object Ensemble Verification: Map query target objects to OpenImages/YOLO classes
        target_entities = set()
        low_query = primary_text.lower()
        if any(w in low_query for w in ["car", "vehicle", "automobile", "sedan", "xe ô tô", "xe hơi"]):
            target_entities.update(["Car", "Vehicle", "Land vehicle", "Automobile"])
        if any(w in low_query for w in ["motorcycle", "motorbike", "scooter", "xe máy"]):
            target_entities.update(["Motorcycle", "Motorbike", "Bicycle"])
        if any(w in low_query for w in ["speaker", "person", "man", "woman", "diễn giả", "người"]):
            target_entities.update(["Person", "Man", "Woman", "Human body", "Human face"])
        if any(w in low_query for w in ["soccer", "football", "ball", "bóng đá"]):
            target_entities.update(["Ball", "Sports equipment", "Footwear", "Person"])

        objects_dir = os.path.join(self.data_dir, "objects")

        # Select candidates for NMS & Output
        top_candidate_indices = np.argsort(-combined_scores)[:500]

        results = []
        visited_frames = {} # video_id -> list of selected frame_indices for NMS
        effective_nms_gap = max(nms_frame_gap, 15)

        for idx in top_candidate_indices:
            score = float(combined_scores[idx])
            item = self.metadata[idx]
            v_id = item["video_id"]
            f_idx = item["frame_idx"]
            f_n = item.get("n", 1)

            # Frame Non-Maximum Suppression (NMS)
            if effective_nms_gap > 0 and v_id in visited_frames:
                too_close = any(abs(f_idx - prev_f) < effective_nms_gap for prev_f in visited_frames[v_id])
                if too_close:
                    continue

            # Object Ensemble Boosting: Check BTC detection JSON for object match
            obj_json_path = os.path.join(objects_dir, v_id, f"{f_n:03d}.json")
            if target_entities and os.path.exists(obj_json_path):
                try:
                    with open(obj_json_path, "r", encoding="utf-8") as f_obj:
                        obj_data = json.load(f_obj)
                        detected_classes = set(obj_data.get("detection_class_entities", []))
                        if target_entities.intersection(detected_classes):
                            score += 0.05 # Boost score for object match ground truth
                        elif "car" in low_query and "Car" not in detected_classes and "Vehicle" not in detected_classes:
                            score -= 0.04 # Penalty for missing requested object
                except Exception:
                    pass

            if v_id not in visited_frames:
                visited_frames[v_id] = []
            visited_frames[v_id].append(f_idx)

            results.append({
                "video_id": v_id,
                "frame_idx": f_idx,
                "frame_filename": item["frame_filename"],
                "score": round(score, 4),
                "rel_path": item["rel_path"],
                "abs_path": item["abs_path"]
            })

            if len(results) >= top_k:
                break

        # Re-sort final top_k after object ensemble boost
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results
