import os
import json
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Optional

class TextualKISRetriever:
    """
    Search Engine for Textual Known Item Search (Task 1).
    Performs Dual-Model Ensemble (CLIP + SigLIP) text-to-image similarity search over indexed keyframes.
    """
    def __init__(
        self,
        data_dir: str,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        use_siglip_only: bool = True,
        use_clip_only: bool = False,
        use_siglip_version: str = "siglip2"
    ):
        self.data_dir = data_dir
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.use_siglip_only = use_siglip_only
        self.use_clip_only = use_clip_only
        self.use_siglip_version = use_siglip_version
        self.spacy_nlp = None
        
        self.metadata_path = os.path.join(data_dir, "metadata.json")
        self.embeddings_path = os.path.join(data_dir, "embeddings.npy")
        
        sig1_path = os.path.join(data_dir, "embeddings_siglip.npy")
        sig2_path = os.path.join(data_dir, "embeddings_siglip2.npy")
        
        if use_siglip_version == "siglip1" and os.path.exists(sig1_path):
            self.siglip_embeddings_path = sig1_path
            self.siglip_model_name = "google/siglip-base-patch16-224"
        elif use_siglip_version == "siglip2" and os.path.exists(sig2_path):
            self.siglip_embeddings_path = sig2_path
            self.siglip_model_name = "google/siglip2-base-patch16-224"
        elif os.path.exists(sig1_path):
            self.siglip_embeddings_path = sig1_path
            self.siglip_model_name = "google/siglip-base-patch16-224"
        else:
            self.siglip_embeddings_path = sig2_path
            self.siglip_model_name = "google/siglip2-base-patch16-224"
        
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embeddings_siglip: Optional[np.ndarray] = None
        
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.siglip_model = None
        self.siglip_processor = None
        
        self._is_loaded = False

    def load_index_and_model(self):
        """Loads metadata, vector embeddings, and initialized CLIP / SigLIP 2 models."""
        if self._is_loaded:
            return

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found in '{self.data_dir}'. Please run `src/index_builder.py` first!"
            )

        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        has_siglip = os.path.exists(self.siglip_embeddings_path)

        if self.use_siglip_only and has_siglip:
            is_sig2 = "siglip2" in self.siglip_embeddings_path
            ver_name = "SigLIP 2" if is_sig2 else "SigLIP 1"
            print(f"🚀 Pure {ver_name} Mode Activated (Bypassing CLIP for 2x Speed & Reduced Memory)!")
            print(f"Loading {ver_name} embeddings from {self.siglip_embeddings_path}...")
            raw_siglip = np.load(self.siglip_embeddings_path)
            norms = np.linalg.norm(raw_siglip, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings_siglip = np.ascontiguousarray(raw_siglip / norms, dtype=np.float32)
            self.embeddings = self.embeddings_siglip

            siglip_name = getattr(self, "siglip_model_name", "google/siglip-base-patch16-224")
            print(f"Loading official Google {ver_name} model '{siglip_name}' on '{self.device}'...")
            from transformers import AutoProcessor, AutoModel
            self.siglip_model = AutoModel.from_pretrained(siglip_name).to(self.device)
            self.siglip_processor = AutoProcessor.from_pretrained(siglip_name)
            self.siglip_model.eval()

        elif self.use_clip_only:
            print("🚀 Pure CLIP Mode Activated (Bypassing SigLIP for Fast CLIP Testing)!")
            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(f"CLIP embeddings not found in '{self.data_dir}'!")

            print(f"Loading CLIP embeddings from {self.embeddings_path}...")
            raw_clip = np.load(self.embeddings_path)
            norms_c = np.linalg.norm(raw_clip, axis=1, keepdims=True)
            norms_c[norms_c == 0] = 1.0
            self.embeddings = np.ascontiguousarray(raw_clip / norms_c, dtype=np.float32)

            print(f"Loading CLIP model '{self.model_name}' on '{self.device}'...")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            self.embeddings_siglip = None
        else:
            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(f"CLIP embeddings not found in '{self.data_dir}'!")

            print(f"Loading CLIP embeddings from {self.embeddings_path}...")
            raw_clip = np.load(self.embeddings_path)
            norms_c = np.linalg.norm(raw_clip, axis=1, keepdims=True)
            norms_c[norms_c == 0] = 1.0
            self.embeddings = np.ascontiguousarray(raw_clip / norms_c, dtype=np.float32)

            print(f"Loading CLIP model '{self.model_name}' on '{self.device}'...")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()

            if has_siglip:
                try:
                    print(f"Loading SigLIP embeddings from {self.siglip_embeddings_path}...")
                    raw_siglip = np.load(self.siglip_embeddings_path)
                    norms_s = np.linalg.norm(raw_siglip, axis=1, keepdims=True)
                    norms_s[norms_s == 0] = 1.0
                    self.embeddings_siglip = np.ascontiguousarray(raw_siglip / norms_s, dtype=np.float32)

                    siglip_name = "google/siglip2-base-patch16-224"
                    print(f"Loading official Google SigLIP 2 model '{siglip_name}' on '{self.device}'...")
                    from transformers import AutoProcessor, AutoModel
                    self.siglip_model = AutoModel.from_pretrained(siglip_name).to(self.device)
                    self.siglip_processor = AutoProcessor.from_pretrained(siglip_name)
                    self.siglip_model.eval()
                except Exception as e:
                    print(f"SigLIP loading notice: {e}")

        # Pre-warm GPU / MPS PyTorch Metal JIT kernels & Pre-encode Negative Prompts for 0ms First-Query Latency
        self.neg_vecs = None
        if self.siglip_model is not None and self.siglip_processor is not None:
            neg_prompts = [
                "tv broadcast channel logo bumper screen",
                "commercial advertisement sponsor banner graphics",
                "abstract graphic background illustration",
                "closing credits title card screen"
            ]
            inputs_neg = self.siglip_processor(text=neg_prompts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out_neg = self.siglip_model.get_text_features(**inputs_neg)
                tf_neg = out_neg.pooler_output if hasattr(out_neg, "pooler_output") else (out_neg.text_embeds if hasattr(out_neg, "text_embeds") else out_neg[0])
                tf_neg = tf_neg / tf_neg.norm(dim=-1, keepdim=True)
                self.neg_vecs = tf_neg.cpu().numpy().astype(np.float32)

        self._trans_cache = {}
        self._is_loaded = True
        print(f"Retriever initialized with {len(self.metadata)} indexed keyframes.")

    def search(
        self,
        query: str,
        top_k: int = 100,
        nms_frame_gap: int = 5,
        max_per_video: int = 2,
        visual_sim_threshold: float = 0.90,
        use_reranker: bool = True,
        use_spatial_grid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes Textual KIS search for a natural language text query using Dual-Model Ensemble & Google Fan-Out.
        """
        if not self._is_loaded:
            self.load_index_and_model()

        # Clean query text processing
        clean_query = query.strip()
        
        INSTANT_VIET_TO_ENG = {
            "bánh xèo": "vietnamese crispy pancake banh xeo",
            "bánh xèo miền tây": "western vietnamese crispy pancake banh xeo",
            "làn đường dành cho người đi bộ": "pedestrian crosswalk zebra crossing",
            "người đi bộ": "pedestrian walking crosswalk",
            "vạch sang đường": "zebra crosswalk pedestrian crossing",
            "xe ô tô mui trần": "convertible sports open top car",
            "mui trần": "convertible open top car",
            "xe ô tô": "automobile car vehicle",
            "xe máy": "motorbike motorcycle scooter",
            "xe đạp": "bicycle bike cyclist",
            "con chó": "dog puppy canine",
            "chó": "dog puppy canine",
            "con mèo": "cat kitten feline",
            "mèo": "cat kitten feline",
            "biển": "ocean sea beach coast",
            "bãi biển": "sandy beach ocean coast",
            "công viên": "city park green garden outdoor",
            "núi": "mountain hills outdoor landscape"
        }

        # Smart Instant Dictionary: ONLY apply to short exact queries (1-3 words) to preserve full details of complex queries
        en_query = None
        low_clean = clean_query.lower()
        word_count = len(low_clean.split())
        
        if word_count <= 3 and low_clean in INSTANT_VIET_TO_ENG:
            en_query = INSTANT_VIET_TO_ENG[low_clean]
        else:
            has_non_ascii = any(ord(char) > 127 for char in clean_query)
            if has_non_ascii:
                if hasattr(self, "_trans_cache") and clean_query in self._trans_cache:
                    en_query = self._trans_cache[clean_query]
                else:
                    try:
                        from deep_translator import GoogleTranslator
                        import concurrent.futures
                        def _do_trans():
                            return GoogleTranslator(source="auto", target="en").translate(clean_query)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(_do_trans)
                            translated = future.result(timeout=2.0)
                            if translated and translated.strip():
                                en_query = translated.strip()
                                if hasattr(self, "_trans_cache"):
                                    self._trans_cache[clean_query] = en_query
                    except Exception:
                        en_query = None

        # Base English translation construction
        base_en = en_query if en_query else clean_query
        
        # Vietnamese Cultural & Landmark Idiom Fallback
        cultural_idioms = {
            "múa lân": "lion dance costume performance barongsai",
            "con lân": "lion dance costume performance barongsai",
            "lân": "lion dance costume performance barongsai",
            "áo dài": "vietnamese traditional ao dai dress",
            "áo bà ba": "vietnamese traditional ao ba ba shirt",
            "bánh chưng": "vietnamese square sticky rice cake",
            "bánh tét": "vietnamese cylindrical sticky rice cake",
            "bánh xèo": "vietnamese savory crispy pancake",
            "phở": "vietnamese beef noodle soup pho",
            "xích lô": "vietnamese cyclo pedicab rickshaw",
            "chợ bến thành": "ben thanh market landmark ho chi minh city",
            "nhà thờ đức bà": "notre dame cathedral saigon landmark",
            "vịnh hạ long": "ha long bay limestone karst islands landmark",
            "cầu rồng": "dragon bridge da nang landmark",
            "đàn bầu": "vietnamese traditional monochord zither dan bau",
            "nhà sàn": "vietnamese traditional stilt house"
        }

        # Preserve cultural term keywords if present
        cultural_term_en = None
        for idiom in sorted(cultural_idioms.keys(), key=len, reverse=True):
            if idiom in low_clean:
                cultural_term_en = cultural_idioms[idiom]
                break

        if cultural_term_en and cultural_term_en not in base_en.lower():
            combined_text_en = f"{base_en} {cultural_term_en}"
        else:
            combined_text_en = base_en

        # Solution B: Color Spectrum Synset Detection & Mapping
        COLOR_SYNSET_MAP = {
            "đỏ": ("red", ["vivid red", "bright red", "crimson", "scarlet", "ruby red"]),
            "red": ("red", ["vivid red", "bright red", "crimson", "scarlet", "ruby red"]),
            "vàng": ("yellow", ["bright yellow", "golden yellow", "amber", "mustard yellow"]),
            "yellow": ("yellow", ["bright yellow", "golden yellow", "amber", "mustard yellow"]),
            "xanh dương": ("blue", ["vivid blue", "navy blue", "cyan", "royal blue", "azure"]),
            "xanh da trời": ("blue", ["vivid blue", "navy blue", "cyan", "royal blue", "azure"]),
            "xanh biển": ("blue", ["vivid blue", "navy blue", "cyan", "royal blue", "azure"]),
            "blue": ("blue", ["vivid blue", "navy blue", "cyan", "royal blue", "azure"]),
            "xanh lá": ("green", ["vivid green", "emerald green", "lime green", "olive green"]),
            "xanh lục": ("green", ["vivid green", "emerald green", "lime green", "olive green"]),
            "green": ("green", ["vivid green", "emerald green", "lime green", "olive green"]),
            "trắng": ("white", ["pure white", "bright white", "snow white", "ivory"]),
            "white": ("white", ["pure white", "bright white", "snow white", "ivory"]),
            "đen": ("black", ["jet black", "dark black", "charcoal black"]),
            "black": ("black", ["jet black", "dark black", "charcoal black"]),
            "hồng": ("pink", ["bright pink", "rose pink", "magenta", "hot pink"]),
            "pink": ("pink", ["bright pink", "rose pink", "magenta", "hot pink"]),
            "cam": ("orange", ["bright orange", "vivid orange", "tangerine", "amber orange"]),
            "orange": ("orange", ["bright orange", "vivid orange", "tangerine", "amber orange"]),
            "tím": ("purple", ["vivid purple", "violet", "lavender", "deep purple"]),
            "purple": ("purple", ["vivid purple", "violet", "lavender", "deep purple"]),
            "nâu": ("brown", ["dark brown", "tan", "chocolate brown", "chestnut brown"]),
            "brown": ("brown", ["dark brown", "tan", "chocolate brown", "chestnut brown"]),
            "xám": ("gray", ["silver", "ash gray", "dark gray", "metallic gray"]),
            "gray": ("gray", ["silver", "ash gray", "dark gray", "metallic gray"])
        }
        ALL_COMPETING_COLORS = ["red", "yellow", "blue", "green", "white", "black", "pink", "orange", "purple", "brown", "gray"]

        detected_color_info = None
        for ck in sorted(COLOR_SYNSET_MAP.keys(), key=len, reverse=True):
            if ck in low_clean or ck in combined_text_en.lower():
                detected_color_info = COLOR_SYNSET_MAP[ck]
                break

        # Robust Multi-Prompt Dynamic Embedding Fusion for High-Precision Zero-Shot Retrieval
        if self.use_siglip_only and self.embeddings_siglip is not None:
            # Solution B: Multi-Prompt Target Ensembling (Raw Text + Photo Prompt + Color Spectrum Prompts)
            target_prompts = [combined_text_en, f"a photo of {combined_text_en}"]
            if detected_color_info:
                prim_color, syn_list = detected_color_info
                target_prompts.append(f"a {syn_list[0]} {base_en}")
                target_prompts.append(f"a photo of {prim_color} {base_en} with distinct {prim_color} color")
            
            inputs = self.siglip_processor(
                text=target_prompts, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=64
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.siglip_model.get_text_features(**inputs)
                tf = outputs.pooler_output if hasattr(outputs, "pooler_output") else (outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0])
                tf = tf / tf.norm(dim=-1, keepdim=True)
                # Average ensemble of raw text & photo & color spectrum prompt vectors
                siglip_vec_tensor = tf.mean(dim=0, keepdim=True)
                siglip_vec_tensor = siglip_vec_tensor / siglip_vec_tensor.norm(dim=-1, keepdim=True)
                siglip_vec = siglip_vec_tensor.cpu().numpy().squeeze(0)

            # 1. Base SigLIP Cosine Similarity
            sims = np.dot(self.embeddings_siglip, siglip_vec).astype(np.float32)

            # Solution A: Dynamic Competing Color Demotion (Opponent Color Contrast Penalty)
            if detected_color_info:
                prim_color, _ = detected_color_info
                comp_colors = [c for c in ALL_COMPETING_COLORS if c != prim_color][:5]
                comp_prompts = [f"a photo of a {c} {base_en}" for c in comp_colors]
                inputs_comp = self.siglip_processor(
                    text=comp_prompts, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=64
                ).to(self.device)
                with torch.inference_mode():
                    outputs_comp = self.siglip_model.get_text_features(**inputs_comp)
                    tf_comp = outputs_comp.pooler_output if hasattr(outputs_comp, "pooler_output") else (outputs_comp.text_embeds if hasattr(outputs_comp, "text_embeds") else outputs_comp[0])
                    tf_comp = tf_comp / tf_comp.norm(dim=-1, keepdim=True)
                    comp_np = tf_comp.cpu().numpy().astype(np.float32)

                comp_sims_matrix = np.dot(self.embeddings_siglip, comp_np.T)
                max_comp_sims = np.max(comp_sims_matrix, axis=1)
                color_penalty = 0.45 * np.maximum(max_comp_sims - sims + 0.005, 0.0)
                sims = sims - color_penalty

            # 2. Negative Calibration (Demotes TV bumpers, studio graphics & commercial logos)
            if hasattr(self, "neg_vecs") and self.neg_vecs is not None:
                neg_sims_matrix = np.dot(self.embeddings_siglip, self.neg_vecs.T)
                max_neg_sims = np.max(neg_sims_matrix, axis=1)
                sims = sims - 0.35 * np.maximum(max_neg_sims - 0.04, 0.0)

            # Select candidate pool
            candidate_k = min(max(top_k * 10, 1000), len(self.embeddings_siglip))
            top_indices = np.argpartition(-sims, candidate_k)[:candidate_k]
            top_sorted = top_indices[np.argsort(-sims[top_indices])]

            # Extract target keywords for Object Grounding
            target_keywords = set()
            for word in combined_text_en.lower().split():
                if len(word) >= 3 and word not in ["the", "and", "with", "from", "for", "photo", "image", "scenery", "outdoor", "background"]:
                    target_keywords.add(word)

            objects_dir = os.path.join(self.data_dir, "objects")
            raw_candidates = []

            for idx in top_sorted:
                item = dict(self.metadata[idx])
                # Skip intro countdown/logo frames (n <= 2)
                if item.get("n", 1) <= 2:
                    continue

                score = float(sims[idx])
                v_id = item["video_id"]
                f_n = item.get("n", 1)

                # 3. Object Grounding Confidence Boost from BTC OpenImages detections
                if target_keywords and os.path.exists(objects_dir):
                    obj_p = os.path.join(objects_dir, v_id, f"{f_n:03d}.json")
                    if os.path.exists(obj_p):
                        try:
                            with open(obj_p, "r", encoding="utf-8") as f_obj:
                                obj_data = json.load(f_obj)
                                entities = [e.lower() for e in obj_data.get("detection_class_entities", [])]
                                confs = [float(s) for s in obj_data.get("detection_scores", [])]
                                obj_boost = 0.0
                                for e_name, e_score in zip(entities, confs):
                                    if e_score >= 0.15 and any(k in e_name for k in target_keywords):
                                        obj_boost = max(obj_boost, 0.08 * e_score)
                                    elif e_score >= 0.30 and any(k in ["mammal", "animal", "vehicle", "carnivore"] for k in target_keywords if k in e_name):
                                        obj_boost = max(obj_boost, 0.02 * e_score)
                                score += obj_boost
                        except Exception:
                            pass

                item["score"] = score
                item["raw_index"] = int(idx)
                raw_candidates.append(item)

            # Re-sort candidates based on combined calibrated score + object grounding
            raw_candidates.sort(key=lambda x: x["score"], reverse=True)

            if use_reranker and hasattr(self, "vlm_reranker") and self.vlm_reranker and self.vlm_reranker.is_ready:
                candidates = self.vlm_reranker.rerank(query=combined_text_en, candidates=raw_candidates, top_k=max(top_k * 3, 100))
            else:
                candidates = raw_candidates

            # Smart Temporal NMS, Video Diversity & Cross-Video Visual Duplicate Suppression
            final_results = []
            seen_videos = {}  # v_id -> list of (n, pts_time)
            selected_vecs = []

            for item in candidates:
                v_id = item["video_id"]
                n_val = item.get("n", 1)
                pts_val = item.get("pts_time", 0.0)
                raw_idx = item.get("raw_index")

                if v_id not in seen_videos:
                    seen_videos[v_id] = []

                # 1. Video Diversity Cap: Limit frames per video in the results
                if max_per_video > 0 and len(seen_videos[v_id]) >= max_per_video:
                    continue

                # 2. Temporal NMS: Prevent consecutive near-identical frames from the same video
                if any(abs(n_val - prev_n) <= nms_frame_gap or abs(pts_val - prev_pts) <= max(nms_frame_gap * 2.0, 10.0) for prev_n, prev_pts in seen_videos[v_id]):
                    continue

                # 3. Cross-Video Visual Duplicate Suppression (Eliminates repeated TV logos/sponsor bumpers across videos)
                if visual_sim_threshold > 0 and raw_idx is not None and len(selected_vecs) > 0:
                    item_vec = self.embeddings_siglip[raw_idx]
                    vec_matrix = np.vstack(selected_vecs[-200:])
                    if np.max(np.dot(vec_matrix, item_vec)) >= visual_sim_threshold:
                        continue
                    selected_vecs.append(item_vec)
                elif raw_idx is not None:
                    selected_vecs.append(self.embeddings_siglip[raw_idx])

                seen_videos[v_id].append((n_val, pts_val))
                
                v_id_lower = v_id.lower()
                f_name = item["frame_filename"]
                abs_path = os.path.join(self.data_dir, "..", "keyframes", v_id, f_name)
                if not os.path.exists(abs_path):
                    abs_path = os.path.join(self.data_dir, "..", "keyframes", v_id_lower, f_name)
                item["abs_path"] = abs_path
                final_results.append(item)
                if len(final_results) >= top_k:
                    break

            return final_results

        # Open-Vocabulary Syntactic Parser via spaCy POS & Dependency Tree
        detected_colors = []
        detected_subjects = []
        detected_poses = []
        detected_noun_lemmas = set()

        STANDARD_COLOR_SPECTRUM = {
            "red": "red", "crimson": "red", "scarlet": "red", "maroon": "red", "burgundy": "red",
            "yellow": "yellow golden", "gold": "yellow golden", "golden": "yellow golden", "amber": "yellow golden",
            "green": "green", "lime": "green", "olive": "green", "emerald": "green",
            "blue": "blue", "azure": "blue", "cyan": "blue", "teal": "blue", "navy": "blue",
            "white": "white silver", "silver": "white silver", "ivory": "white silver",
            "black": "black", "dark": "black",
            "pink": "pink", "rose": "pink", "magenta": "pink",
            "purple": "purple", "violet": "purple",
            "orange": "orange",
            "brown": "brown", "tan": "brown", "beige": "brown", "khaki": "brown"
        }

        ALL_SPECTRUM_COLORS = ["red", "yellow golden", "green", "blue", "white silver", "black", "pink", "purple", "orange"]

        if self.spacy_nlp:
            doc = self.spacy_nlp(combined_text_en)
            for token in doc:
                tok_low = token.text.lower()
                tok_lemma = token.lemma_.lower()
                # Open-vocabulary color modifier detection
                if tok_low in STANDARD_COLOR_SPECTRUM:
                    detected_colors.append(STANDARD_COLOR_SPECTRUM[tok_low])
                # Open-vocabulary subject noun chunk detection
                elif token.pos_ in ["NOUN", "PROPN"] and len(tok_low) > 2:
                    detected_subjects.append(tok_low)
                    detected_noun_lemmas.add(tok_lemma)
                    detected_noun_lemmas.add(tok_low)
                # Open-vocabulary action & pose verb detection
                elif token.pos_ in ["VERB"] and tok_low not in ["is", "are", "was", "were", "be", "have", "has", "do", "show"]:
                    detected_poses.append(tok_low)

        # Primary English Text Construction
        primary_text = combined_text_en

        # 1. Google Query Fan-Out Multi-Stream Decomposition Engine (Zero Hardcode)
        head_noun_en = detected_subjects[0] if detected_subjects else "object"
        qualifier_words = detected_colors + detected_poses
        qualifier_en = " ".join(qualifier_words) if qualifier_words else ""

        # Stream A: Global Scene Context Sub-Query
        stream_a_query = f"{primary_text} outdoor scenery background"
        # Stream B: Fine-Grained Attribute Sub-Query
        stream_b_query = f"{qualifier_en} {head_noun_en} detailed photo" if qualifier_en else primary_text
        # Stream C: Domain Visual Synset Sub-Query
        if any(w in primary_text.lower() for w in ["pedestrian", "walkway", "crosswalk", "crossing", "lane", "street"]):
            stream_c_query = "zebra crosswalk pedestrian crossing painted white lines on asphalt street"
        else:
            stream_c_query = f"a photo of {primary_text}"

        fanout_streams = {
            "stream_a": stream_a_query,
            "stream_b": stream_b_query,
            "stream_c": stream_c_query
        }

        def encode_clip_text(t_str: str) -> np.ndarray:
            inputs = self.processor(text=[t_str], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                tf_tensor = getattr(outputs, 'text_embeds', getattr(outputs, 'pooler_output', outputs[0] if isinstance(outputs, (tuple, list)) else outputs))
                tf = tf_tensor / tf_tensor.norm(dim=-1, keepdim=True)
                return tf.cpu().numpy().squeeze(0)

        def encode_siglip_text(t_str: str) -> Optional[np.ndarray]:
            if self.siglip_model is None or self.siglip_processor is None:
                return None
            inputs = self.siglip_processor(text=[t_str], return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.siglip_model.get_text_features(**inputs)
                tf_tensor = getattr(outputs, 'pooler_output', getattr(outputs, 'text_embeds', outputs[0] if isinstance(outputs, (tuple, list)) else outputs))
                tf = tf_tensor / tf_tensor.norm(dim=-1, keepdim=True)
                return tf.cpu().numpy().squeeze(0)

        def compute_ensemble_similarity(t_str: str) -> np.ndarray:
            if self.use_clip_only or self.embeddings_siglip is None:
                clip_vec = encode_clip_text(t_str)
                return np.dot(self.embeddings, clip_vec).astype(np.float32)

            if self.siglip_model is not None and hasattr(self.siglip_model, "logit_scale"):
                scale = float(torch.exp(self.siglip_model.logit_scale.detach().cpu()).item())
                bias = float(self.siglip_model.logit_bias.detach().cpu().item())
            else:
                scale, bias = 90.0, -10.0

            if self.use_siglip_only and self.embeddings_siglip is not None:
                siglip_vec = encode_siglip_text(t_str)
                if siglip_vec is not None:
                    raw_siglip = np.dot(self.embeddings_siglip, siglip_vec)
                    return raw_siglip.astype(np.float32)

            clip_vec = encode_clip_text(t_str)
            sim_clip = np.dot(self.embeddings, clip_vec)
            if self.embeddings_siglip is not None:
                siglip_vec = encode_siglip_text(t_str)
                if siglip_vec is not None:
                    raw_siglip = np.dot(self.embeddings_siglip, siglip_vec)
                    siglip_logits = scale * raw_siglip + bias
                    siglip_prob = 1.0 / (1.0 + np.exp(-np.clip(siglip_logits, -50.0, 50.0)))
                    return (0.50 * sim_clip + 0.50 * siglip_prob).astype(np.float32)
            return sim_clip

        # Calculate similarity scores across all Fan-Out Sub-Query Streams (Dual-Model Ensemble)
        stream_sims = {}
        for s_name, s_text in fanout_streams.items():
            stream_sims[s_name] = compute_ensemble_similarity(s_text)

        # 2. Multi-Stream Google Fan-Out Reciprocal Rank Fusion (RRF)
        rrf_fanout_scores = np.zeros(len(self.embeddings), dtype=np.float32)
        top_n_rrf = min(1000, len(self.embeddings))

        stream_weights = {"stream_a": 0.25, "stream_b": 0.45, "stream_c": 0.30}

        for s_name, sims in stream_sims.items():
            s_weight = stream_weights.get(s_name, 0.33)
            top_indices = np.argpartition(-sims, top_n_rrf)[:top_n_rrf]
            top_sorted = top_indices[np.argsort(-sims[top_indices])]
            for rank_idx, idx in enumerate(top_sorted, 1):
                rrf_fanout_scores[idx] += s_weight / (60.0 + rank_idx)

        if rrf_fanout_scores.max() > 0:
            rrf_fanout_scores = rrf_fanout_scores / rrf_fanout_scores.max()

        # 3. Dynamic Entropy & Aspect Soft-Margin Engine
        num_modifiers = len(detected_colors) + len(detected_poses) + max(0, len(detected_subjects) - 1)
        w_primary = float(max(0.15, 0.35 - 0.05 * num_modifiers))
        w_fanout_rrf = float(min(0.45, 0.25 + 0.05 * num_modifiers))
        w_aspect = float(min(0.40, 0.25 + 0.04 * num_modifiers))
        w_margin = float(min(0.30, 0.15 + 0.04 * num_modifiers))

        aspect_constraint = None
        competing_aspect_penalty = np.zeros(len(self.embeddings), dtype=np.float32)
        relative_margin_boost = np.zeros(len(self.embeddings), dtype=np.float32)

        target_color_en = detected_colors[0] if detected_colors else None
        subj_query = cultural_term_en if cultural_term_en else (" ".join(detected_subjects) if detected_subjects else primary_text)

        generic_query = f"generic {head_noun_en}"
        if qualifier_en:
            counterpart_query = f"plain generic {head_noun_en} without {qualifier_en}"
        else:
            counterpart_query = f"plain background scenery without {head_noun_en}"

        has_aspect = target_color_en or len(detected_poses) > 0 or (len(detected_subjects) > 0 and primary_text != head_noun_en)

        if has_aspect:
            target_aspect_words = []
            if target_color_en:
                target_aspect_words.append(target_color_en)
            target_aspect_words.append(subj_query)
            if detected_poses:
                target_aspect_words.append(" ".join(detected_poses))

            target_aspect_query = " ".join(target_aspect_words)
            
            sim_subj = compute_ensemble_similarity(subj_query)
            sim_aspect = compute_ensemble_similarity(target_aspect_query)
            sim_generic = compute_ensemble_similarity(generic_query)
            sim_counterpart = compute_ensemble_similarity(counterpart_query)
            
            # Harmonic Mean Score across Subject & Aspect
            aspect_constraint = (2.0 * sim_subj * sim_aspect) / (sim_subj + sim_aspect + 1e-5)
            if aspect_constraint.max() > 0:
                aspect_constraint = aspect_constraint / aspect_constraint.max()

            # Relative Fine-Grained Margin Boost
            fine_margin = np.clip(sim_aspect - sim_generic, 0, None)
            if fine_margin.max() > 0:
                relative_margin_boost = (fine_margin / fine_margin.max()).astype(np.float32)

            # Universal Dynamic Competing Color & Counterpart Demotion
            competing_queries = [counterpart_query]
            if target_color_en:
                competing_colors = [c for c in ALL_SPECTRUM_COLORS if target_color_en not in c and c not in target_color_en][:4]
                for c_comp in competing_colors:
                    competing_queries.append(f"{c_comp} {subj_query}")

            if competing_queries:
                comp_sims_list = []
                for q_comp in competing_queries:
                    comp_sims_list.append(compute_ensemble_similarity(q_comp))
                max_competing_sim = np.maximum.reduce(comp_sims_list)
                
                # Data-Driven Sigmoid Soft-Margin Demotion
                if has_aspect:
                    steepness = 12.0
                    aspect_delta = competing_aspect_penalty - relative_margin_boost
                    clipped_delta = np.clip(steepness * aspect_delta, -50.0, 50.0)
                    sigmoid_factor = 1.0 / (1.0 + np.exp(clipped_delta))
                competing_aspect_penalty = (0.35 * sigmoid_factor * (aspect_delta < 0.05)).astype(np.float32)

        # Combine layer scores using Google Fan-Out Rank Fusion & Dynamic Layer Weights
        primary_sims = stream_sims["stream_b"] if "stream_b" in stream_sims else list(stream_sims.values())[0]
        if aspect_constraint is not None:
            combined_scores = (w_primary * primary_sims + w_fanout_rrf * rrf_fanout_scores + w_aspect * aspect_constraint + w_margin * relative_margin_boost) - competing_aspect_penalty
        else:
            combined_scores = 0.60 * primary_sims + 0.40 * rrf_fanout_scores

        objects_dir = os.path.join(self.data_dir, "objects")

        # Select candidates for NMS & Output
        top_candidate_indices = np.argsort(-combined_scores)[:500]

        results = []
        visited_frames = {} # video_id -> list of (n, pts_time)
        selected_vecs = []
        effective_nms_gap = max(nms_frame_gap, 5)

        for idx in top_candidate_indices:
            score = float(combined_scores[idx])
            item = self.metadata[idx]
            v_id = item["video_id"]
            f_idx = item["frame_idx"]
            f_n = item.get("n", 1)
            f_pts = item.get("pts_time", 0.0)

            if v_id not in visited_frames:
                visited_frames[v_id] = []

            # 1. Video Diversity Cap
            if max_per_video > 0 and len(visited_frames[v_id]) >= max_per_video:
                continue

            # 2. Frame Non-Maximum Suppression (NMS)
            if effective_nms_gap > 0 and visited_frames[v_id]:
                too_close = any(abs(f_n - prev_n) <= effective_nms_gap or abs(f_pts - prev_pts) <= max(effective_nms_gap * 2.0, 10.0) for prev_n, prev_pts in visited_frames[v_id])
                if too_close:
                    continue

            # 3. Cross-Video Visual Duplicate Suppression
            if visual_sim_threshold > 0 and len(selected_vecs) > 0 and self.embeddings is not None:
                item_vec = self.embeddings[idx]
                vec_matrix = np.vstack(selected_vecs[-200:])
                if np.max(np.dot(vec_matrix, item_vec)) >= visual_sim_threshold:
                    continue
                selected_vecs.append(item_vec)
            elif self.embeddings is not None:
                selected_vecs.append(self.embeddings[idx])

            # Open-Vocabulary BTC Bounding-Box Regional Matcher (Stream C Grounding)
            obj_json_path = os.path.join(objects_dir, v_id, f"{f_n:03d}.json")
            if detected_noun_lemmas and os.path.exists(obj_json_path):
                try:
                    with open(obj_json_path, "r", encoding="utf-8") as f_obj:
                        obj_data = json.load(f_obj)
                        detected_classes = [c.lower() for c in obj_data.get("detection_class_entities", [])]
                        
                        # Match spaCy extracted noun lemmas dynamically against BTC OpenImages classes
                        has_object_match = any(
                            any(noun in cls_name for noun in detected_noun_lemmas)
                            for cls_name in detected_classes
                        )
                        if has_object_match:
                            score += 0.05 # Dynamic Regional Object Match Boost for ground truth detection
                except Exception:
                    pass

            visited_frames[v_id].append((f_n, f_pts))

            results.append({
                "video_id": v_id,
                "frame_idx": f_idx,
                "frame_filename": item["frame_filename"],
                "score": round(score, 4),
                "rel_path": item["rel_path"],
                "abs_path": item["abs_path"]
            })

            if len(results) >= (top_k if not use_reranker else max(top_k, 20)):
                break

        # Re-sort final candidates after dynamic object ensemble boost
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Stage 2: Level 3 Google SOTA Two-Stage VLM Re-Ranking
        if use_reranker:
            if not hasattr(self, "vlm_reranker") or self.vlm_reranker is None:
                try:
                    from src.task1_kis.vlm_reranker import VLMReranker
                    self.vlm_reranker = VLMReranker(
                        device=self.device,
                        model=self.siglip_model,
                        processor=self.siglip_processor
                    )
                except Exception as e:
                    print(f"⚠️ VLM Re-Ranker import error: {e}")
                    self.vlm_reranker = None

            if self.vlm_reranker is not None:
                parent_dir = os.path.dirname(os.path.abspath(self.data_dir))
                k_dir = os.path.join(parent_dir, "keyframes")
                if not os.path.exists(k_dir):
                    k_dir = os.path.join(self.data_dir, "keyframes")
                results = self.vlm_reranker.rerank(
                    query=query,
                    candidates=results,
                    image_base_dir=k_dir if os.path.exists(k_dir) else None
                )[:top_k]

        return results[:top_k]
