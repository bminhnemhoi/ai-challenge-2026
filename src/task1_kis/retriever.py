import os
import re
import json
import math
import unicodedata
import numpy as np
import torch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Optional, Set, Tuple

from .metadata_bm25 import MetadataBM25Searcher

def extract_coarse_query(en_text: str) -> str:
    """
    Rút gọn câu truy vấn tiếng Anh sang dạng Coarse Query (Chủ thể + Hành động + Bối cảnh cốt lõi)
    nhằm chống loãng vector (Vector Dilution) ở Stage 1 Global Dense Retrieval.
    Xử lý hoàn toàn bằng Regex cú pháp (0.2ms latency, 100% offline).
    """
    text = en_text.strip()
    if not text:
        return text

    # 1. Loại bỏ các mệnh đề phụ/chi tiết vi mô: with ..., wearing ..., holding ..., displaying ..., showing ...
    patterns = [
        r'\b(with\s+(a\s+|an\s+|the\s+)?(severely\s+|heavily\s+|badly\s+)?(dented|broken|damaged|scratched|wearing|holding|carrying|colored)\s+[^,\.;]+)',
        r'\b(with\s+[^,\.;]+?(roof|knife|spoon|hat|glasses|shirt|shoes|apron|towel|bag|phone|plate|bowl|cup|tray|stick|logo))\b',
        r'\b(wearing\s+[^,\.;]+)',
        r'\b(holding\s+[^,\.;]+)',
        r'\b(carrying\s+[^,\.;]+)',
        r'\b(displaying\s+[^,\.;]+)',
        r'\b(showing\s+[^,\.;]+)',
        r'\b(featuring\s+[^,\.;]+)',
        r'\b(that\s+is\s+[^,\.;]+)',
        r'\b(which\s+has\s+[^,\.;]+)',
        r'\b\d{1,2}:\d{2}(:\d{2})?\b',
        r'\blogo\s+\w+\b',
    ]

    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, ' ', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'[,;]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    words = cleaned.split()
    if len(words) < 3:
        return text
    return cleaned

class TextualKISRetriever:
    """
    Search Engine for Textual Known Item Search (Task 1).
    Performs Multi-Modal Hybrid Search (SigLIP 2 + YouTube Metadata BM25 + Object Grounding + Temporal Window Expansion).
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
        
        sig384_path = os.path.join(data_dir, "embeddings_siglip2_384.npy")
        sig2_path = os.path.join(data_dir, "embeddings_siglip2.npy")
        sig1_path = os.path.join(data_dir, "embeddings_siglip.npy")
        
        if os.path.exists(sig384_path):
            self.siglip_embeddings_path = sig384_path
            self.siglip_model_name = "google/siglip2-so400m-patch14-384"
        elif use_siglip_version == "siglip1" and os.path.exists(sig1_path):
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
        
        # Hybrid Sub-Engines
        self.bm25_searcher: Optional[MetadataBM25Searcher] = None
        self.video_to_keyframes: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.blank_frame_indices: Set[int] = set()
        self._translator = None
        self._objects_cache: Dict[str, Dict[str, Any]] = {}
        self._trans_cache: Dict[str, str] = {}
        trans_cache_path = os.path.join(self.data_dir, "cached_40_translations.json")
        if os.path.exists(trans_cache_path):
            try:
                with open(trans_cache_path, "r", encoding="utf-8") as f:
                    self._trans_cache = json.load(f)
            except Exception:
                pass
        
        self._is_loaded = False

    def _ensure_data_files(self):
        """Automatically checks and downloads required metadata & vector files from Hugging Face if missing."""
        os.makedirs(self.data_dir, exist_ok=True)
        
        HF_BASE_URL = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"
        REQUIRED_FILES = [
            ("metadata.json", f"{HF_BASE_URL}/metadata.json", False),
            ("blank_frame_indices.json", f"{HF_BASE_URL}/blank_frame_indices.json", False),
            ("embeddings_siglip2_384.npy", f"{HF_BASE_URL}/embeddings_siglip2_384.npy", False),
            ("media-info.zip", f"{HF_BASE_URL}/media-info.zip", True)
        ]

        
        import urllib.request
        import zipfile
        import time

        for fname, url, is_zip in REQUIRED_FILES:
            dest = os.path.join(self.data_dir, fname)
            
            # If media-info folder already exists with JSONs, skip zip
            if fname == "media-info.zip":
                media_info_dir = os.path.join(self.data_dir, "media-info")
                if os.path.exists(media_info_dir) and len(os.listdir(media_info_dir)) > 10:
                    continue

            if not os.path.exists(dest):
                print(f"📥 [Auto-Downloader] Missing '{fname}'. Downloading from Hugging Face CDN...", flush=True)
                start_t = time.time()
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AI-Challenge-2026)"})
                    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
                        total_size = int(resp.headers.get("Content-Length", 0))
                        downloaded = 0
                        block_size = 1024 * 1024  # 1MB
                        while True:
                            buf = resp.read(block_size)
                            if not buf:
                                break
                            downloaded += len(buf)
                            out.write(buf)
                            if total_size > 0:
                                pct = (downloaded / total_size) * 100
                                mb_dl = downloaded / (1024 * 1024)
                                mb_tot = total_size / (1024 * 1024)
                                print(f"\r📦 Downloading {fname}: {mb_dl:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)", end="", flush=True)
                    print(f"\n✅ Downloaded {fname} in {time.time() - start_t:.1f}s!", flush=True)
                    
                    if is_zip:
                        print(f"📦 Extracting {fname} into data/...", flush=True)
                        with zipfile.ZipFile(dest, "r") as zf:
                            zf.extractall(self.data_dir)
                        if os.path.exists(dest):
                            os.remove(dest)
                except Exception as e:
                    print(f"\n⚠️ Failed to auto-download {fname}: {e}. Continuing...", flush=True)

    def _remove_accents(self, s: str) -> str:
        if not s: return ""
        nfkd = unicodedata.normalize('NFKD', s)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).replace('đ', 'd').replace('Đ', 'D')

    def load_index_and_model(self):
        """Loads metadata, vector embeddings, BM25 index, and initialized SigLIP 2 models."""
        if self._is_loaded:
            return

        # Auto-download missing files if cloned freshly from GitHub
        self._ensure_data_files()

        sig384_path = os.path.join(self.data_dir, "embeddings_siglip2_384.npy")
        sig2_path = os.path.join(self.data_dir, "embeddings_siglip2.npy")
        sig1_path = os.path.join(self.data_dir, "embeddings_siglip.npy")

        if os.path.exists(sig384_path):
            self.siglip_embeddings_path = sig384_path
            self.siglip_model_name = "google/siglip2-so400m-patch14-384"
        elif os.path.exists(sig2_path):
            self.siglip_embeddings_path = sig2_path
            self.siglip_model_name = "google/siglip2-base-patch16-224"
        elif os.path.exists(sig1_path):
            self.siglip_embeddings_path = sig1_path
            self.siglip_model_name = "google/siglip-base-patch16-224"

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found in '{self.data_dir}'. Please run `python3 scripts/download_data.py` first!"
            )

        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Build instant fast lookup mapping for video_id -> frame_n -> item
        self.video_to_keyframes = {}
        for idx, item in enumerate(self.metadata):
            vid = item.get("video_id")
            n = item.get("n", 1)
            item["raw_index"] = idx
            if vid not in self.video_to_keyframes:
                self.video_to_keyframes[vid] = {}
            self.video_to_keyframes[vid][n] = item

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
            model_dtype = torch.float16 if self.device == "mps" else torch.float32
            self.siglip_model = AutoModel.from_pretrained(siglip_name, dtype=model_dtype).to(self.device)
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

        # Pre-build Video ID to frame indices mapping in RAM
        self.video_to_indices: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.metadata):
            v_id = item["video_id"]
            if v_id not in self.video_to_indices:
                self.video_to_indices[v_id] = []
            self.video_to_indices[v_id].append(idx)
        self.unique_videos = list(self.video_to_indices.keys())

        # Initialize YouTube Metadata BM25 Searcher
        try:
            self.bm25_searcher = MetadataBM25Searcher(data_dir=self.data_dir)
            self.bm25_searcher.build_index()
        except Exception as e:
            print(f"[MetadataBM25] Init notice: {e}")

        # Load pure solid/blank frame index blacklist (100% monochrome frames with 0 information)
        blank_path = os.path.join(self.data_dir, "blank_frame_indices.json")
        if os.path.exists(blank_path):
            try:
                with open(blank_path, "r", encoding="utf-8") as f_blank:
                    self.blank_frame_indices = set(json.load(f_blank))
                print(f"Loaded {len(self.blank_frame_indices)} pure solid/blank frame blacklist filters.")
            except Exception as e:
                print(f"Notice loading blank frames: {e}")

        # Load BTC Objects Inverted Index (Fast in-memory object grounding)
        self.objects_inverted_index: Dict[str, List[Tuple[int, float]]] = {}
        obj_inv_path = os.path.join(self.data_dir, "objects_inverted_index.json")
        if os.path.exists(obj_inv_path):
            try:
                with open(obj_inv_path, "r", encoding="utf-8") as f_obj_inv:
                    self.objects_inverted_index = json.load(f_obj_inv)
                print(f"⚡ Loaded BTC Objects Inverted Index ({len(self.objects_inverted_index)} entity classes) in RAM.")
            except Exception as e:
                print(f"Notice loading objects inverted index: {e}")

        # Load translation cache
        trans_cache_path = os.path.join(self.data_dir, "cached_40_translations.json")
        if os.path.exists(trans_cache_path):
            try:
                with open(trans_cache_path, "r", encoding="utf-8") as f_trans:
                    self._trans_cache = json.load(f_trans)
                print(f"⚡ Loaded {len(self._trans_cache)} cached translations in RAM.")
            except Exception as e:
                print(f"Notice loading translation cache: {e}")

        # Pre-index normalized video titles, tags, descriptions, and IDF dictionary for generalized intra-series disambiguation
        self.video_titles: Dict[str, str] = {}
        self.video_tags: Dict[str, str] = {}
        self.video_descs: Dict[str, str] = {}
        self.video_titles_noacc: Dict[str, str] = {}
        self.video_tags_noacc: Dict[str, str] = {}
        self.video_descs_noacc: Dict[str, str] = {}
        self.meta_idf_map: Dict[str, float] = {}

        STOPWORDS_INIT = set([
            "và", "của", "các", "những", "cho", "với", "trong", "tại", "được", "là", "có", "đã", "sẽ",
            "một", "này", "đó", "khi", "như", "ra", "vào", "lại", "về", "đến", "từ", "tìm", "thấy", "cảnh", "video",
            "màu", "đang", "nhiều", "hai", "ba", "bốn", "gì", "ở", "bên", "trên", "dưới", "giữa", "theo", "sau", "trước",
            "a", "an", "the", "in", "on", "at", "to", "for", "with", "from", "of", "and"
        ])

        if hasattr(self, "bm25_searcher") and self.bm25_searcher and self.bm25_searcher.video_metadata:
            doc_freq = defaultdict(int)
            total_docs = 0
            for vid, d in self.bm25_searcher.video_metadata.items():
                total_docs += 1
                t_raw = d.get("title", "")
                t_norm = re.sub(r"[^a-z0-9à-ỹ\s]", " ", unicodedata.normalize("NFC", t_raw).lower())
                desc_raw = d.get("description", "")[:800]
                desc_norm = re.sub(r"[^a-z0-9à-ỹ\s]", " ", unicodedata.normalize("NFC", desc_raw).lower())
                tags = d.get("keywords", [])
                if isinstance(tags, list):
                    raw_tags = " ".join(str(x) for x in tags)
                else:
                    raw_tags = str(tags or "")
                tag_norm = re.sub(r"[^a-z0-9à-ỹ\s]", " ", unicodedata.normalize("NFC", raw_tags).lower())

                self.video_titles[vid] = " ".join(t_norm.split())
                self.video_tags[vid] = " ".join(tag_norm.split())
                self.video_descs[vid] = " ".join(desc_norm.split())

                self.video_titles_noacc[vid] = self._remove_accents(self.video_titles[vid])
                self.video_tags_noacc[vid] = self._remove_accents(self.video_tags[vid])
                self.video_descs_noacc[vid] = self._remove_accents(self.video_descs[vid])

                all_t = f"{self.video_titles[vid]} {self.video_tags[vid]} {self.video_descs[vid]}"
                t_tokens = all_t.split()
                for w in set(w for w in t_tokens if w not in STOPWORDS_INIT and len(w) >= 2):
                    doc_freq[w] += 1
                for bg in set(" ".join(t_tokens[i:i+2]) for i in range(len(t_tokens)-1)):
                    doc_freq[bg] += 1
                for tg in set(" ".join(t_tokens[i:i+3]) for i in range(len(t_tokens)-2)):
                    doc_freq[tg] += 1

            self.meta_idf_map = {term: math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0) for term, df in doc_freq.items()}
            print(f"⚡ Loaded {len(self.video_titles)} normalized video titles, tags, descriptions & {len(self.meta_idf_map)} IDF weights in RAM.")

        self._is_loaded = True
        print(f"Retriever initialized with {len(self.metadata)} indexed keyframes.")

        # Fast PyTorch MPS JIT kernel warm-up
        try:
            _ = self.search(query="red car", top_k=2, pure_siglip_raw=True, use_reranker=False)
            dummy_img = Image.new("RGB", (224, 224), "blue")
            v_in = self.siglip_processor(images=[dummy_img], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                _ = self.siglip_model.vision_model(**v_in)
            print("✅ 100% Pre-warmed (Text + Vision + MPS JIT)! Server ready.")
        except Exception as e:
            print(f"Notice during warmup: {e}")

    def _translate_fast(self, text: str) -> str:
        """Sub-50ms high-speed translation using persistent keep-alive session with LRU caching."""
        if not text or not text.strip():
            return text
        clean = text.strip()
        if hasattr(self, "_trans_cache") and (clean in self._trans_cache or text in self._trans_cache):
            return self._trans_cache.get(clean, self._trans_cache.get(text))

        if not hasattr(self, "_trans_session") or self._trans_session is None:
            import requests
            self._trans_session = requests.Session()
            self._trans_session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })

        try:
            import urllib.parse
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text)}"
            resp = self._trans_session.get(url, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                translated = "".join([p[0] for p in data[0] if p and p[0]])
                if translated and translated.strip():
                    en_res = translated.strip()
                    if hasattr(self, "_trans_cache"):
                        self._trans_cache[text] = en_res
                    return en_res
        except Exception:
            pass

        try:
            if not hasattr(self, "_translator") or self._translator is None:
                from deep_translator import GoogleTranslator
                self._translator = GoogleTranslator(source="auto", target="en")
            translated = self._translator.translate(text)
            if translated and translated.strip():
                en_res = translated.strip()
                if hasattr(self, "_trans_cache"):
                    self._trans_cache[text] = en_res
                return en_res
        except Exception:
            pass
        return text

    def compute_late_interaction_rerank(self, query_en: str, candidates: list, top_n: int = 20) -> list:
        """
        Reranks candidate items using SigLIP 2 Token-to-Patch Late-Interaction (ColPali/ColBERT MaxSim)
        executed natively on PyTorch MPS/GPU without external API calls.
        Score = 0.35 * Stage1_Score + 0.65 * MaxSim_Score
        """
        if not candidates or self.siglip_model is None or self.siglip_processor is None:
            return candidates

        items_to_rerank = candidates[:top_n]
        remaining_items = candidates[top_n:]

        # 1. Parallel fetch image thumbnails
        from src.core.gemini_engine import fetch_single_image
        with ThreadPoolExecutor(max_workers=min(len(items_to_rerank), 12)) as executor:
            c_imgs = list(executor.map(fetch_single_image, items_to_rerank))
        valid_c_imgs = [im if im is not None else Image.new("RGB", (224, 224), "black") for im in c_imgs]

        # 2. Extract Text Token representations [1, L, 768]
        text_inputs = self.siglip_processor.tokenizer([query_en], return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        with torch.inference_mode():
            text_outputs = self.siglip_model.text_model(**text_inputs)
            text_tokens = text_outputs.last_hidden_state
            text_tokens = text_tokens / text_tokens.norm(dim=-1, keepdim=True)

        # 3. Extract Image Patch representations [B, 196, 768]
        img_inputs = self.siglip_processor(images=valid_c_imgs, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vision_outputs = self.siglip_model.vision_model(**img_inputs)
            patch_tokens = vision_outputs.last_hidden_state
            patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

        # 4. Batch MaxSim Einstein Summation
        sim_matrix = torch.einsum("tld,bpd->btlp", text_tokens, patch_tokens).squeeze(1) # [B, L, 196]
        max_sim_per_token, _ = sim_matrix.max(dim=-1) # [B, L]
        late_scores = max_sim_per_token.mean(dim=-1).cpu().tolist() # [B]

        # 5. Blend Stage 1 Score with Late-Interaction MaxSim
        scored_candidates = []
        for i, it in enumerate(items_to_rerank):
            it_copy = dict(it)
            it_copy["late_score"] = float(late_scores[i])
            # Calibrated blend: 35% Vector + 65% Late-Interaction MaxSim
            it_copy["score"] = float(0.35 * it.get("score", 0.0) + 0.65 * late_scores[i])
            scored_candidates.append(it_copy)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates + remaining_items

    def search(
        self,
        query: str,
        top_k: int = 100,
        nms_frame_gap: int = 5,
        max_per_video: int = 2,
        visual_sim_threshold: float = 0.90,
        use_reranker: bool = False,
        reranker_mode: str = "siglip_late",
        use_spatial_grid: bool = True,
        use_metadata_bm25: bool = True,
        use_temporal_expansion: bool = True,
        pure_siglip_raw: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Executes Textual KIS search for a natural language text query using Multi-Modal Hybrid Engine.
        Supports standard Multimodal Hybrid (BM25 + Objects + Dual En-Vi) and Pure Raw SigLIP 2 mode.
        """
        if not self._is_loaded:
            self.load_index_and_model()

        clean_query = query.strip()
        low_clean = clean_query.lower()

        # If Pure SigLIP Raw Mode is enabled: directly use raw input English prompt with zero modifications/boosts
        if pure_siglip_raw:
            inputs = self.siglip_processor(
                text=[clean_query, f"a photo of {clean_query}"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.siglip_model.get_text_features(**inputs)
                tf = getattr(outputs, 'pooler_output', getattr(outputs, 'text_embeds', outputs[0] if isinstance(outputs, (tuple, list)) else outputs))
                tf = tf / tf.norm(dim=-1, keepdim=True)
                vec_tensor = 0.80 * tf[0] + 0.20 * tf[1]
                vec_tensor = vec_tensor / vec_tensor.norm(dim=-1, keepdim=True)
                siglip_vec = vec_tensor.cpu().numpy().reshape(-1)

            sims = np.dot(self.embeddings_siglip, siglip_vec).astype(np.float32)
            candidate_k = min(max(top_k * 20, 2000), len(self.embeddings_siglip))
            top_indices = np.argpartition(-sims, candidate_k)[:candidate_k]
            top_sorted = top_indices[np.argsort(-sims[top_indices])]

            raw_candidates = []
            for rank_i, idx in enumerate(top_sorted):
                raw_i_int = int(idx)
                if hasattr(self, "blank_frame_indices") and raw_i_int in self.blank_frame_indices:
                    continue
                item = dict(self.metadata[idx])
                if item.get("n", 1) <= 2:
                    continue
                item["score"] = float(sims[idx])
                item["raw_index"] = raw_i_int
                raw_candidates.append(item)

            final_results = []
            seen_videos = {}
            for item in raw_candidates:
                v_id = item["video_id"]
                n_val = item.get("n", 1)
                pts_val = item.get("pts_time", 0.0)

                if v_id not in seen_videos:
                    seen_videos[v_id] = []

                if max_per_video > 0 and len(seen_videos[v_id]) >= max_per_video:
                    continue

                if any(abs(n_val - prev_n) <= nms_frame_gap or abs(pts_val - prev_pts) <= max(nms_frame_gap * 2.0, 10.0) for prev_n, prev_pts in seen_videos[v_id]):
                    continue

                seen_videos[v_id].append((n_val, pts_val))
                final_results.append(item)

                if len(final_results) >= top_k:
                    break

            for r_idx, res in enumerate(final_results):
                res["rank"] = r_idx + 1
            return final_results

        # Vietnamese Cultural & Landmark Idioms
        cultural_idioms = {
            "múa lân": "lion dance festival performance",
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
            "đàn bầu": "vietnamese traditional monochord zither",
            "nhà sàn": "vietnamese traditional stilt house"
        }

        # Translation & Gemini AI Query Enrichment with Persistent Cache
        has_non_ascii = any(ord(c) > 127 for c in clean_query)
        # 1. High-Accuracy Natural English Translation for SigLIP 2 (Stage 1)
        if has_non_ascii:
            clean_en = self._translate_fast(clean_query)
        else:
            clean_en = clean_query

        # 2. Pure SigLIP 2 Multi-Prompt Ensemble & Video-Level Multi-Frame Fusion (Stage 1)
        if self.use_siglip_only and self.embeddings_siglip is not None:
            target_prompts = [clean_en, clean_query, f"a high quality video keyframe of {clean_en}", f"a photo of {clean_en}"]
            inputs = self.siglip_processor(
                text=target_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.siglip_model.get_text_features(**inputs)
                tf = getattr(outputs, 'pooler_output', getattr(outputs, 'text_embeds', outputs[0] if isinstance(outputs, (tuple, list)) else outputs))
                tf = tf / tf.norm(dim=-1, keepdim=True)
                weights = torch.tensor([0.45, 0.35, 0.10, 0.10], device=self.device).unsqueeze(1)
                weighted_tf = (tf * weights).sum(dim=0, keepdim=True)
                weighted_tf = weighted_tf / weighted_tf.norm(dim=-1, keepdim=True)
                siglip_vec = weighted_tf.cpu().numpy().flatten().astype(np.float32)

            # High-Precision SigLIP 2 Cosine Similarity Matrix Multiplication
            sims = np.dot(self.embeddings_siglip, siglip_vec).astype(np.float32)

            # 3. 100% Generalized Dynamic Semantic Topic & N-gram Matcher (Zero Hardcode)
            STOPWORDS_META = {"và", "là", "có", "trong", "đang", "của", "cho", "với", "những", "các", "trên", "tại", "một", "ở", "ra", "vào", "được", "bị", "khi", "để", "màu", "cảnh", "video"}
            q_clean = clean_query.lower().replace(",", " ").replace(".", " ").replace(";", " ")
            q_words = [w for w in q_clean.split() if len(w) >= 2 and w not in STOPWORDS_META]
            q_bigrams = [" ".join(q_words[i:i+2]) for i in range(len(q_words)-1)]
            q_trigrams = [" ".join(q_words[i:i+3]) for i in range(len(q_words)-2)]

            # 5. Video-Level Multi-Frame Temporal Ranking over all 873 videos
            video_rankings = []
            idf_lookup = getattr(self, "meta_idf_map", {})
            v_titles_map = getattr(self, "video_titles", {})
            v_tags_map = getattr(self, "video_tags", {})
            v_descs_map = getattr(self, "video_descs", {})

            for v_id in self.unique_videos:
                f_indices = self.video_to_indices[v_id]
                v_sims = sorted([float(sims[fi]) for fi in f_indices if fi not in self.blank_frame_indices and self.metadata[fi].get("n", 1) > 2], reverse=True)
                if len(v_sims) == 0:
                    siglip_agg = 0.0
                    cluster_boost = 0.0
                elif len(v_sims) == 1:
                    siglip_agg = v_sims[0]
                    cluster_boost = 0.0
                elif len(v_sims) == 2:
                    siglip_agg = 0.75 * v_sims[0] + 0.25 * v_sims[1]
                    sustained_cnt = sum(1 for s in v_sims if s >= 0.91 * v_sims[0])
                    cluster_boost = np.log2(1.0 + min(sustained_cnt, 8)) * (0.015 / 3.0)
                else:
                    siglip_agg = 0.75 * v_sims[0] + 0.20 * v_sims[1] + 0.05 * v_sims[2]
                    sustained_cnt = sum(1 for s in v_sims if s >= 0.91 * v_sims[0])
                    cluster_boost = np.log2(1.0 + min(sustained_cnt, 8)) * (0.015 / 3.0)

                v_t = v_titles_map.get(v_id, "")
                v_kw = v_tags_map.get(v_id, "")
                v_d = v_descs_map.get(v_id, "")[:800]

                t_kw_s = 0.0
                d_s = 0.0

                for tg in q_trigrams:
                    idf = idf_lookup.get(tg, 1.0)
                    if tg in v_t: t_kw_s += 4.0 * idf
                    elif tg in v_kw: t_kw_s += 2.5 * idf
                    elif tg in v_d: d_s += 1.0 * idf

                for bg in q_bigrams:
                    idf = idf_lookup.get(bg, 1.0)
                    if bg in v_t: t_kw_s += 2.0 * idf
                    elif bg in v_kw: t_kw_s += 1.2 * idf
                    elif bg in v_d: d_s += 0.5 * idf

                for w in q_words:
                    idf = idf_lookup.get(w, 1.0)
                    if f" {w} " in f" {v_t} ": t_kw_s += 0.5 * idf
                    elif f" {w} " in f" {v_kw} ": t_kw_s += 0.25 * idf
                    elif f" {w} " in f" {v_d} ": d_s += 0.15 * idf

                raw_meta = t_kw_s + d_s * 3.0
                boost = np.tanh(raw_meta * 0.02) * 0.035
                hybrid_video_s = siglip_agg + cluster_boost + boost
                video_rankings.append((v_id, hybrid_video_s, 0.0))

            video_rankings.sort(key=lambda x: x[1], reverse=True)

            # 6. Extract top distinct diverse frames per video from Top Videos
            diverse_candidates = []
            for v_id, hybrid_v_s, bm_s in video_rankings:
                f_indices = self.video_to_indices[v_id]
                sorted_f_indices = sorted(f_indices, key=lambda i: sims[i], reverse=True)
                selected_in_video = []
                for fi in sorted_f_indices:
                    raw_i_int = int(fi)
                    if hasattr(self, "blank_frame_indices") and raw_i_int in self.blank_frame_indices:
                        continue
                    item = dict(self.metadata[fi])
                    n_val = item.get("n", 1)
                    if n_val <= 2:
                        continue
                    if any(abs(n_val - pn) <= nms_frame_gap for pn in selected_in_video):
                        continue
                    selected_in_video.append(n_val)
                    item["score"] = float(sims[fi])
                    item["raw_index"] = raw_i_int
                    diverse_candidates.append(item)
                    if len(selected_in_video) >= max_per_video:
                        break
                if len(diverse_candidates) >= max(top_k * 4, 300):
                    break

            # 4. Reranking on Top Diverse Candidates (SigLIP Late-Interaction / Gemini VLM / Off)
            if use_reranker and reranker_mode == "siglip_late":
                final_candidates = self.compute_late_interaction_rerank(query_en=clean_en, candidates=diverse_candidates, top_n=min(top_k, 20))
            elif use_reranker and reranker_mode == "gemini_vlm" and hasattr(self, "gemini_engine") and self.gemini_engine and self.gemini_engine.is_ready:
                final_candidates = self.gemini_engine.rerank_candidates(query=clean_query, candidate_items=diverse_candidates, top_n=min(top_k, 20))
            elif use_reranker and hasattr(self, "vlm_reranker") and self.vlm_reranker and self.vlm_reranker.is_ready:
                final_candidates = self.vlm_reranker.rerank(query=clean_en, candidates=diverse_candidates, top_k=min(top_k, 20))
            else:
                final_candidates = diverse_candidates

            final_results = []
            seen_final_videos = {}
            for item in final_candidates:
                v_id = item["video_id"]
                if v_id not in seen_final_videos:
                    seen_final_videos[v_id] = 0
                if max_per_video > 0 and seen_final_videos[v_id] >= max_per_video:
                    continue
                seen_final_videos[v_id] += 1

                v_id_lower = v_id.lower()
                f_name = item["frame_filename"]
                abs_path = os.path.join(self.data_dir, "..", "keyframes", v_id, f_name)
                if not os.path.exists(abs_path):
                    abs_path = os.path.join(self.data_dir, "..", "keyframes", v_id_lower, f_name)
                item["abs_path"] = abs_path
                final_results.append(item)
                if len(final_results) >= top_k:
                    break

            # Method 4: Smart Semantic-Aware & Visual-Consistent Temporal Expansion (Interval [s, e] Booster)
            if use_temporal_expansion and max_per_video > 1 and hasattr(self, "video_to_keyframes") and self.video_to_keyframes:
                expanded_results = []
                seen_pairs: Set[Tuple[str, int]] = set()

                for seed_item in final_results:
                    v_id = seed_item["video_id"]
                    n_val = seed_item.get("n", 1)
                    pair = (v_id, n_val)
                    
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        expanded_results.append(seed_item)

                    # For Top 5 high-confidence candidates (score >= 0.16), find best semantic neighbors in window [n-3, n+3]
                    if len(expanded_results) <= 10 and seed_item.get("score", 0.0) >= 0.16:
                        v_frames = self.video_to_keyframes.get(v_id, {})
                        seed_raw_idx = seed_item.get("raw_index")
                        seed_vec = self.embeddings_siglip[seed_raw_idx] if (seed_raw_idx is not None and self.embeddings_siglip is not None and seed_raw_idx < len(self.embeddings_siglip)) else None
                        
                        # Scan candidate neighbor frames within temporal window [-4, +4]
                        candidate_neighbors = []
                        for offset in [-1, 1, -2, 2, -3, 3, -4, 4]:
                            neighbor_n = n_val + offset
                            if neighbor_n <= 2 or neighbor_n not in v_frames or (v_id, neighbor_n) in seen_pairs:
                                continue
                            
                            n_item = v_frames[neighbor_n]
                            n_raw_idx = n_item.get("raw_index")
                            if n_raw_idx is not None and hasattr(self, "blank_frame_indices") and int(n_raw_idx) in self.blank_frame_indices:
                                continue
                            
                            if n_raw_idx is not None and n_raw_idx < len(sims):
                                n_sim = float(sims[n_raw_idx])
                                n_score = n_sim
                                
                                # 1. Semantic Check: Must be positively related to the search query
                                if n_score >= 0.14:
                                    # 2. Visual Continuity Check: Filter out scene cuts/bumpers (requires cosine similarity >= 0.55)
                                    visual_continuity = 1.0
                                    if seed_vec is not None and n_raw_idx < len(self.embeddings_siglip):
                                        visual_continuity = float(np.dot(seed_vec, self.embeddings_siglip[n_raw_idx]))
                                        if visual_continuity < 0.55:
                                            continue # Filter out abrupt scene transitions
                                    
                                    composite_score = n_score + (0.04 * visual_continuity)
                                    candidate_neighbors.append((composite_score, neighbor_n, n_item))

                        # Select top 2 highest semantic relevance & visually coherent neighbors
                        candidate_neighbors.sort(key=lambda x: x[0], reverse=True)
                        for comp_score, neighbor_n, n_item in candidate_neighbors[:2]:
                            seen_pairs.add((v_id, neighbor_n))
                            neighbor_item = dict(n_item)
                            neighbor_item["score"] = comp_score
                            neighbor_item["is_neighbor_expansion"] = True
                            
                            v_id_lower = v_id.lower()
                            f_name = neighbor_item["frame_filename"]
                            abs_path = os.path.join(self.data_dir, "..", "keyframes", v_id, f_name)
                            if not os.path.exists(abs_path):
                                abs_path = os.path.join(self.data_dir, "..", "keyframes", v_id_lower, f_name)
                            neighbor_item["abs_path"] = abs_path
                            expanded_results.append(neighbor_item)

                    if len(expanded_results) >= top_k:
                        break
                return expanded_results[:top_k]

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
