"""
Metadata BM25 Searcher for AIC 2026 Task 1 (Textual KIS).
Builds a lightweight in-memory BM25 index over YouTube metadata (title, description, keywords, author)
from all 873 videos in data/media-info.
"""

import os
import re
import json
import math
from typing import Dict, List, Set, Tuple

import unicodedata

# Event / Venue / Competition / Show / Landmark entities ONLY for YouTube Metadata BM25
EVENT_METADATA_ENTITIES = [
    "đua xe đạp", "cúp truyền hình", "truyền hình tp hcm", "chợ bến thành", "cầu rồng", "vịnh hạ long",
    "ngoại hạng anh", "bản tin thời sự", "thời sự", "vòng chung kết", "thành phố hồ chí minh",
    "hà nội", "đà nẵng", "sầm sơn", "tam kỳ", "huế", "đà lạt", "quảng trường thống nhất",
    "lan tỏa năng lượng tích cực", "món ngon mỗi ngày", "thế giới động vật", "phim hoạt hình",
    "múa lân", "lễ hội", "ca nhạc", "trao giải", "hội nghị"
]

# Stopwords specifically for YouTube Video Metadata Search (excludes visual adjectives/colors/generic objects)
METADATA_STOPWORDS = {
    "và", "của", "các", "những", "cho", "với", "trong", "tại", "được", "là", "có", "đã", "sẽ",
    "một", "này", "đó", "khi", "như", "ra", "vào", "lại", "về", "đến", "từ", "tìm", "thấy", "cảnh", "video",
    "màu", "đỏ", "vàng", "xanh", "trắng", "đen", "hồng", "cam", "tím", "nâu", "xám",
    "xe", "người", "con", "cái", "chiếc", "ảnh", "hình", "đang", "nhiều", "hai", "ba", "bốn",
    "red", "yellow", "blue", "green", "white", "black", "pink", "orange", "purple", "brown", "gray",
    "car", "cat", "dog", "person", "woman", "man", "and", "the", "with", "from", "for", "photo", "image", "clip", "full", "tap", "tập", "hd"
}

def tokenize_text(text: str) -> List[str]:
    """
    Enhanced Vietnamese Tokenizer for Video Metadata:
    1. Unicode NFC normalization.
    2. Event & landmark entity extraction (e.g., 'đua_xe_đạp', 'cúp_truyền_hình').
    3. Excludes generic visual stopwords (colors, objects, counters).
    """
    if not text:
        return []
    
    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text).lower()
    
    # Extract known event/landmark phrases first
    extracted_compounds = []
    text_for_tokens = text
    for phrase in EVENT_METADATA_ENTITIES:
        if phrase in text_for_tokens:
            compound_token = phrase.replace(" ", "_")
            extracted_compounds.append(compound_token)
    
    # Replace non-alphanumeric with spaces (retaining Vietnamese diacritics)
    clean = re.sub(r"[^a-z0-9à-ỹ\s]", " ", text)
    words = [w for w in clean.split() if len(w) >= 3 and w not in METADATA_STOPWORDS]
    
    all_tokens = words + extracted_compounds
    return list(dict.fromkeys(all_tokens))

class MetadataBM25Searcher:
    def __init__(self, data_dir: str = "./data", k1: float = 1.5, b: float = 0.75):
        self.data_dir = data_dir
        self.k1 = k1
        self.b = b
        
        self.doc_len: Dict[str, int] = {}       # video_id -> length
        self.avg_doc_len: float = 0.0
        self.doc_freq: Dict[str, int] = {}      # term -> count of docs containing term
        self.inverted_index: Dict[str, Dict[str, int]] = {} # term -> {video_id: term_freq}
        self.video_metadata: Dict[str, dict] = {} # video_id -> raw json data
        self.total_docs: int = 0
        self.is_ready: bool = False

    def build_index(self):
        """
        Loads all 873 metadata JSON files and constructs in-memory BM25 inverted index.
        """
        media_info_dir = os.path.join(self.data_dir, "media-info")
        if not os.path.exists(media_info_dir):
            media_info_dir = os.path.join(self.data_dir, "media_info")
        
        if not os.path.exists(media_info_dir):
            print(f"[MetadataBM25] Notice: media-info directory not found at {media_info_dir}")
            return

        json_files = [f for f in os.listdir(media_info_dir) if f.endswith(".json")]
        if not json_files:
            return

        total_len = 0
        for fname in json_files:
            video_id = os.path.splitext(fname)[0]
            fpath = os.path.join(media_info_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.video_metadata[video_id] = meta
                    
                    # Combine title (weighted 3x), keywords (weighted 2x), and description
                    title = meta.get("title", "")
                    description = meta.get("description", "")
                    keywords = meta.get("keywords", [])
                    if isinstance(keywords, list):
                        keywords_str = " ".join([str(k) for k in keywords])
                    else:
                        keywords_str = str(keywords or "")
                    
                    # Construct weighted document text
                    doc_text = f"{title} {title} {title} {keywords_str} {keywords_str} {description}"
                    tokens = tokenize_text(doc_text)
                    
                    self.doc_len[video_id] = len(tokens)
                    total_len += len(tokens)
                    
                    # Term frequencies
                    term_counts: Dict[str, int] = {}
                    for token in tokens:
                        term_counts[token] = term_counts.get(token, 0) + 1
                        
                    for term, count in term_counts.items():
                        if term not in self.inverted_index:
                            self.inverted_index[term] = {}
                        self.inverted_index[term][video_id] = count
                        self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
            except Exception:
                continue

        self.total_docs = len(self.doc_len)
        if self.total_docs > 0:
            self.avg_doc_len = total_len / self.total_docs
            self.is_ready = True
            print(f"[MetadataBM25] Built index for {self.total_docs} videos ({len(self.inverted_index)} unique terms).")

    def search(self, query: str) -> Dict[str, float]:
        """
        Calculates normalized BM25 relevance scores for all matching videos.
        Filters out generic single words to only boost specific entities/events.
        Returns: {video_id: normalized_score in [0.0, 1.0]}
        """
        if not self.is_ready or not query:
            return {}

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return {}

        has_compound = any("_" in t for t in query_tokens)
        scores: Dict[str, float] = {}
        matched_tokens_per_doc: Dict[str, int] = {}
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
                
            n_qi = self.doc_freq[token]
            idf = math.log((self.total_docs - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
            if idf <= 1.0:  # Skip generic terms that appear in too many videos
                continue
                
            posting_list = self.inverted_index[token]
            for video_id, tf in posting_list.items():
                doc_l = self.doc_len[video_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / max(self.avg_doc_len, 1.0)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-6))
                scores[video_id] = scores.get(video_id, 0.0) + term_score
                matched_tokens_per_doc[video_id] = matched_tokens_per_doc.get(video_id, 0) + 1

        # High-Precision Filter: Must match a compound phrase or at least 2 distinct specific terms
        filtered_scores: Dict[str, float] = {}
        for video_id, score_val in scores.items():
            if has_compound or matched_tokens_per_doc.get(video_id, 0) >= 2:
                filtered_scores[video_id] = score_val

        if not filtered_scores:
            return {}

        # Min-Max Normalization to [0.0, 1.0]
        max_s = max(filtered_scores.values())
        if max_s > 0:
            return {vid: s / max_s for vid, s in filtered_scores.items()}
        return filtered_scores
