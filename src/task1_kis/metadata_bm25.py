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

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into lowercase unigrams and bigrams, stripping special characters.
    """
    if not text:
        return []
    text = text.lower()
    # Replace non-alphanumeric with spaces (retaining Vietnamese diacritics)
    clean = re.sub(r"[^\w\s\d]", " ", text)
    words = [w for w in clean.split() if len(w) >= 2]
    
    # Filter common generic stopwords
    stopwords = {
        "và", "của", "các", "những", "cho", "với", "trong", "tại", "được", "là", "có", "đã", "sẽ",
        "and", "the", "with", "from", "for", "photo", "image", "video", "clip", "full", "tap", "tập"
    }
    filtered_words = [w for w in words if w not in stopwords]
    
    # Add adjacent bigrams for strong entity/phrase matching (e.g., "đua_xe", "bến_thành")
    bigrams = [f"{filtered_words[i]}_{filtered_words[i+1]}" for i in range(len(filtered_words) - 1)]
    return filtered_words + bigrams

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
        Returns: {video_id: normalized_score in [0.0, 1.0]}
        """
        if not self.is_ready or not query:
            return {}

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return {}

        scores: Dict[str, float] = {}
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
                
            n_qi = self.doc_freq[token]
            idf = math.log((self.total_docs - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
            if idf <= 0:
                continue
                
            posting_list = self.inverted_index[token]
            for video_id, tf in posting_list.items():
                doc_l = self.doc_len[video_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / max(self.avg_doc_len, 1.0)))
                term_score = idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-6))
                scores[video_id] = scores.get(video_id, 0.0) + term_score

        if not scores:
            return {}

        # Min-Max Normalization to [0.0, 1.0]
        max_s = max(scores.values())
        if max_s > 0:
            return {vid: s / max_s for vid, s in scores.items()}
        return scores
