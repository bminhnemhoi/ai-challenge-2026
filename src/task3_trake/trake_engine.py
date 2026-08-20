"""
Task 3: Temporal Retrieval and Alignment of Key Events (TRAKE) Engine for AIC 2026.
Matches a chronological sequence of N semantic events (E_1, E_2, ..., E_N) to video keyframe timelines.
Format output: video_id, frame_idx_1, frame_idx_2, ..., frame_idx_N
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from src.task1_kis import TextualKISRetriever

class TRAKEEngine:
    """
    Temporal Sequence Alignment Engine using:
    1. Multi-Event Joint Video Voting across SigLIP 2 & BM25 indices.
    2. Exact Monotonic Dynamic Programming (Viterbi Time Warping) for strict chronological order: t_1 < t_2 < ... < t_N.
    """
    def __init__(self, data_dir: str, retriever: Optional[TextualKISRetriever] = None):
        self.data_dir = data_dir
        self.retriever = retriever or TextualKISRetriever(data_dir=data_dir, use_siglip_only=True, use_siglip_version="siglip2")
        self._is_loaded = False

    def load_index(self):
        """Loads underlying vector index and metadata."""
        if self._is_loaded:
            return
        if not self.retriever._is_loaded:
            self.retriever.load_index_and_model()
        self._is_loaded = True

    def align_sequence(
        self,
        event_descriptions: List[str],
        video_id: Optional[str] = None,
        top_k: int = 100
    ) -> Dict[str, Any]:
        """
        Finds the best video and optimal chronological sequence of keyframes matching N events.

        Args:
            event_descriptions: List of N event descriptions in chronological order
            video_id: Optional specific video ID to align against
            top_k: Number of candidate video alignments to return

        Returns:
            Dict containing best alignment details and list of candidate sequence predictions.
        """
        if not self._is_loaded:
            self.load_index()

        cleaned_events = [e.strip() for e in event_descriptions if e.strip()]
        if not cleaned_events:
            return {"count": 0, "alignments": [], "predictions": []}

        num_events = len(cleaned_events)

        # 1. Joint Video Scoring
        # Query retriever for each event to aggregate candidate videos
        event_query_results = []
        video_event_scores: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        for ev_idx, event_text in enumerate(cleaned_events):
            res = self.retriever.search(query=event_text, top_k=200, nms_frame_gap=1, max_per_video=20)
            event_query_results.append(res)
            for item in res:
                v = item["video_id"]
                if v not in video_event_scores:
                    video_event_scores[v] = {i: [] for i in range(num_events)}
                video_event_scores[v][ev_idx].append(item)

        # Calculate joint video affinity score
        video_ranked_list = []
        target_vids = [video_id] if video_id and video_id in self.retriever.video_to_keyframes else list(self.retriever.video_to_keyframes.keys())

        for v in target_vids:
            if v not in self.retriever.video_to_keyframes:
                continue
            
            # Event coverage score
            cov_score = 0.0
            matched_events = 0
            if v in video_event_scores:
                for ev_idx in range(num_events):
                    items = video_event_scores[v][ev_idx]
                    if items:
                        cov_score += max(it["score"] for it in items)
                        matched_events += 1

            if matched_events > 0 or video_id:
                video_ranked_list.append((v, cov_score, matched_events))

        video_ranked_list.sort(key=lambda x: (x[2], x[1]), reverse=True)
        top_candidate_vids = [v[0] for v in video_ranked_list[:max(top_k, 20)]]

        if not top_candidate_vids and video_id:
            top_candidate_vids = [video_id]

        alignments = []
        predictions = []

        HF_CDN_BASE = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

        # 2. Dynamic Programming Monotonic Alignment for each candidate video
        for v in top_candidate_vids:
            v_frames_dict = self.retriever.video_to_keyframes.get(v, {})
            if not v_frames_dict:
                continue

            # Sort keyframes chronologically by n / pts_time
            sorted_keyframes = sorted(v_frames_dict.values(), key=lambda x: x.get("n", 0))
            num_frames = len(sorted_keyframes)
            if num_frames < num_events:
                continue

            # Compute similarity matrix (N x M)
            sim_matrix = np.zeros((num_events, num_frames), dtype=np.float32)

            for ev_idx in range(num_events):
                ev_items = video_event_scores.get(v, {}).get(ev_idx, [])
                frame_to_score = {it["n"]: it["score"] for it in ev_items}
                for f_idx, frame_data in enumerate(sorted_keyframes):
                    fn = frame_data.get("n", 0)
                    sim_matrix[ev_idx, f_idx] = frame_to_score.get(fn, 0.01)

            # DP Table: dp[i, j] = max score of aligning first (i+1) events ending at frame j
            dp = np.full((num_events, num_frames), -1e9, dtype=np.float32)
            parent = np.full((num_events, num_frames), -1, dtype=np.int32)

            # Base case: event 0
            dp[0, :] = sim_matrix[0, :]

            # DP transitions: dp[i, j] = sim[i, j] + max_{k < j} dp[i-1, k]
            for i in range(1, num_events):
                max_prev = -1e9
                best_k = -1
                for j in range(i, num_frames):
                    # update best previous step k < j
                    if dp[i - 1, j - 1] > max_prev:
                        max_prev = dp[i - 1, j - 1]
                        best_k = j - 1

                    if best_k != -1:
                        dp[i, j] = sim_matrix[i, j] + max_prev
                        parent[i, j] = best_k

            # Best ending frame for event (N - 1)
            best_last_idx = int(np.argmax(dp[num_events - 1, :]))
            best_total_score = float(dp[num_events - 1, best_last_idx])

            if best_total_score <= -1e8:
                continue

            # Backtrack optimal sequence
            aligned_sequence = []
            curr_idx = best_last_idx
            for i in range(num_events - 1, -1, -1):
                frame_obj = sorted_keyframes[curr_idx]
                aligned_sequence.append({
                    "event_index": i + 1,
                    "event_description": cleaned_events[i],
                    "video_id": v,
                    "n": frame_obj.get("n", 0),
                    "frame_idx": frame_obj.get("frame_idx", 0),
                    "frame_filename": frame_obj.get("frame_filename", f"{frame_obj.get('n', 0):03d}.jpg"),
                    "pts_time": frame_obj.get("pts_time", 0.0),
                    "cdn_url": f"{HF_CDN_BASE}/{v}/{frame_obj.get('frame_filename', f'{frame_obj.get(\"n\", 0):03d}.jpg')}",
                    "sim_score": round(float(sim_matrix[i, curr_idx]), 4)
                })
                curr_idx = parent[i, curr_idx]

            aligned_sequence.reverse()

            frame_indices_list = [item["frame_idx"] for item in aligned_sequence]
            avg_score = round(best_total_score / num_events, 4)

            alignments.append({
                "video_id": v,
                "score": avg_score,
                "frame_indices": frame_indices_list,
                "sequence": aligned_sequence
            })

            predictions.append({
                "video_id": v,
                "frame_indices": frame_indices_list,
                "score": avg_score
            })

        alignments.sort(key=lambda x: x["score"], reverse=True)
        predictions.sort(key=lambda x: x["score"], reverse=True)

        return {
            "events_count": num_events,
            "count": len(alignments),
            "alignments": alignments[:top_k],
            "predictions": predictions[:top_k]
        }

