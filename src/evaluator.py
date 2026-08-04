import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class AIC2026Evaluator:
    """
    AIC 2026 Official Evaluation Score Calculator
    Support Textual KIS, Visual Q&A, and TRAKE scoring metrics.
    """

    @staticmethod
    def calculate_kis_r_score(
        pred_video: str,
        pred_frame: int,
        gt_video: str,
        gt_start_frame: int,
        gt_end_frame: int
    ) -> float:
        """
        Calculates R-Score for Textual KIS.
        R-Score = 1 if pred_video == gt_video AND gt_start_frame <= pred_frame <= gt_end_frame
        else 0.
        """
        if pred_video == gt_video and (gt_start_frame <= pred_frame <= gt_end_frame):
            return 1.0
        return 0.0

    @staticmethod
    def calculate_vqa_r_score(
        pred_video: str,
        pred_frame: int,
        pred_answer: str,
        gt_video: str,
        gt_start_frame: int,
        gt_end_frame: int,
        gt_answer: str
    ) -> float:
        """
        Calculates R-Score for Visual Q&A.
        R-Score = 1 if pred_video == gt_video AND pred_frame in range AND pred_answer semantically matches gt_answer.
        """
        if pred_video != gt_video:
            return 0.0
        if not (gt_start_frame <= pred_frame <= gt_end_frame):
            return 0.0
        
        # Simple semantic match (case-insensitive strip comparison)
        clean_pred = str(pred_answer).strip().lower()
        clean_gt = str(gt_answer).strip().lower()
        if clean_pred == clean_gt or clean_pred in clean_gt or clean_gt in clean_pred:
            return 1.0
        return 0.0

    @staticmethod
    def calculate_trake_r_score(
        pred_video: str,
        pred_frames: List[int],
        gt_video: str,
        gt_intervals: List[Tuple[int, int]]
    ) -> float:
        """
        Calculates R-Score for TRAKE (Temporal Alignment).
        R-Score = 0 if pred_video != gt_video.
        Else R-Score = (number of correctly aligned keyframes) / N.
        """
        if pred_video != gt_video:
            return 0.0
        
        N = len(gt_intervals)
        if N == 0 or len(pred_frames) != N:
            return 0.0
        
        matched_count = 0
        for j in range(N):
            p_frame = pred_frames[j]
            s_j, e_j = gt_intervals[j]
            if s_j <= p_frame <= e_j:
                matched_count += 1
                
        return float(matched_count / N)

    @staticmethod
    def calculate_final_score(r_scores_list: List[float]) -> Dict[str, float]:
        """
        Calculates R@k for k in {1, 5, 20, 50, 100} and Final Score (mean of 5 R@k values).
        r_scores_list: List of R-Scores of up to 100 predictions ranked in order.
        """
        # Ensure max 100 predictions
        scores = r_scores_list[:100]
        if not scores:
            return {"R@1": 0.0, "R@5": 0.0, "R@20": 0.0, "R@50": 0.0, "R@100": 0.0, "FinalScore": 0.0}

        k_thresholds = [1, 5, 20, 50, 100]
        r_at_k = {}
        
        for k in k_thresholds:
            # Top-k R-Score is the max R-Score among first k predictions
            sub_scores = scores[:min(k, len(scores))]
            r_at_k[f"R@{k}"] = max(sub_scores) if sub_scores else 0.0

        final_score = float(np.mean([r_at_k[f"R@{k}"] for k in k_thresholds]))
        r_at_k["FinalScore"] = final_score
        return r_at_k
