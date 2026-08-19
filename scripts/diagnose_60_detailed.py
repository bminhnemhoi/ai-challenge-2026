import json
import numpy as np
import time
from src.task1_kis.retriever import TextualKISRetriever

def run_diagnostics():
    print("=" * 90)
    print("🔍 TASK 1 KIS FULL 60-SAMPLE DIAGNOSTICS & AUDIT")
    print("=" * 90)
    
    retriever = TextualKISRetriever(data_dir="data", use_siglip_only=True)
    retriever.load_index_and_model()
    
    with open("data/ground_truth.json", "r", encoding="utf-8") as f:
        gt_all = json.load(f)
        
    missed_samples = []
    low_rank_samples = [] # rank > 20
    top1_count = 0
    top5_count = 0
    top10_count = 0
    top20_count = 0
    top50_count = 0
    top100_count = 0
    
    start_time = time.time()
    
    for idx, sample in enumerate(gt_all, 1):
        q_vi = sample["kis_query_vi"]
        tgt_vid = sample["video_id"]
        tgt_frame = sample.get("frame_filename", f"{sample['n']:03d}.jpg")
        
        t0 = time.time()
        results = retriever.search(q_vi, top_k=100, max_per_video=1, use_reranker=False)
        dt = time.time() - t0
        
        found_rank = None
        found_score = 0.0
        
        for r_i, r in enumerate(results, 1):
            if r["video_id"] == tgt_vid:
                found_rank = r_i
                found_score = r.get("score", 0.0)
                break
                
        status_str = ""
        if found_rank is not None:
            if found_rank == 1:
                top1_count += 1
                status_str = f"👑 Top 1 (Score: {found_score:.4f})"
            elif found_rank <= 5:
                top5_count += 1
                status_str = f"🥇 Top {found_rank} (Score: {found_score:.4f})"
            elif found_rank <= 10:
                top10_count += 1
                status_str = f"🥈 Top {found_rank} (Score: {found_score:.4f})"
            elif found_rank <= 20:
                top20_count += 1
                status_str = f"🥉 Top {found_rank} (Score: {found_score:.4f})"
            elif found_rank <= 50:
                top50_count += 1
                status_str = f"⚡ Top {found_rank} (Score: {found_score:.4f})"
                low_rank_samples.append((idx, tgt_vid, q_vi, found_rank, found_score))
            elif found_rank <= 100:
                top100_count += 1
                status_str = f"📍 Top {found_rank} (Score: {found_score:.4f})"
                low_rank_samples.append((idx, tgt_vid, q_vi, found_rank, found_score))
            else:
                status_str = f"⚠️ Top {found_rank} (>100)"
                missed_samples.append((idx, tgt_vid, q_vi, found_rank, found_score))
        else:
            status_str = "❌ NOT FOUND IN TOP 200"
            missed_samples.append((idx, tgt_vid, q_vi, None, 0.0))
            
        print(f"[{idx:02d}] {tgt_vid} ({tgt_frame}) | {q_vi[:45]}... -> {status_str} ({dt*1000:.0f}ms)")
        
    total_time = time.time() - start_time
    avg_latency = total_time / len(gt_all)
    
    print("\n" + "=" * 90)
    print("📊 OVERALL BENCHMARK RESULTS (60 Ground Truth Samples):")
    print("=" * 90)
    print(f"  • Top 1 Accuracy:   {top1_count:2d} / 60 ({top1_count/60*100:5.1f}%)")
    print(f"  • Top 5 Accuracy:   {top1_count+top5_count:2d} / 60 ({(top1_count+top5_count)/60*100:5.1f}%)")
    print(f"  • Top 10 Accuracy:  {top1_count+top5_count+top10_count:2d} / 60 ({(top1_count+top5_count+top10_count)/60*100:5.1f}%)")
    print(f"  • Top 20 Accuracy:  {top1_count+top5_count+top10_count+top20_count:2d} / 60 ({(top1_count+top5_count+top10_count+top20_count)/60*100:5.1f}%)")
    print(f"  • Top 50 Accuracy:  {top1_count+top5_count+top10_count+top20_count+top50_count:2d} / 60 ({(top1_count+top5_count+top10_count+top20_count+top50_count)/60*100:5.1f}%)")
    print(f"  • Top 100 Accuracy: {top1_count+top5_count+top10_count+top20_count+top50_count+top100_count:2d} / 60 ({(top1_count+top5_count+top10_count+top20_count+top50_count+top100_count)/60*100:5.1f}%)")
    print(f"  • Total Missed (>100): {len(missed_samples):2d} / 60")
    print(f"  • Average Latency:     {avg_latency*1000:.1f} ms / query")
    print("=" * 90)
    
    if missed_samples:
        print("\n🚨 MISSED SAMPLES DETAILS (> Top 100):")
        for idx, vid, q, rank, score in missed_samples:
            print(f"  - Sample #{idx:02d} ({vid}): Rank={rank}, Score={score:.4f} | Query: '{q}'")
            
    if low_rank_samples:
        print("\n⚠️ LOW RANK SAMPLES (Rank 21 to 100):")
        for idx, vid, q, rank, score in low_rank_samples:
            print(f"  - Sample #{idx:02d} ({vid}): Rank={rank}, Score={score:.4f} | Query: '{q}'")

if __name__ == "__main__":
    run_diagnostics()
