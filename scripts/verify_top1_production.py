import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
from src.task1_kis.retriever import TextualKISRetriever

def main():
    retriever = TextualKISRetriever(data_dir="data", use_siglip_only=True)
    retriever.load_index_and_model()

    with open("data/ground_truth.json", "r", encoding="utf-8") as f:
        gt_all = json.load(f)

    top1 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    latencies = []

    print("=" * 80)
    print("RUNNING PRODUCTION TextualKISRetriever.search() ON ALL 60 GROUND TRUTH SAMPLES")
    print("=" * 80)

    for idx, sample in enumerate(gt_all, 1):
        q_vi = sample["kis_query_vi"]
        tgt_vid = sample["video_id"]
        tgt_frame = sample.get("frame_filename", f"{sample['n']:03d}.jpg")

        t0 = time.perf_counter()
        results = retriever.search(query=q_vi, top_k=100, max_per_video=1, use_reranker=False)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        ranked_vids = [r["video_id"] for r in results]
        found_rank = ranked_vids.index(tgt_vid) + 1 if tgt_vid in ranked_vids else None
        target_score = next((r["score"] for r in results if r["video_id"] == tgt_vid), 0.0)

        if found_rank == 1:
            top1 += 1
            top5 += 1
            top10 += 1
            top20 += 1
            status = f"👑 Top 1 ({target_score:.4f})"
        elif found_rank and found_rank <= 5:
            top5 += 1
            top10 += 1
            top20 += 1
            status = f"🥇 Top {found_rank} ({target_score:.4f})"
        elif found_rank and found_rank <= 10:
            top10 += 1
            top20 += 1
            status = f"🥈 Top {found_rank} ({target_score:.4f})"
        elif found_rank and found_rank <= 20:
            top20 += 1
            status = f"🥉 Top {found_rank} ({target_score:.4f})"
        else:
            status = f"❌ Rank {found_rank} ({target_score:.4f})"

        print(f"[{idx:02d}] {tgt_vid} ({tgt_frame}) -> {status}")

    avg_lat = sum(latencies) / len(latencies)
    print("\n" + "=" * 80)
    print("FINAL PRODUCTION RETRIEVER RESULTS (60 Ground Truth Samples):")
    print("=" * 80)
    print(f"  • Top 1 Accuracy:   {top1:2d} / 60 ({top1/60*100:5.1f}%)")
    print(f"  • Top 5 Accuracy:   {top5:2d} / 60 ({top5/60*100:5.1f}%)")
    print(f"  • Top 10 Accuracy:  {top10:2d} / 60 ({top10/60*100:5.1f}%)")
    print(f"  • Top 20 Accuracy:  {top20:2d} / 60 ({top20/60*100:5.1f}%)")
    print(f"  • Average Latency:  {avg_lat:.1f} ms / query")
    print("=" * 80)

if __name__ == "__main__":
    main()
