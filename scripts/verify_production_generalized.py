import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
from src.task1_kis.retriever import TextualKISRetriever

def verify():
    retriever = TextualKISRetriever(data_dir="data", use_siglip_only=True)
    retriever.load_index_and_model()

    with open("data/ground_truth.json", "r", encoding="utf-8") as f:
        gt_all = json.load(f)

    top1, top5, top10, top20 = 0, 0, 0, 0
    latencies = []

    print(f"\n=======================================================")
    print(f"  VERIFYING PRODUCTION RETRIEVER (100% GENERALIZED)   ")
    print(f"=======================================================\n")

    for idx, sample in enumerate(gt_all, 1):
        q_vi = sample["kis_query_vi"]
        tgt_vid = sample["video_id"]

        t0 = time.perf_counter()
        results = retriever.search(
            query=q_vi,
            top_k=20,
            max_per_video=1,
            use_reranker=False,
            use_temporal_expansion=False
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        ranked_vids = [item["video_id"] for item in results]
        found_rank = ranked_vids.index(tgt_vid) + 1 if tgt_vid in ranked_vids else 999

        if found_rank == 1:
            top1 += 1
            top5 += 1
            top10 += 1
            top20 += 1
            icon = "👑 Top 1 "
        elif found_rank <= 5:
            top5 += 1
            top10 += 1
            top20 += 1
            icon = f"🥇 Top {found_rank:2d}"
        elif found_rank <= 10:
            top10 += 1
            top20 += 1
            icon = f"🥈 Top {found_rank:2d}"
        elif found_rank <= 20:
            top20 += 1
            icon = f"🥉 Top {found_rank:2d}"
        else:
            icon = f"❌ Rank {found_rank:2d}"

        print(f"[{idx:02d}/60] {tgt_vid} -> {icon} | {elapsed_ms:5.1f}ms | {q_vi[:65]}...")

    avg_lat = sum(latencies) / len(latencies)
    print("\n" + "=" * 60)
    print("                 FINAL BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Top 1  Accuracy : {top1:2d} / 60 ({top1/60*100:5.2f}%)")
    print(f"  Top 5  Accuracy : {top5:2d} / 60 ({top5/60*100:5.2f}%)")
    print(f"  Top 10 Accuracy : {top10:2d} / 60 ({top10/60*100:5.2f}%)")
    print(f"  Top 20 Accuracy : {top20:2d} / 60 ({top20/60*100:5.2f}%)")
    print(f"  Average Latency : {avg_lat:5.1f} ms / query (Target < 350ms)")
    print("=" * 60)

if __name__ == "__main__":
    verify()
