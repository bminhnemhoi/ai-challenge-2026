import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
from src.task1_kis.retriever import TextualKISRetriever

print("Loading official TextualKISRetriever...", flush=True)
t0 = time.time()
retriever = TextualKISRetriever(data_dir="data", use_siglip_only=True)
retriever.load_index_and_model()
print(f"Loaded in {time.time() - t0:.2f}s", flush=True)

with open("data/ground_truth.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)

print(f"Loaded {len(gt_data)} ground truth queries.\n", flush=True)

top1, top5, top10, top20 = 0, 0, 0, 0
latencies = []

print("="*90)
print(f"{'#':<3} | {'Target Video':<12} | {'Rank':<6} | {'Status':<8} | {'Latency':<8} | {'Top 1 Match Video'}")
print("="*90)

for i, sample in enumerate(gt_data):
    tgt = sample["video_id"]
    q_vi = sample["kis_query_vi"]
    
    t_start = time.time()
    results = retriever.search(query=q_vi, top_k=100)
    dur_ms = (time.time() - t_start) * 1000
    latencies.append(dur_ms)
    
    ranked_vids = []
    seen = set()
    for r in results:
        vid = r.get("video_id")
        if vid and vid not in seen:
            seen.add(vid)
            ranked_vids.append(vid)
            
    rank = ranked_vids.index(tgt) + 1 if tgt in ranked_vids else 999
    
    if rank == 1:
        top1 += 1; top5 += 1; top10 += 1; top20 += 1
        status = "TOP 1 🎯"
    elif rank <= 5:
        top5 += 1; top10 += 1; top20 += 1
        status = "TOP 5 ✨"
    elif rank <= 10:
        top10 += 1; top20 += 1
        status = "TOP 10 ✅"
    elif rank <= 20:
        top20 += 1
        status = f"TOP 20 (R{rank})"
    else:
        status = f"MISS (R{rank})"
        
    top1_vid = ranked_vids[0] if ranked_vids else "None"
    print(f"Q{i+1:02d} | {tgt:<12} | {rank:<6} | {status:<8} | {dur_ms:6.1f}ms | {top1_vid}")

print("="*90)
print(f"📊 OFFICIAL RETRIEVER ACCURACY SUMMARY (SigLIP 2 SO400M-384):")
print(f"   🎯 Top 1 Accuracy : {top1:2d} / {len(gt_data)} ({top1/len(gt_data)*100:6.2f}%)")
print(f"   ✨ Top 5 Accuracy : {top5:2d} / {len(gt_data)} ({top5/len(gt_data)*100:6.2f}%)")
print(f"   ✅ Top 10 Accuracy: {top10:2d} / {len(gt_data)} ({top10/len(gt_data)*100:6.2f}%)")
print(f"   🌟 Top 20 Accuracy: {top20:2d} / {len(gt_data)} ({top20/len(gt_data)*100:6.2f}%)")
print(f"   ⚡ Average Latency: {sum(latencies)/len(latencies):.2f} ms/query")
print("="*90)
