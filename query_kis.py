#!/usr/bin/env python3
import os
import sys
import argparse
from src.task1_kis import TextualKISRetriever

def main():
    parser = argparse.ArgumentParser(description="AIC 2026 - Task 1 Textual KIS CLI Search")
    parser.add_argument("query", nargs="?", type=str, help="Text query to search for")
    parser.add_argument("--query", "-q", dest="named_query", type=str, help="Text query to search for")
    parser.add_argument("--top_k", "-k", type=int, default=10, help="Number of results to show (default: 10)")
    parser.add_argument("--nms", type=int, default=5, help="NMS frame gap (default: 5)")
    args = parser.parse_args()

    query_str = args.named_query or args.query

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    retriever = TextualKISRetriever(data_dir=data_dir, use_siglip_only=True, use_siglip_version="siglip2")
    retriever.load_index_and_model()

    if query_str:
        run_search(retriever, query_str, args.top_k, args.nms)
    else:
        print("\n=== AIC 2026 Task 1: Textual KIS Search Interactive CLI ===")
        print("Type your query (or 'exit' / 'q' to quit):\n")
        while True:
            try:
                user_input = input("Query > ").strip()
                if not user_input or user_input.lower() in ["exit", "q", "quit"]:
                    break
                run_search(retriever, user_input, args.top_k, args.nms)
            except (KeyboardInterrupt, EOFError):
                break

def run_search(retriever, query, top_k, nms_gap):
    print(f"\nSearching for: '{query}'...")
    results = retriever.search(query, top_k=top_k, nms_frame_gap=nms_gap)
    
    print(f"\nTop {len(results)} Results:")
    print("-" * 65)
    print(f"{'Rank':<6} | {'Video ID':<12} | {'Frame ID':<10} | {'Score':<8} | Path")
    print("-" * 65)
    for idx, item in enumerate(results, 1):
        print(f"#{idx:<5} | {item['video_id']:<12} | {item['frame_idx']:<10} | {item['score']:<8.4f} | {item['rel_path']}")
    print("-" * 65 + "\n")

if __name__ == "__main__":
    main()
