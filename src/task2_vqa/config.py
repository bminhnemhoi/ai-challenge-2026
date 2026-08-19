"""Shared constants for Task 2 (Visual Q&A)."""

import os

# --- VLM (Member 2) ---
MODEL_ID = os.environ.get("AIC_VLM_MODEL_ID", "gemini-2.0-flash")
MAX_SIDE = int(os.environ.get("AIC_VLM_MAX_SIDE", "768")) 
MAX_NEW_TOKENS = int(os.environ.get("AIC_VLM_MAX_NEW_TOKENS", "24"))

# --- Pipeline chung (Member 2 + Member 3) ---
DEFAULT_VIDEO_DIR = os.environ.get("AIC_VIDEO_DIR", "./data/videos")
DEFAULT_DATA_DIR = os.environ.get("AIC_DATA_DIR", "./data")
VLM_TOP_K = int(os.environ.get("AIC_VLM_TOP_K", "5"))  
RETRIEVER_TOP_K = int(os.environ.get("AIC_RETRIEVER_TOP_K", "20")) 
FRAME_WINDOW = int(os.environ.get("AIC_FRAME_WINDOW", "2"))
FRAME_STEP = int(os.environ.get("AIC_FRAME_STEP", "5"))
