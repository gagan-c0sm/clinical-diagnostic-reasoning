"""
D2 Script: Download supplementary training datasets.

Run with:
    .\\venv\\Scripts\\python.exe scripts\\download_datasets.py [--datasets medcase medical_o1 direct_sft]
"""

import sys
import os
from pathlib import Path

# Set all Hugging Face caches before importing datasets/huggingface_hub.
project_root = Path(__file__).resolve().parent.parent
cache_root = project_root / "data" / ".hf_cache"
os.environ["HF_HOME"] = str(cache_root)
os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
os.environ["HF_XET_CACHE"] = str(cache_root / "xet")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download import main

if __name__ == "__main__":
    main()
