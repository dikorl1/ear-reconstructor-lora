#datasets.py
#– классы ImgSetDataset, InpaintSetOnTheFlyDataset и билдеры путей, которые у тебя уже описаны в Trebovaniia-k-kodu…
#– функции для загрузки списков путей, генерации синтетических масок (по типу synthetic_ear_mask), сборки train/val наборов по manifests.
"""
datasets.py - Data loading and preprocessing for Stage A & B training

Supports:
- ImgSetDataset: Simple image loading for Stage A (DreamBooth)
- InpaintSetOnTheFlyDataset: On-the-fly mask generation for Stage B (Inpaint)
- JSONL manifest parsing for age buckets, pose, view, prompts
"""

import os
import glob
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import Dataset


# ============================================================================
# Configuration
# ============================================================================

ROOT = "/content/drive/MyDrive/datasets"  # Change to your dataset path
RESOLUTION = 1024
MIN_SIDE = 512
MIN_EAR_BOX = 200  # Minimum ear bounding box size in pixels


@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    resolution: int = RESOLUTION
    center_crop: bool = True
    pad_mode: str = "reflect"
    pad_margin_px: int = 40


# ============================================================================
# Stage A: ImgSetDataset (DreamBooth-LoRA Prior)
# ============================================================================

class ImgSetDataset(Dataset):
    """
    Simple dataset for Stage A DreamBooth training.
    Loads images and applies resizing/cropping.
    """
    
    def __init__(
        self,
        paths: List[str],
        resolution: int = RESOLUTION,
        center_crop: bool = True,
    ):
        self.paths = paths
        self.resolution = resolution
        self.center_crop = center_crop
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading
