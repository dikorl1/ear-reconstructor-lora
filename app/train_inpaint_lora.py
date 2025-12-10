#train_inpaint_lora.py
#– Stage B: обучение LoRA для инпейнта уха (U‑Net only, text encoder off по умолчанию).
#– Работа с парами image, mask, masked; поддержка synthetic/ручных масок.

"""
train_inpaint_lora.py - Stage B: Inpaint LoRA training for ear reconstruction

Two-stage approach:
1. Load Stage A LoRA weights (DreamBooth prior)
2. Fine-tune on masked image pairs (image + mask + masked)
3. U-Net only training (text encoder frozen)

Key parameters from requirements:
- LR: 5e-5 to 1e-4
- LoRA rank: 16-32
- Batch size: 2 (on 16GB GPU)
- Steps: 600-1200
- Denoise strength: 0.35-0.55 for inference
- CFG scale: 4.5-5.5
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import get_peft_model, LoraConfig, PeftModel
from accelerate import Accelerator
from tqdm.auto import tqdm

from datasets import InpaintSetOnTheFlyDataset


# ============================================================================
# Configuration
# ============================================================================

class TrainConfig:
    """Training configuration for Stage B Inpaint LoRA"""
    
    def __init__(self):
        # Model
        self.pretrained_model_name_or_path = "SG161222/RealisticVisionV4.0"
        self.stage_a_lora_path = "./outputs/stage_a_lora/checkpoint-1200"
        
        # LoRA
        self.lora_rank = 32
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        
        # Training
        self.learning_rate = 5e-5
        self.train_batch_size = 2
        self.gradient_accumulation_steps = 2
        self.max_train_steps = 1000
        self.num_train_epochs = 10
        self.checkpointing_steps = 100
        
        # Data
        self.resolution = 1024
        self.train_data_paths = []  # Will be populated from args
        
        # Inference
        self.num_validation_images = 4
    
