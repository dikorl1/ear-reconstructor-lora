#qc/prep_qc.py
#– реализация описанной QC‑логики: VarLaplacian, blockiness, шум, min‑size, ear‑box > 200 px и т.д.
#– запись qcreport.csv со статусами OK/WARN/REJECT.

"""
qc/prep_qc.py - Quality Control Pipeline for Dataset Preparation

Implements automatic QC checks based on requirements:
- VarLaplacian for sharpness (WARN > 50, REJECT < 50)
- FFT analysis for moiré patterns
- JPEG blockiness detection (8×8 blocks)
- Noise floor computation
- Automatic rejection of low-quality samples
- CSV report generation
"""

import os
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2
from PIL import Image
from scipy import signal, fft


# ============================================================================
# QC Status & Thresholds
# ============================================================================

class QCStatus(Enum):
    """Quality control status"""
    PASS = "PASS"
    WARN = "WARN"
    REJECT = "REJECT"


@dataclass
class QCThresholds:
    """QC decision thresholds"""
    # VarLaplacian (sharpness)
    var_laplacian_warn: float = 50.0
    var_laplacian_reject: float = 30.0
    
    # JPEG blockiness (0-1, where 1 = strong blockiness)
    blockiness_warn: float = 0.4
    blockiness_reject: float = 0.6
    
    # Noise floor (0-1)
    noise_warn: float = 0.15
    noise_reject: float = 0.25
    
    # Resolution
    min_width: int = 512
    min_height: int = 512
    
    # File size
    min_file_size_kb: int = 50
    max_file_size_kb: int = 50000


@dataclass
class QCResult:
    """Result of QC analysis"""
    path: str
    status: QCStatus
    
    # Metrics
    var_laplacian: float = 0.0
    blockiness: float = 0.0
    noise_floor: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    file_size_kb: float = 0.0
    
    # Details
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


# ============================================================================
# Image Metrics
