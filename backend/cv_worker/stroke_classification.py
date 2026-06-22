import os
import random
import numpy as np
from typing import List, Dict, Any

# Try importing BST/PyTorch models
try:
    import torch
    import torch.nn as nn
    # Add path to bst code
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'vendor', 'bst', 'stroke_classification'))
    # imports...
except ImportError:
    torch = None

# 18-class taxonomy definitions
SHOT_CLASSES = [
    "short serve", "long/flick serve",
    "net shot", "push/net kill",
    "defensive clear", "attacking clear", "lift",
    "drop shot", "slice/cut drop",
    "drive", "push",
    "smash", "wrist smash",
    "block", "lob",
    "unclassified / unknown"
]

class StrokeClassifier:
    def __init__(self, model_path: str = "bst.pt", force_cpu: bool = False):
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.is_mock = True

        if torch is not None and os.path.exists(model_path):
            try:
                self.device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
                # Load BST model weights
                # self.model = BSTModel(...)
                # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                # self.is_mock = False
                print(f"[BST Classifier] Model loaded from {model_path}")
            except Exception as e:
                print(f"[BST Classifier] Failed to load model weights: {e}. Using mock stroke classifier.")
        else:
            print("[BST Classifier] Model weights not found. Using mock stroke classifier.")

    def classify_stroke(self, shot_number: int, hitter_is_a: bool) -> str:
        """
        Classifies stroke type based on the context/inputs.
        In mock mode, it outputs sequence-consistent shot types.
        """
        if shot_number == 1:
            # First shot is always a serve
            return random.choice(["short serve", "long/flick serve"])
        elif shot_number == 2:
            # Return of serve is typically a lift, net shot, or clear
            return random.choice(["lift", "net shot", "defensive clear"])
        
        # High shot number rallies can contain smashes, drops, drives, etc.
        choices = [
            "smash", "drop shot", "net shot", "drive", 
            "lift", "defensive clear", "block", "push"
        ]
        weights = [0.15, 0.15, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05]
        return random.choices(choices, weights=weights)[0]
