import numpy as np
import cv2
from typing import List, Tuple, Optional

def compute_homography_matrix(src_points: List[Tuple[float, float]], dst_points: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    Computes the 3x3 homography matrix mapping pixel points to court coordinates.
    Default destination coordinates map to [0,1] x [0,1] box.
    """
    if len(src_points) != 4:
        raise ValueError("Exactly 4 source points are required (Up-Left, Up-Right, Down-Left, Down-Right).")
    
    src = np.array(src_points, dtype=np.float32)
    
    if dst_points is None:
        # Map to normalized [0,1] x [0,1]
        dst = np.array([
            [0.0, 0.0],  # Up-Left
            [1.0, 0.0],  # Up-Right
            [0.0, 1.0],  # Down-Left
            [1.0, 1.0]   # Down-Right
        ], dtype=np.float32)
    else:
        dst = np.array(dst_points, dtype=np.float32)
        
    matrix, _ = cv2.findHomography(src, dst)
    return matrix

def pixel_to_court(x: float, y: float, matrix: np.ndarray) -> Tuple[float, float]:
    """
    Applies the homography matrix to project 2D pixel coordinates to normalized court coordinates.
    """
    if matrix is None:
        return 0.0, 0.0
    
    point = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, matrix)
    return float(projected[0][0][0]), float(projected[0][0][1])
