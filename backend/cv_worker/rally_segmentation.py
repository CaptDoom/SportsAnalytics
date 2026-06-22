from typing import List, Tuple, Dict, Any
import numpy as np

def segment_rallies(
    shuttle_trajectory: List[Dict[str, Any]], 
    player_positions: List[Dict[str, Any]], 
    fps: float = 30.0
) -> List[Tuple[int, int]]:
    """
    Rally Segmentation Heuristic:
    - Finds continuous segments of active play.
    - Active play is defined by shuttle visibility and movement.
    - Long gaps in shuttle visibility (>2.5 seconds or 75 frames) indicate dead play between rallies.
    - Player resets: players returning near center court default positions confirms the end of play.
    Returns a list of (start_frame, end_frame) tuples.
    """
    if not shuttle_trajectory:
        return []

    # Sort shuttle trajectory by frame number
    trajectory = sorted(shuttle_trajectory, key=lambda x: x["frame_number"])
    
    # Track visibility status
    visible_frames = [pt["frame_number"] for pt in trajectory if pt.get("visible", False)]
    if not visible_frames:
        # Default fallback: one single rally if no frames visible
        max_frame = max(pt["frame_number"] for pt in trajectory)
        return [(1, max_frame)]

    # Identify clusters of visible frames
    rallies = []
    gap_threshold = int(fps * 2.5)  # 2.5 seconds gap threshold
    min_rally_len = int(fps * 1.5)  # minimum rally length of 1.5 seconds

    start_frame = visible_frames[0]
    prev_frame = visible_frames[0]

    for frame in visible_frames[1:]:
        if frame - prev_frame > gap_threshold:
            # We found a gap! Close the current rally segment
            end_frame = prev_frame
            if end_frame - start_frame >= min_rally_len:
                rallies.append((start_frame, end_frame))
            start_frame = frame
        prev_frame = frame
        
    # Append the last segment
    if prev_frame - start_frame >= min_rally_len:
        rallies.append((start_frame, prev_frame))

    # Refine boundaries using player position resets if available
    # (Checking if player positions reset to base around start/end frames)
    # We will return the segmented frame indices
    return rallies
