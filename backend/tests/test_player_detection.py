import numpy as np
import pytest
from cv_worker.player_detection import PlayerDetector, BoundingBox

def test_player_detector_mock():
    # Initialize detector in mock mode (using a dummy path)
    detector = PlayerDetector(model_path="nonexistent_model.pt")
    assert detector.is_mock is True
    
    # Create a dummy frame (height 480, width 640, 3 channels)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run player detection
    boxes = detector.detect_players(dummy_frame)
    
    assert len(boxes) == 2
    
    # Verify both detected boxes are BoundingBox objects
    for box in boxes:
        assert isinstance(box, BoundingBox)
        assert box.confidence > 0.90
        assert box.class_id == 0  # Person
        
    # Check coordinates are within boundaries
    box_1, box_2 = boxes
    assert box_1.x_min >= 0 and box_1.x_max <= 640
    assert box_1.y_min >= 0 and box_1.y_max <= 480
    assert box_2.x_min >= 0 and box_2.x_max <= 640
    assert box_2.y_min >= 0 and box_2.y_max <= 480
