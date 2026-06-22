import pytest
from cv_worker.tracking import ShuttleTracker, ShuttlePoint

def test_shuttle_tracker_mock():
    # Initialize tracker in mock mode (using a dummy path)
    tracker = ShuttleTracker(model_path="nonexistent_weights.pth")
    assert tracker.is_mock is True
    
    # Run shuttle tracking on a dummy video path
    points = tracker.track_shuttle("dummy_video.mp4")
    
    assert len(points) == 150
    
    # Check that points are BoundingBox/ShuttlePoint objects and have correct fields
    for pt in points:
        assert isinstance(pt, ShuttlePoint)
        assert hasattr(pt, 'frame_number')
        assert hasattr(pt, 'x')
        assert hasattr(pt, 'y')
        assert hasattr(pt, 'visibility')
        assert hasattr(pt, 'confidence')
        
    # Verify the parabolic mock values
    p1 = points[0]
    assert p1.frame_number == 1
    assert p1.visibility is True
    
    # Occluded frame test
    p_occ = points[54] # Frame 55 should be occluded
    assert p_occ.frame_number == 55
    assert p_occ.visibility is False
    assert p_occ.x == 0.0
    assert p_occ.y == 0.0
