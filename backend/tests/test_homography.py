import numpy as np
import pytest
from cv_worker.homography import compute_homography_matrix, pixel_to_court

def test_homography_corners():
    # Define arbitrary pixel coordinates for 4 corners of the court
    src_points = [
        (100.0, 100.0),  # Up-Left
        (400.0, 100.0),  # Up-Right
        (50.0, 500.0),   # Down-Left
        (450.0, 500.0)   # Down-Right
    ]
    
    # Compute homography matrix
    matrix = compute_homography_matrix(src_points)
    
    assert matrix is not None
    assert matrix.shape == (3, 3)
    
    # Project each source point and verify it maps to the correct corner
    ul_x, ul_y = pixel_to_court(100.0, 100.0, matrix)
    ur_x, ur_y = pixel_to_court(400.0, 100.0, matrix)
    dl_x, dl_y = pixel_to_court(50.0, 500.0, matrix)
    dr_x, dr_y = pixel_to_court(450.0, 500.0, matrix)
    
    assert pytest.approx(ul_x, abs=1e-5) == 0.0
    assert pytest.approx(ul_y, abs=1e-5) == 0.0
    
    assert pytest.approx(ur_x, abs=1e-5) == 1.0
    assert pytest.approx(ur_y, abs=1e-5) == 0.0
    
    assert pytest.approx(dl_x, abs=1e-5) == 0.0
    assert pytest.approx(dl_y, abs=1e-5) == 1.0
    
    assert pytest.approx(dr_x, abs=1e-5) == 1.0
    assert pytest.approx(dr_y, abs=1e-5) == 1.0
