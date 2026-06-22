import os
import time
from uuid import UUID
from celery import Celery
from sqlmodel import Session, select
from shared.config import settings
from shared.database import engine
from shared.models import Match, ShuttleTrajectory, PlayerPosition
from cv_worker.tracking import ShuttleTracker
from cv_worker.player_detection import PlayerDetector
from cv_worker.homography import pixel_to_court

# Initialize Celery app
celery_app = Celery(
    "tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True
)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_match_cv_stage1(self, match_id: str):
    """
    CV Pipeline Stage 1:
    - Set status to 'processing_cv'
    - Run Shuttle Tracking
    - Run Player Bounding Box Detection
    - Store coordinates in database (pixel space + normalized court space using homography matrix)
    """
    print(f"[Celery] Starting CV Stage 1 for Match: {match_id}")
    
    with Session(engine) as session:
        match_uuid = UUID(match_id) if isinstance(match_id, str) else match_id
        match = session.get(Match, match_uuid)
        if not match:
            print(f"[Celery Error] Match {match_id} not found.")
            return False

        try:
            match.processing_status = "processing_cv"
            session.add(match)
            session.commit()

            # Retrieve homography matrix from match calibration data
            homography_matrix = None
            if match.court_calibration and "homography_matrix" in match.court_calibration:
                import numpy as np
                homography_matrix = np.array(match.court_calibration["homography_matrix"])

            # 1. Run Shuttle Tracker
            tracker = ShuttleTracker(
                model_path=settings.TRACKNET_WEIGHTS_PATH,
                force_cpu=settings.FORCE_CPU,
                vram_limit_mb=settings.VRAM_LIMIT_MB
            )
            shuttle_points = tracker.track_shuttle(match.video_uri or "dummy.mp4")

            # Store Shuttle Trajectories
            for pt in shuttle_points:
                # Map coordinates using homography
                court_x, court_y = None, None
                if homography_matrix is not None and pt.visibility:
                    court_x, court_y = pixel_to_court(pt.x, pt.y, homography_matrix)

                trajectory_row = ShuttleTrajectory(
                    match_id=match.match_id,
                    frame_number=pt.frame_number,
                    pixel_x=pt.x,
                    pixel_y=pt.y,
                    court_x=court_x,
                    court_y=court_y,
                    visible=pt.visibility
                )
                session.add(trajectory_row)

            # 2. Run Player Detector
            detector = PlayerDetector(
                model_path=settings.YOLO_WEIGHTS_PATH,
                force_cpu=settings.FORCE_CPU
            )

            # We simulate frame-by-frame player detection (e.g. for the duration of the tracking frames)
            # Create player positions entries
            max_frames = len(shuttle_points)
            for frame_idx in range(1, max_frames + 1):
                # Standard empty dummy frame for mock or real input
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                boxes = detector.detect_players(dummy_frame)

                # Store Player Bounding Boxes & Positions
                for player_idx, box in enumerate(boxes):
                    player_id = match.player_a_id if player_idx == 0 else match.player_b_id
                    if not player_id:
                        continue

                    # Average center coordinate of player bounding box
                    px_center_x = (box.x_min + box.x_max) / 2.0
                    px_center_y = (box.y_min + box.y_max) / 2.0

                    # Map coordinates using homography
                    court_x, court_y = None, None
                    if homography_matrix is not None:
                        court_x, court_y = pixel_to_court(px_center_x, px_center_y, homography_matrix)

                    pos_row = PlayerPosition(
                        match_id=match.match_id,
                        frame_number=frame_idx,
                        player_id=player_id,
                        court_x=court_x,
                        court_y=court_y,
                        pose_keypoints=box.to_dict() # store box coordinates in JSON field for v1
                    )
                    session.add(pos_row)

            session.commit()
            print(f"[Celery] CV Stage 1 complete for Match: {match_id}")

            # Chain Stage 2 task
            process_match_cv_stage2.delay(match_id)
            return True

        except Exception as exc:
            session.rollback()
            print(f"[Celery Error] Stage 1 failed for Match {match_id}: {exc}")
            # Exponential backoff retry
            raise self.retry(exc=exc)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_match_cv_stage2(self, match_id: str):
    """
    CV Pipeline Stage 2:
    - Pose Estimation (MMPose)
    - Stroke classification (BST/TemPose)
    - Rally segmentation
    - Updates matches.processing_status to 'done' (or 'failed' on error)
    """
    print(f"[Celery] Starting CV Stage 2 (Stroke Classification) for Match: {match_id}")
    
    with Session(engine) as session:
        match_uuid = UUID(match_id) if isinstance(match_id, str) else match_id
        match = session.get(Match, match_uuid)
        if not match:
            print(f"[Celery Error] Match {match_id} not found.")
            return False

        try:
            # We mock the Stage 2 execution for Phase 1
            # In Phase 2 we will replace this with actual pose estimation and BST transformer loading!
            time.sleep(1.0)  # simulate processing
            
            match.processing_status = "done"
            session.add(match)
            session.commit()
            
            print(f"[Celery] CV Stage 2 complete. Status set to 'done' for Match: {match_id}")
            return True
            
        except Exception as exc:
            session.rollback()
            match.processing_status = "failed"
            session.add(match)
            session.commit()
            print(f"[Celery Error] Stage 2 failed for Match {match_id}: {exc}")
            raise self.retry(exc=exc)
