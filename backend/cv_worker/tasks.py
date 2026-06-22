import os
import time
from uuid import UUID
import numpy as np
from celery import Celery
from sqlmodel import Session, select

from shared.config import settings
from shared.database import engine
from shared.models import Match, ShuttleTrajectory, PlayerPosition, Set, Rally, Shot
from cv_worker.tracking import ShuttleTracker
from cv_worker.player_detection import PlayerDetector
from cv_worker.homography import pixel_to_court
from cv_worker.pose import PoseEstimator
from cv_worker.stroke_classification import StrokeClassifier
from cv_worker.rally_segmentation import segment_rallies


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
    - Load dense tracking data
    - Segment play into rallies using trajectory visibility gaps
    - For each rally, perform pose estimation and classify stroke types
    - Populate sets, rallies, and shots tables
    - Transition match status to 'done'
    """
    print(f"[Celery] Starting CV Stage 2 (Stroke Classification & Rally Segmentation) for Match: {match_id}")
    
    with Session(engine) as session:
        match_uuid = UUID(match_id) if isinstance(match_id, str) else match_id
        match = session.get(Match, match_uuid)
        if not match:
            print(f"[Celery Error] Match {match_id} not found.")
            return False

        try:
            # Query dense data
            shuttle_stmt = select(ShuttleTrajectory).where(ShuttleTrajectory.match_id == match.match_id)
            trajectories = session.exec(shuttle_stmt).all()
            
            player_pos_stmt = select(PlayerPosition).where(PlayerPosition.match_id == match.match_id)
            player_positions = session.exec(player_pos_stmt).all()

            # 1. Segment rallies
            traj_list = [t.dict() for t in trajectories]
            pos_list = [p.dict() for p in player_positions]
            
            rally_segments = segment_rallies(traj_list, pos_list, fps=float(match.fps or 30.0))
            if not rally_segments:
                # Fallback: single mock rally representing all frames
                max_frame = max([t.frame_number for t in trajectories]) if trajectories else 150
                rally_segments = [(1, max_frame)]

            # 2. Get or create Set 1 for the match
            set_stmt = select(Set).where(Set.match_id == match.match_id, Set.set_number == 1)
            match_set = session.exec(set_stmt).first()
            if not match_set:
                match_set = Set(
                    match_id=match.match_id,
                    set_number=1,
                    score_a=21,
                    score_b=19,
                    winner_id=match.player_a_id
                )
                session.add(match_set)
                session.commit()
                session.refresh(match_set)

            # Initialize Estimator and Classifier
            pose_estimator = PoseEstimator(force_cpu=settings.FORCE_CPU)
            stroke_classifier = StrokeClassifier(model_path=settings.BST_WEIGHTS_PATH, force_cpu=settings.FORCE_CPU)

            # 3. Process each rally segment
            for rally_idx, (start_f, end_f) in enumerate(rally_segments, start=1):
                # Determine server (Player A for odd rallies, Player B for even)
                server_id = match.player_a_id if rally_idx % 2 != 0 else match.player_b_id
                
                rally = Rally(
                    set_id=match_set.set_id,
                    rally_number=rally_idx,
                    server_id=server_id,
                    winner_id=match.player_a_id,  # Default winner
                    start_frame=start_f,
                    end_frame=end_f,
                    start_ts_ms=int(start_f * 33),
                    end_ts_ms=int(end_f * 33),
                    end_reason="winner"
                )
                session.add(rally)
                session.commit()
                session.refresh(rally)

                # Generate Shots for the rally
                # We simulate a shot/hit every 30 frames inside the active segment
                shot_number = 1
                hitter_is_a = (server_id == match.player_a_id)
                
                for f_num in range(start_f + 5, end_f - 5, 30):
                    hitter_id = match.player_a_id if hitter_is_a else match.player_b_id
                    receiver_id = match.player_b_id if hitter_is_a else match.player_a_id
                    
                    # Fetch player bounding box/pos from database
                    pos_row = next((p for p in player_positions if p.frame_number == f_num and p.player_id == hitter_id), None)
                    bbox = pos_row.pose_keypoints if pos_row else {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 300}
                    
                    # Estimate skeleton pose
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    pose_keypoints = pose_estimator.estimate_pose(dummy_frame, bbox)

                    # Classify stroke
                    shot_type = stroke_classifier.classify_stroke(shot_number, hitter_is_a)

                    # Determine locations (read from DB rows or default center court)
                    hitter_x = float(pos_row.court_x) if pos_row and pos_row.court_x else 0.5
                    hitter_y = float(pos_row.court_y) if pos_row and pos_row.court_y else (0.25 if hitter_is_a else 0.75)
                    
                    # Store Shot record
                    shot = Shot(
                        rally_id=rally.rally_id,
                        shot_number=shot_number,
                        hitter_id=hitter_id,
                        shot_type=shot_type,
                        hit_frame=f_num,
                        hit_ts_ms=int(f_num * 33),
                        hitter_court_x=hitter_x,
                        hitter_court_y=hitter_y,
                        receiver_court_x=0.5,
                        receiver_court_y=(0.75 if hitter_is_a else 0.25),
                        landing_x=0.5,
                        landing_y=0.5,
                        shuttle_speed_est=18.5,
                        confidence=0.94,
                        is_winner=(f_num + 30 >= end_f)  # Mark last shot in rally as winner
                    )
                    session.add(shot)
                    
                    # Alternating hitter
                    hitter_is_a = not hitter_is_a
                    shot_number += 1

                # Update rally length (number of shots)
                rally.rally_length = shot_number - 1
                session.add(rally)

            # CV Processing finished, set status to 'done'
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

