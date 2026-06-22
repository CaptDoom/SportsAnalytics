import pytest
from unittest.mock import patch
from sqlmodel import SQLModel, create_engine, Session
from shared.models import Player, Match, Set, ShuttleTrajectory, PlayerPosition
from cv_worker.tasks import process_match_cv_stage1, process_match_cv_stage2

# Use in-memory database for testing Celery tasks synchronously
TEST_DB_URL = "sqlite:///:memory:"

@pytest.fixture(name="db_session")
def db_session_fixture():
    # Setup test engine
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    
    # Mock the database engine inside cv_worker.tasks
    with patch("cv_worker.tasks.engine", engine):
        with Session(engine) as session:
            yield session
            
    SQLModel.metadata.drop_all(engine)

@patch("cv_worker.tasks.process_match_cv_stage2.delay")
def test_process_match_cv_stage1_sync(mock_stage2_delay, db_session: Session):
    # 1. Setup seed data
    player_a = Player(name="Viktor Axelsen", country="Denmark")
    player_b = Player(name="Lee Zii Jia", country="Malaysia")
    db_session.add(player_a)
    db_session.add(player_b)
    db_session.commit()
    db_session.refresh(player_a)
    db_session.refresh(player_b)

    match = Match(
        player_a_id=player_a.player_id,
        player_b_id=player_b.player_id,
        tournament="Thomas Cup",
        source_type="broadcast",
        processing_status="pending",
        court_calibration={
            "homography_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        }
    )
    db_session.add(match)
    db_session.commit()
    db_session.refresh(match)

    # 2. Run Stage 1 task synchronously
    # Disable retry exception raise for testing
    with patch("celery.app.task.Task.retry", side_effect=Exception("celery_retry")):
        result = process_match_cv_stage1(str(match.match_id))
        
    assert result is True

    # 3. Verify status changes and DB inserts
    db_session.refresh(match)
    assert match.processing_status == "processing_cv"

    # Verify Shuttle Trajectory entries
    trajectories = db_session.query(ShuttleTrajectory).filter(ShuttleTrajectory.match_id == match.match_id).all()
    assert len(trajectories) == 150  # Mock returns 150 frames
    assert trajectories[0].pixel_x > 0.0
    assert pytest.approx(float(trajectories[0].court_x), abs=1e-1) == float(trajectories[0].pixel_x)  # Homography was identity


    # Verify Player Positions entries
    positions = db_session.query(PlayerPosition).filter(PlayerPosition.match_id == match.match_id).all()
    # 150 frames * 2 players = 300 entries
    assert len(positions) == 300
    
    # Assert Stage 2 task was queued via .delay
    mock_stage2_delay.assert_called_once_with(str(match.match_id))
