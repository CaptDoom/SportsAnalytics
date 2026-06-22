import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/badminton")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    S3_BUCKET: str = os.getenv("S3_BUCKET", "badminton-videos")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    
    JWT_SECRET: str = os.getenv("JWT_SECRET", "supersecretjwtkeysupersecretjwtkey")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "mock-key")
    
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
    VRAM_LIMIT_MB: int = int(os.getenv("VRAM_LIMIT_MB", "3500"))
    
    TRACKNET_WEIGHTS_PATH: str = os.getenv("TRACKNET_WEIGHTS_PATH", "backend/cv_worker/vendor/tracknetv3/weights/model.onnx")
    YOLO_WEIGHTS_PATH: str = os.getenv("YOLO_WEIGHTS_PATH", "yolov8n.onnx")
    BST_WEIGHTS_PATH: str = os.getenv("BST_WEIGHTS_PATH", "backend/cv_worker/vendor/bst/weights/bst.onnx")


    class Config:
        env_file = ".env"

settings = Settings()
