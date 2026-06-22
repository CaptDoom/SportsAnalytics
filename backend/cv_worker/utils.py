import gc
import torch

def get_vram_usage_mb() -> float:
    """Returns current CUDA VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def get_peak_vram_usage_mb() -> float:
    """Returns peak CUDA VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def clear_gpu_cache() -> None:
    """Clears PyTorch CUDA memory cache and triggers garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def verify_vram_limit(limit_mb: float = 3500.0) -> bool:
    """Checks if peak VRAM usage exceeds the limit and prints alert if it does."""
    peak = get_peak_vram_usage_mb()
    if peak > limit_mb:
        print(f"[VRAM ALERT] Peak VRAM usage of {peak:.2f} MB exceeds threshold of {limit_mb:.2f} MB!")
        return False
    return True
