"""DeepFaceLive Reborn: offline and live face swap with DFM models."""
try:   # make pip-installed CUDA/cuDNN libraries (nvidia-*-cu12) visible to onnxruntime without LD_LIBRARY_PATH
    import onnxruntime as _ort
    _ort.preload_dlls()
except Exception:
    pass

from .engine import FaceSwapEngine, SwapConfig, list_models, resolve_model
