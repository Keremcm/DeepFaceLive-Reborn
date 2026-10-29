"""GFPGAN 1.4 face enhancer (ONNX, 512x512 FFHQ-style crops)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from xlib.onnxruntime import InferenceSession_with_device
from xlib.onnxruntime.device import ORTDeviceInfo

ENHANCER_PATH = Path(__file__).resolve().parent.parent / 'models' / 'enhancers' / 'gfpgan_1.4.onnx'
SIZE = 512


class FaceEnhancer:
    def __init__(self, device: ORTDeviceInfo, path: Path = ENHANCER_PATH):
        if not path.is_file():
            raise FileNotFoundError(f'face enhancer model missing: {path}')
        self._sess = InferenceSession_with_device(str(path), device)
        self._in = self._sess.get_inputs()[0].name

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """bgr: 512x512x3 uint8 -> enhanced 512x512x3 uint8"""
        x = bgr[..., ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        y = self._sess.run(None, {self._in: np.ascontiguousarray(x)})[0][0]
        y = (np.clip(y, -1.0, 1.0) + 1.0) * 0.5
        return (y.transpose(1, 2, 0)[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
