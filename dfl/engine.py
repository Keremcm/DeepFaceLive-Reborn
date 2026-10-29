"""Single-process face swap engine: detect -> landmarks -> align -> DFM swap -> merge.

Same algorithm as the live backends in apps/DeepFaceLive/backend, without the
multiprocessing pipeline, so it can be used for offline photo/video processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numexpr as ne
import numpy as np

from modelhub import onnx as onnx_models
from modelhub.DFLive.DFMModel import DFMModel
from xlib.face import ELandmarks2D, FLandmarks2D, FRect
from xlib.image import ImageProcessor
from xlib.onnxruntime.device import ORTDeviceInfo

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [ROOT / 'models']


@dataclass
class SwapConfig:
    # detection
    detect_threshold : float = 0.5
    max_faces : int = 1             # 0 = every face
    face_id : int = 0               # which face (largest first) when max_faces == 1
    # landmarks / alignment
    marker_coverage : float = 1.4
    face_coverage : float = 2.2
    exclude_moving_parts : bool = True
    # swap
    morph_factor : float = 0.75
    two_pass : bool = False
    # merge
    mask_source : bool = True
    mask_celeb : bool = True
    mask_lmrks : bool = False
    mask_erode : int = 5
    mask_blur : int = 25
    color_transfer : bool = True    # rct
    opacity : float = 1.0
    # high quality (offline photo/video only; several times slower)
    hq : bool = False
    hq_landmark_shift : float = 0.03   # relative crop shift for landmark averaging
    # face enhancer (GFPGAN, offline only)
    enhance : bool = False
    enhance_strength : float = 0.4     # 0..1 blend of enhanced face over the swapped one
    enhance_coverage : float = 1.8     # landmark crop coverage for the 512px enhancer input
    # video
    smoothing : float = 0.0         # 0..1, EMA weight of previous landmarks (video only)
    scene_cut_dist : float = 0.25   # reset smoothing if face jumps further than this (fraction of frame)


def resolve_model(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    for d in MODEL_DIRS:
        for cand in (d / name_or_path, d / f'{name_or_path}.dfm', d / f'{name_or_path.replace(" ", "_")}.dfm'):
            if cand.is_file():
                return cand
    raise FileNotFoundError(f'DFM model not found: {name_or_path} (searched {[str(d) for d in MODEL_DIRS]})')


def list_models() -> List[Path]:
    seen, out = set(), []
    for d in MODEL_DIRS:
        if d.is_dir():
            for p in sorted(d.glob('*.dfm')):
                if p.resolve() not in seen:
                    seen.add(p.resolve())
                    out.append(p)
    return out


def pick_device(cpu: bool = False) -> ORTDeviceInfo:
    devs = onnx_models.YoloV5Face.get_available_devices()
    if not cpu:
        for d in devs:
            if not d.is_cpu():
                return d
    for d in devs:
        if d.is_cpu():
            return d
    raise RuntimeError('no onnxruntime device available')


class FaceSwapEngine:
    def __init__(self, model: str | Path, config: Optional[SwapConfig] = None, cpu: bool = False):
        self.cfg = config or SwapConfig()
        self.device = pick_device(cpu)
        self.model_path = resolve_model(str(model))
        self.detector = onnx_models.YoloV5Face(self.device)
        self.marker = onnx_models.FaceMesh(self.device)
        self.dfm = DFMModel(self.model_path, self.device)
        self.resolution = self.dfm.get_input_res()[0]
        self._prev_lmrks: List[np.ndarray] = []
        self._enhancer = None

    # -- state ---------------------------------------------------------
    def reset(self):
        """Forget temporal state (call between unrelated clips/images)."""
        self._prev_lmrks = []

    # -- steps ---------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[FRect]:
        H, W = frame.shape[:2]
        rects = self.detector.extract(frame, threshold=self.cfg.detect_threshold, augment=self.cfg.hq)[0]
        rects = [FRect.from_ltrb((l / W, t / H, r / W, b / H)) for l, t, r, b in rects]
        rects = [r for r in rects if r.get_area() != 0]
        rects = FRect.sort_by_area_size(rects)
        if self.cfg.max_faces == 1:
            rects = rects[self.cfg.face_id:self.cfg.face_id + 1]
        elif self.cfg.max_faces > 1:
            rects = rects[:self.cfg.max_faces]
        return rects

    def _landmarks_once(self, frame: np.ndarray, rect: FRect, x_off: float = 0.0, y_off: float = 0.0) -> FLandmarks2D:
        face_img, uni_mat = rect.cut(frame, self.cfg.marker_coverage, 192, x_offset=x_off, y_offset=y_off)
        H, W = face_img.shape[:2]
        lmrks = self.marker.extract(face_img)[0][..., 0:2] / (W, H)
        return FLandmarks2D.create(ELandmarks2D.L468, lmrks).transform(uni_mat, invert=True)

    def landmarks(self, frame: np.ndarray, rect: FRect) -> FLandmarks2D:
        if not self.cfg.hq:
            return self._landmarks_once(frame, rect)
        # average landmarks over slightly shifted crops: removes per-frame landmark noise
        d = self.cfg.hq_landmark_shift
        shifts = [(0, 0), (-d, 0), (d, 0), (0, -d), (0, d)]
        pts = [self._landmarks_once(frame, rect, x, y).as_numpy() for x, y in shifts]
        return FLandmarks2D.create(ELandmarks2D.L468, np.mean(pts, 0))

    def _smooth(self, face_idx: int, ulmrks: FLandmarks2D) -> FLandmarks2D:
        a = self.cfg.smoothing
        if a <= 0:
            return ulmrks
        cur = ulmrks.as_numpy()
        while len(self._prev_lmrks) <= face_idx:
            self._prev_lmrks.append(None)
        prev = self._prev_lmrks[face_idx]
        if prev is not None and np.linalg.norm(cur.mean(0) - prev.mean(0)) < self.cfg.scene_cut_dist:
            cur = prev * a + cur * (1.0 - a)
            ulmrks = FLandmarks2D.create(ELandmarks2D.L468, cur)
        self._prev_lmrks[face_idx] = cur
        return ulmrks

    def _swap_one(self, frame: np.ndarray, merged: np.ndarray, ulmrks: FLandmarks2D) -> np.ndarray:
        cfg = self.cfg
        H, W = frame.shape[:2]
        align_img, uni_mat = ulmrks.cut(frame, cfg.face_coverage, self.resolution,
                                        exclude_moving_parts=cfg.exclude_moving_parts, y_offset=-0.08)
        align_ulmrks = ulmrks.transform(uni_mat)
        lmrks_mask = align_ulmrks.get_convexhull_mask(align_img.shape[:2], color=(255,), dtype=np.uint8)

        face, celeb_mask, src_mask = self.dfm.convert(align_img, morph_factor=cfg.morph_factor)
        face, celeb_mask, src_mask = face[0], celeb_mask[0], src_mask[0]
        if cfg.two_pass or cfg.hq:
            face, celeb_mask, _ = self.dfm.convert(face, morph_factor=cfg.morph_factor)
            face, celeb_mask = face[0], celeb_mask[0]

        return self._merge(merged, align_img, src_mask, celeb_mask, lmrks_mask, face, uni_mat, W, H)

    def _merge(self, frame_f, align_img, src_mask, celeb_mask, lmrks_mask, face, uni_mat, W, H):
        cfg = self.cfg
        masks = []
        if cfg.mask_source: masks.append(ImageProcessor(src_mask).to_ufloat32().get_image('HW'))
        if cfg.mask_celeb:  masks.append(ImageProcessor(celeb_mask).to_ufloat32().get_image('HW'))
        if cfg.mask_lmrks:  masks.append(ImageProcessor(lmrks_mask).to_ufloat32().get_image('HW'))
        if masks:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask * m
        else:
            mask = np.ones((self.resolution, self.resolution), np.float32)

        mask = ImageProcessor(mask).erode_blur(cfg.mask_erode, cfg.mask_blur, fade_to_border=True).get_image('HWC')

        fh, fw = align_img.shape[:2]
        to_frame = uni_mat.invert().to_exact_mat(fw, fh, W, H)
        frame_mask = ImageProcessor(mask).warp_affine(to_frame, W, H).clip2((1.0 / 255.0), 0.0, 1.0, 1.0).get_image('HWC')

        face_ip = ImageProcessor(face).to_ufloat32()
        if cfg.color_transfer:
            face_ip = face_ip.rct(like=align_img, mask=mask, like_mask=mask)
        interp = ImageProcessor.Interpolation.LANCZOS4 if cfg.hq else None
        frame_face = face_ip.warp_affine(to_frame, W, H, interpolation=interp).get_image('HWC')

        one, op = np.float32(1.0), np.float32(cfg.opacity)
        if cfg.opacity == 1.0:
            return ne.evaluate('frame_f*(one-frame_mask) + frame_face*frame_mask')
        return ne.evaluate('frame_f*(one-frame_mask) + frame_f*frame_mask*(one-op) + frame_face*frame_mask*op')

    # -- public --------------------------------------------------------
    def _enhance(self, img: np.ndarray, ulmrks: FLandmarks2D) -> np.ndarray:
        """Run GFPGAN on the face of an already merged uint8 frame and blend it back."""
        from .enhancer import SIZE, FaceEnhancer
        if self._enhancer is None:
            self._enhancer = FaceEnhancer(self.device)
        cfg = self.cfg
        H, W = img.shape[:2]
        crop, uni_mat = ulmrks.cut(img, cfg.enhance_coverage, SIZE, exclude_moving_parts=cfg.exclude_moving_parts, y_offset=-0.08)
        enhanced = self._enhancer(crop)
        s = np.float32(cfg.enhance_strength)
        mixed = crop.astype(np.float32) * (1 - s) + enhanced.astype(np.float32) * s

        mask = ulmrks.transform(uni_mat).get_convexhull_mask((SIZE, SIZE), color=(1,), dtype=np.float32)
        mask = ImageProcessor(mask).erode_blur(15, 40, fade_to_border=True).get_image('HWC')

        to_frame = uni_mat.invert().to_exact_mat(SIZE, SIZE, W, H)
        interp = ImageProcessor.Interpolation.LANCZOS4
        frame_face = ImageProcessor(mixed / 255.0).warp_affine(to_frame, W, H, interpolation=interp).get_image('HWC')
        frame_mask = ImageProcessor(mask).warp_affine(to_frame, W, H).clip2((1.0 / 255.0), 0.0, 1.0, 1.0).get_image('HWC')
        out = img.astype(np.float32) / 255.0 * (1 - frame_mask) + frame_face * frame_mask
        return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def swap_frame(self, frame: np.ndarray) -> np.ndarray:
        """frame: HxWx3 uint8 BGR. Returns the swapped frame (uint8 BGR); unchanged if no face."""
        rects = self.detect(frame)
        if not rects:
            return frame
        merged = ImageProcessor(frame).to_ufloat32().get_image('HWC')
        all_lmrks = []
        for i, rect in enumerate(rects):
            ulmrks = self._smooth(i, self.landmarks(frame, rect))
            all_lmrks.append(ulmrks)
            merged = self._swap_one(frame, merged, ulmrks)
        out = (np.clip(merged, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if self.cfg.enhance:
            for ulmrks in all_lmrks:
                out = self._enhance(out, ulmrks)
        return out
