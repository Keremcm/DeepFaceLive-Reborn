"""Photo / video file processing with the FaceSwapEngine."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from .engine import FaceSwapEngine

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEO_EXT = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.flv', '.wmv'}

# progress callback: (done_frames, total_frames or 0, fps) -> return True to cancel
Progress = Callable[[int, int, float], Optional[bool]]


def kind_of(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    return 'image' if ext in IMAGE_EXT else 'video' if ext in VIDEO_EXT else None


def process_image(engine: FaceSwapEngine, src: Path, dst: Path) -> bool:
    """Returns True if at least one face was swapped."""
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)   # applies EXIF orientation
    if img is None:
        raise ValueError(f'cannot read image: {src}')
    engine.reset()
    out = engine.swap_frame(img)
    dst.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 97] if dst.suffix.lower() in ('.jpg', '.jpeg') else []
    if not cv2.imwrite(str(dst), out, params):
        raise IOError(f'cannot write image: {dst}')
    return out is not img


_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _probe(src: Path) -> dict:
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=r_frame_rate,nb_frames,duration', '-show_entries', 'format=duration',
                        '-of', 'json', str(src)], capture_output=True, text=True)
    info = {}
    try:
        d = json.loads(r.stdout)
        s = d['streams'][0]
        n, den = s['r_frame_rate'].split('/')
        info['fps'] = float(n) / float(den) if float(den) else 0.0
        info['frames'] = int(s['nb_frames']) if s.get('nb_frames', 'N/A').isdigit() else 0
        info['duration'] = float(s.get('duration') or d.get('format', {}).get('duration') or 0)
    except Exception:
        pass
    return info


def process_video(engine: FaceSwapEngine, src: Path, dst: Path, crf: int = 18, nvenc: bool = False,
                  limit: int = 0, progress: Optional[Progress] = None) -> int:
    """Swap every frame, keep source audio. Returns number of frames written."""
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found in PATH')
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {src}')
    info = _probe(src)
    fps = info.get('fps') or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = info.get('frames') or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(info.get('duration', 0) * fps)
    if limit:
        total = min(total, limit) if total else limit
    # phone videos store portrait as rotated frames + metadata; opencv may not apply it
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rot = _ROTATE.get(int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360)

    def read():
        ok, f = cap.read()
        return ok, (cv2.rotate(f, rot) if ok and rot is not None else f)

    ok, frame = read()
    if not ok:
        raise ValueError(f'cannot read first frame: {src}')
    H, W = frame.shape[:2]

    dst.parent.mkdir(parents=True, exist_ok=True)
    vcodec = ['-c:v', 'h264_nvenc', '-cq', str(crf), '-preset', 'p5'] if nvenc else \
             ['-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium']
    cmd = ['ffmpeg', '-y', '-v', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', f'{fps}', '-i', '-',
           '-i', str(src), '-map', '0:v:0', '-map', '1:a:0?',
           *vcodec, '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart',
           '-shortest', str(dst)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    engine.reset()
    done, t0, cancelled = 0, time.time(), False
    try:
        while ok:
            ff.stdin.write(engine.swap_frame(frame).tobytes())
            done += 1
            if progress and progress(done, total, done / max(time.time() - t0, 1e-6)):
                cancelled = True
                break
            if limit and done >= limit:
                break
            ok, frame = read()
    finally:
        cap.release()
        try:
            ff.stdin.close()
        except BrokenPipeError:
            pass
        ff.wait()
    if ff.returncode != 0:
        raise RuntimeError(f'ffmpeg failed (exit {ff.returncode})')
    if cancelled:
        dst.unlink(missing_ok=True)
    return done
