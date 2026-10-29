import argparse
import sys
import time
from pathlib import Path

from .engine import FaceSwapEngine, SwapConfig, list_models
from .media import kind_of, process_image, process_video


def _bar(done, total, fps):
    if total:
        pct = done / total
        eta = (total - done) / max(fps, 1e-6)
        sys.stderr.write(f'\r  {done}/{total} [{"#" * int(pct * 30):<30}] {pct * 100:5.1f}%  {fps:5.1f} fps  eta {eta:5.0f}s ')
    else:
        sys.stderr.write(f'\r  {done} frames  {fps:5.1f} fps ')
    sys.stderr.flush()


def _out_path(src: Path, out: Path, multi: bool, model: str) -> Path:
    if out is None:
        return src.with_name(f'{src.stem}_{Path(model).stem}{src.suffix}')
    if multi or out.is_dir():
        return out / src.name
    return out


def cmd_models(args):
    for p in list_models():
        print(f'{p.stem:30s} {p.stat().st_size / 1e6:8.0f} MB   {p}')


def cmd_swap(args):
    src = Path(args.input)
    if not src.exists():
        sys.exit(f'input not found: {src}')
    files = sorted(f for f in src.iterdir() if kind_of(f)) if src.is_dir() else [src]
    if not files:
        sys.exit('no supported photo/video files found')
    out = Path(args.output) if args.output else None
    if src.is_dir() and out is None:
        out = src / 'swapped'

    cfg = SwapConfig(detect_threshold=args.threshold,
                     max_faces=0 if args.all_faces else 1, face_id=args.face_id,
                     morph_factor=args.morph, two_pass=args.two_pass,
                     mask_erode=args.erode, mask_blur=args.blur, opacity=args.opacity,
                     color_transfer=not args.no_color_transfer, smoothing=args.smooth, hq=args.hq,
                     enhance=args.enhance, enhance_strength=args.enhance_strength)
    t0 = time.time()
    engine = FaceSwapEngine(args.model, cfg, cpu=args.cpu)
    print(f'model {engine.model_path.name} ({engine.resolution}px) on {engine.device}  [{time.time() - t0:.1f}s load]')

    for f in files:
        dst = _out_path(f, out, src.is_dir(), args.model)
        print(f'{f.name} -> {dst}')
        try:
            if kind_of(f) == 'image':
                if not process_image(engine, f, dst):
                    print('  warning: no face found, copied unchanged')
            else:
                n = process_video(engine, f, dst, crf=args.crf, nvenc=args.nvenc, limit=args.limit, progress=_bar)
                sys.stderr.write('\n')
                print(f'  {n} frames written')
        except Exception as e:
            sys.stderr.write('\n')
            print(f'  FAILED: {e}')
            if not src.is_dir():
                sys.exit(1)


def cmd_gui(args):
    from .ui.app import run
    run()


def cmd_live(args):
    import cv2
    from .vcam import VirtualCamera
    cfg = SwapConfig(detect_threshold=args.threshold, max_faces=0 if args.all_faces else 1, morph_factor=args.morph,
                     mask_erode=args.erode, mask_blur=args.blur, smoothing=args.smooth)
    engine = FaceSwapEngine(args.model, cfg, cpu=args.cpu)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f'camera {args.camera} cannot be opened')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    vcam, n, t0 = None, 0, time.time()
    print(f'model {engine.model_path.name} on {engine.device}; Ctrl+C to stop')
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                sys.exit('camera read failed')
            img = engine.swap_frame(img)
            if vcam is None:
                vcam = VirtualCamera(args.vcam, img.shape[1], img.shape[0], 30)
                print(f'sending to virtual camera {vcam.device}')
            vcam.send(cv2.flip(img, 1) if args.mirror else img)
            n += 1
            if time.time() - t0 >= 2:
                sys.stderr.write(f'\r  {n / (time.time() - t0):5.1f} fps ')
                n, t0 = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if vcam:
            vcam.close()


def main():
    p = argparse.ArgumentParser(prog='dfl', description='DeepFaceLive offline face swap')
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('models', help='list available .dfm models')
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser('gui', help='open the graphical interface')
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser('live', help='camera -> swap -> virtual camera (no window)')
    sp.add_argument('-m', '--model', required=True)
    sp.add_argument('--camera', type=int, default=0)
    sp.add_argument('--vcam', default='/dev/video10', help='v4l2loopback device')
    sp.add_argument('--mirror', action='store_true', help='mirror the virtual camera output')
    sp.add_argument('--width', type=int, default=640)
    sp.add_argument('--height', type=int, default=480)
    sp.add_argument('--all-faces', action='store_true')
    sp.add_argument('--threshold', type=float, default=0.5)
    sp.add_argument('--morph', type=float, default=0.75)
    sp.add_argument('--erode', type=int, default=5)
    sp.add_argument('--blur', type=int, default=25)
    sp.add_argument('--smooth', type=float, default=0.3)
    sp.add_argument('--cpu', action='store_true')
    sp.set_defaults(func=cmd_live)

    sp = sub.add_parser('swap', help='swap faces in a photo, video or a folder of them')
    sp.add_argument('input', help='photo, video or directory')
    sp.add_argument('-o', '--output', help='output file or directory (default: <input>_<model>.<ext>)')
    sp.add_argument('-m', '--model', required=True, help='model name (see `models`) or path to .dfm')
    sp.add_argument('--all-faces', action='store_true', help='swap every face, not just the largest')
    sp.add_argument('--face-id', type=int, default=0, help='which face by size when not --all-faces (0 = largest)')
    sp.add_argument('--threshold', type=float, default=0.5, help='face detection threshold')
    sp.add_argument('--morph', type=float, default=0.75, help='morph factor (models that support it)')
    sp.add_argument('--two-pass', action='store_true', help='run the model twice (sharper identity, slower)')
    sp.add_argument('--erode', type=int, default=5, help='mask erode')
    sp.add_argument('--blur', type=int, default=25, help='mask blur')
    sp.add_argument('--opacity', type=float, default=1.0)
    sp.add_argument('--no-color-transfer', action='store_true')
    sp.add_argument('--smooth', type=float, default=0.5, help='video: landmark smoothing 0..0.9 (reduces jitter)')
    sp.add_argument('--crf', type=int, default=18, help='video quality (lower = better)')
    sp.add_argument('--nvenc', action='store_true', help='video: encode with NVIDIA hardware encoder')
    sp.add_argument('--limit', type=int, default=0, help='video: only the first N frames (for quick tests)')
    sp.add_argument('--hq', action='store_true', help='high quality: every frame is processed with extra passes (much slower)')
    sp.add_argument('--enhance', action='store_true', help='face enhancer (GFPGAN 1.4) on the swapped face; slower')
    sp.add_argument('--enhance-strength', type=float, default=0.4, help='enhancer blend 0..1')
    sp.add_argument('--cpu', action='store_true', help='force CPU')
    sp.set_defaults(func=cmd_swap)

    args = p.parse_args()
    args.func(args)
