"""Virtual camera output through v4l2loopback (Linux) using pyvirtualcam."""
from __future__ import annotations

from pathlib import Path
from typing import List

SETUP_HINT = ('sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFaceLive" exclusive_caps=1')


def _label(dev: Path) -> str:
    try:
        return (Path('/sys/class/video4linux') / dev.name / 'name').read_text().strip()
    except OSError:
        return ''


def is_loopback(dev: Path) -> bool:
    """v4l2loopback devices are not capture hardware: hide them from the physical camera list."""
    return Path('/sys/class/video4linux', dev.name, 'device').exists() is False and dev.exists() and bool(_label(dev))


def loopback_devices() -> List[str]:
    """Devices created by v4l2loopback (no parent hardware device in sysfs)."""
    out = []
    for p in sorted(Path('/dev').glob('video*')):
        if is_loopback(p):
            out.append(str(p))
    return out


class VirtualCamera:
    def __init__(self, device: str, width: int, height: int, fps: int = 30):
        try:
            import pyvirtualcam
        except ImportError:
            raise RuntimeError('pyvirtualcam kurulu değil:  pip install pyvirtualcam')
        if not Path(device).exists():
            raise RuntimeError(f'{device} yok. Önce sanal kamera cihazını oluştur:\n{SETUP_HINT}')
        try:
            self._cam = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                            fmt=pyvirtualcam.PixelFormat.BGR, device=device)
        except Exception as e:
            raise RuntimeError(f'Sanal kamera açılamadı ({device}): {e}')
        self.device = self._cam.device

    def send(self, bgr) -> None:
        self._cam.send(bgr)

    def close(self) -> None:
        self._cam.close()
