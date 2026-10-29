"""PyQt6 front-end for the offline/live FaceSwapEngine."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                             QSpinBox, QTabWidget, QVBoxLayout, QWidget)

import dataclasses

from ..engine import FaceSwapEngine, SwapConfig, list_models
from ..vcam import VirtualCamera, loopback_devices
from ..media import IMAGE_EXT, VIDEO_EXT, kind_of, process_image, process_video


class EngineManager:
    """Keeps one engine alive (VRAM is limited) and serialises access to it."""
    def __init__(self):
        self.lock = threading.Lock()
        self._engine: FaceSwapEngine | None = None
        self._key = None

    def get(self, model: str, cfg: SwapConfig, cpu: bool) -> FaceSwapEngine:
        key = (model, cpu)
        if self._engine is None or self._key != key:
            self._engine = None            # free old sessions before loading new ones
            self._engine = FaceSwapEngine(model, cfg, cpu=cpu)
            self._key = key
        self._engine.cfg = cfg
        return self._engine


def to_qimage(bgr) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class FileWorker(QThread):
    progress = pyqtSignal(int, int, float)   # done, total, fps
    status = pyqtSignal(str)
    finished_one = pyqtSignal(str)           # output path
    failed = pyqtSignal(str)

    def __init__(self, mgr, model, cpu, cfg_fn, jobs, nvenc, crf):
        super().__init__()
        self.mgr, self.model, self.cpu, self.cfg_fn, self.jobs, self.nvenc, self.crf = mgr, model, cpu, cfg_fn, jobs, nvenc, crf
        self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        with self.mgr.lock:
            try:
                self.status.emit('Model yükleniyor...')
                engine = self.mgr.get(self.model, self.cfg_fn(), self.cpu)
                for src, dst in self.jobs:
                    if self._cancel:
                        break
                    self.status.emit(f'İşleniyor: {src.name}')
                    if kind_of(src) == 'image':
                        process_image(engine, src, dst)
                    else:
                        n = process_video(engine, src, dst, crf=self.crf, nvenc=self.nvenc,
                                          progress=lambda d, t, f: (self.progress.emit(d, t, f), self._cancel)[1])
                        if self._cancel:
                            break
                    self.finished_one.emit(str(dst))
            except Exception as e:
                self.failed.emit(str(e))


class LiveWorker(QThread):
    frame = pyqtSignal(QImage)
    fps = pyqtSignal(float)
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, mgr, model, cpu, cfg_fn, cam, size, flip, swap_on_fn, vcam_device=None, vcam_flip_fn=lambda: False):
        super().__init__()
        self.mgr, self.model, self.cpu, self.cfg_fn = mgr, model, cpu, cfg_fn
        self.cam, self.size, self.flip, self.swap_on_fn, self.vcam_device, self.vcam_flip_fn = cam, size, flip, swap_on_fn, vcam_device, vcam_flip_fn
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        cap, vcam = None, None
        with self.mgr.lock:
            try:
                self.status.emit('Model yükleniyor...')
                engine = self.mgr.get(self.model, self.cfg_fn(), self.cpu)
                cap = cv2.VideoCapture(self.cam)
                if not cap.isOpened():
                    raise RuntimeError(f'Kamera {self.cam} açılamadı')
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                self.status.emit('Canlı')
                engine.reset()
                n, t0 = 0, time.time()
                while not self._stop:
                    ok, img = cap.read()
                    if not ok:
                        raise RuntimeError('Kameradan görüntü alınamadı')
                    engine.cfg = self.cfg_fn()
                    if self.swap_on_fn():
                        img = engine.swap_frame(img)
                    if self.vcam_device and vcam is None:
                        h, w = img.shape[:2]
                        vcam = VirtualCamera(self.vcam_device, w, h, 30)
                        self.status.emit(f'Canlı → sanal kamera {vcam.device}')
                    if vcam is not None:
                        vcam.send(cv2.flip(img, 1) if self.vcam_flip_fn() else img)
                    self.frame.emit(to_qimage(cv2.flip(img, 1) if self.flip else img))
                    n += 1
                    if time.time() - t0 >= 1.0:
                        self.fps.emit(n / (time.time() - t0))
                        n, t0 = 0, time.time()
            except Exception as e:
                self.failed.emit(str(e))
            finally:
                if cap is not None:
                    cap.release()
                if vcam is not None:
                    vcam.close()


class DropList(QListWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        self.files_dropped.emit([Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DeepFaceLive Reborn')
        self.resize(1100, 700)
        self.mgr = EngineManager()
        self.file_worker: FileWorker | None = None
        self.live_worker: LiveWorker | None = None
        self.files: list[Path] = []
        self.out_dir: Path | None = None
        self.last_output: str | None = None

        root = QWidget()
        lay = QHBoxLayout(root)
        lay.addWidget(self._build_settings(), 0)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_file_tab(), 'Fotoğraf / Video')
        self.tabs.addTab(self._build_live_tab(), 'Canlı')
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage('Hazır')

    # ---- settings panel ------------------------------------------------
    def _build_settings(self) -> QWidget:
        box = QGroupBox('Ayarlar')
        box.setFixedWidth(270)
        f = QFormLayout(box)

        self.model_cb = QComboBox()
        self._refresh_models()
        f.addRow('Model', self.model_cb)

        self.all_faces = QCheckBox('Tüm yüzleri değiştir')
        f.addRow(self.all_faces)

        def dspin(lo, hi, val, step, dec=2):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(dec); s.setValue(val); return s

        def ispin(lo, hi, val):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); return s

        self.threshold = dspin(0.1, 0.95, 0.5, 0.05)
        self.morph = dspin(0, 1, 0.75, 0.05)
        self.opacity = dspin(0, 1, 1.0, 0.05)
        self.erode = ispin(-30, 50, 5)
        self.blur = ispin(0, 100, 25)
        self.smooth = dspin(0, 0.9, 0.5, 0.05)
        self.two_pass = QCheckBox('İki geçiş (yavaş, keskin)')
        self.color_tr = QCheckBox('Renk aktarımı'); self.color_tr.setChecked(True)
        self.cpu = QCheckBox('CPU kullan (GPU yerine)')

        f.addRow('Yüz algılama eşiği', self.threshold)
        f.addRow('Morph', self.morph)
        f.addRow('Opaklık', self.opacity)
        f.addRow('Maske daralt', self.erode)
        f.addRow('Maske yumuşat', self.blur)
        f.addRow('Titreme azaltma', self.smooth)
        f.addRow(self.two_pass)
        f.addRow(self.color_tr)
        f.addRow(self.cpu)

        self.nvenc = QCheckBox('GPU ile kodla (NVENC)')
        self.crf = ispin(10, 35, 18)
        f.addRow(QLabel('<b>Video çıktısı</b>'))
        f.addRow(self.nvenc)
        f.addRow('Kalite (düşük = iyi)', self.crf)
        return box

    def _refresh_models(self):
        self.model_cb.clear()
        for p in list_models():
            self.model_cb.addItem(p.stem, str(p))
        if self.model_cb.count() == 0:
            self.model_cb.addItem('(model yok: models/ klasörüne .dfm koy)', '')

    def make_cfg(self) -> SwapConfig:
        return SwapConfig(detect_threshold=self.threshold.value(), max_faces=0 if self.all_faces.isChecked() else 1,
                          morph_factor=self.morph.value(), two_pass=self.two_pass.isChecked(),
                          mask_erode=self.erode.value(), mask_blur=self.blur.value(), opacity=self.opacity.value(),
                          color_transfer=self.color_tr.isChecked(), smoothing=self.smooth.value())

    def _model(self) -> str | None:
        m = self.model_cb.currentData()
        if not m:
            QMessageBox.warning(self, 'Model yok', 'Önce bir .dfm modeli seçmelisin.')
            return None
        return m

    # ---- file tab --------------------------------------------------------
    def _build_file_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel('Dosyaları aşağıya sürükle veya "Dosya ekle" ile seç (fotoğraf / video).'))
        self.file_list = DropList()
        self.file_list.files_dropped.connect(self.add_files)
        v.addWidget(self.file_list)

        row = QHBoxLayout()
        b_add = QPushButton('Dosya ekle...'); b_add.clicked.connect(self.pick_files)
        b_clear = QPushButton('Temizle'); b_clear.clicked.connect(self.clear_files)
        b_out = QPushButton('Çıktı klasörü...'); b_out.clicked.connect(self.pick_out_dir)
        self.out_label = QLabel('Çıktı: girdinin yanına')
        row.addWidget(b_add); row.addWidget(b_clear); row.addWidget(b_out); row.addWidget(self.out_label, 1)
        v.addLayout(row)

        self.preview = QLabel('Önizleme')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet('background:#111; color:#888;')
        v.addWidget(self.preview, 1)

        self.hq_cb = QCheckBox('Yüksek kalite (her kare çift geçişle işlenir, çok daha yavaş)')
        self.hq_cb.setToolTip('Yalnızca fotoğraf/video için. Yüz algılama artırılır, landmark\'lar ortalanır,\n'
                              'model iki kez çalışır ve birleştirmede Lanczos kullanılır.')
        v.addWidget(self.hq_cb)

        erow = QHBoxLayout()
        self.enh_cb = QCheckBox('Yüz iyileştirici (GFPGAN)')
        self.enh_cb.setToolTip('Swap sonrası yüzü keskinleştirir. Yüksek güçte göz/burun çevresinde yapay detay çıkarabilir.')
        self.enh_strength = QDoubleSpinBox(); self.enh_strength.setRange(0.0, 1.0); self.enh_strength.setSingleStep(0.05)
        self.enh_strength.setDecimals(2); self.enh_strength.setValue(0.4)
        erow.addWidget(self.enh_cb); erow.addWidget(QLabel('Güç')); erow.addWidget(self.enh_strength); erow.addStretch(1)
        v.addLayout(erow)

        self.progress = QProgressBar(); self.progress.setValue(0)
        self.prog_text = QLabel('')
        v.addWidget(self.progress); v.addWidget(self.prog_text)

        row = QHBoxLayout()
        self.b_start = QPushButton('Başlat'); self.b_start.clicked.connect(self.start_files)
        self.b_cancel = QPushButton('İptal'); self.b_cancel.setEnabled(False); self.b_cancel.clicked.connect(self._cancel_files)
        self.b_open = QPushButton('Çıktıyı aç'); self.b_open.setEnabled(False); self.b_open.clicked.connect(self.open_output)
        row.addWidget(self.b_start); row.addWidget(self.b_cancel); row.addStretch(1); row.addWidget(self.b_open)
        v.addLayout(row)
        return w

    def pick_files(self):
        exts = ' '.join(f'*{e}' for e in sorted(IMAGE_EXT | VIDEO_EXT))
        names, _ = QFileDialog.getOpenFileNames(self, 'Dosya seç', str(Path.home()), f'Medya ({exts});;Hepsi (*)')
        self.add_files([Path(n) for n in names])

    def add_files(self, paths):
        for p in paths:
            cands = sorted(f for f in p.iterdir() if kind_of(f)) if p.is_dir() else [p]
            for f in cands:
                if kind_of(f) and f not in self.files:
                    self.files.append(f)
                    self.file_list.addItem(f.name)

    def clear_files(self):
        self.files.clear(); self.file_list.clear()

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Çıktı klasörü', str(Path.home()))
        if d:
            self.out_dir = Path(d)
            self.out_label.setText(f'Çıktı: {d}')

    def _out_path(self, src: Path, model: str) -> Path:
        name = f'{src.stem}_{Path(model).stem}{src.suffix}'
        return (self.out_dir or src.parent) / name

    def start_files(self):
        model = self._model()
        if not model:
            return
        if not self.files:
            QMessageBox.information(self, 'Dosya yok', 'Önce dosya ekle.')
            return
        if self.live_worker is not None:
            QMessageBox.information(self, 'Canlı açık', 'Önce canlı sekmesini durdur (GPU belleği).')
            return
        jobs = [(f, self._out_path(f, model)) for f in self.files]
        hq = self.hq_cb.isChecked()
        enh, enh_s = self.enh_cb.isChecked(), self.enh_strength.value()
        cfg_fn = lambda: dataclasses.replace(self.make_cfg(), hq=hq, enhance=enh, enhance_strength=enh_s)
        self.file_worker = w = FileWorker(self.mgr, model, self.cpu.isChecked(), cfg_fn, jobs,
                                          self.nvenc.isChecked(), self.crf.value())
        w.progress.connect(self._on_progress)
        w.status.connect(self.statusBar().showMessage)
        w.finished_one.connect(self._on_file_done)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._files_finished)
        self.b_start.setEnabled(False); self.b_cancel.setEnabled(True)
        self.progress.setValue(0)
        w.start()

    def _cancel_files(self):
        if self.file_worker:
            self.file_worker.cancel()
            self.statusBar().showMessage('İptal ediliyor...')

    def _on_progress(self, done, total, fps):
        self.progress.setMaximum(max(total, 1)); self.progress.setValue(min(done, max(total, 1)))
        self.prog_text.setText(f'{done}/{total or "?"} kare   {fps:.1f} fps')

    def _on_file_done(self, path):
        self.last_output = path
        self.b_open.setEnabled(True)
        if kind_of(Path(path)) == 'image':
            pm = QPixmap(path)
            self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
        else:
            cap = cv2.VideoCapture(path)
            ok, img = cap.read(); cap.release()
            if ok:
                self.preview.setPixmap(QPixmap.fromImage(to_qimage(img)).scaled(
                    self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_failed(self, msg):
        QMessageBox.critical(self, 'Hata', msg)

    def _files_finished(self):
        self.b_start.setEnabled(True); self.b_cancel.setEnabled(False)
        self.statusBar().showMessage('Bitti')
        self.file_worker = None

    def open_output(self):
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output))

    # ---- live tab ----------------------------------------------------------
    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.cam_cb = QComboBox()
        for i in range(8):
            if Path(f'/dev/video{i}').exists() and f'/dev/video{i}' not in loopback_devices():
                self.cam_cb.addItem(f'Kamera {i}', i)
        if self.cam_cb.count() == 0:
            self.cam_cb.addItem('Kamera 0', 0)
        self.res_cb = QComboBox()
        for s in [(640, 480), (1280, 720)]:
            self.res_cb.addItem(f'{s[0]}x{s[1]}', s)
        self.flip = QCheckBox('Aynala'); self.flip.setChecked(True)
        self.swap_on = QCheckBox('Swap açık'); self.swap_on.setChecked(True)
        self.b_live = QPushButton('Başlat'); self.b_live.clicked.connect(self.toggle_live)
        for x in (self.cam_cb, self.res_cb, self.flip, self.swap_on):
            row.addWidget(x)
        row.addStretch(1); row.addWidget(self.b_live)
        v.addLayout(row)

        vrow = QHBoxLayout()
        self.vcam_cb = QCheckBox('Sanal kameraya gönder')
        self.vcam_cb.setToolTip('OBS / Zoom / Discord gibi uygulamalarda bu cihazı kamera olarak seç.\n'
                                'İlk kurulum: sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFaceLive" exclusive_caps=1')
        self.vcam_dev = QComboBox(); self.vcam_dev.setEditable(True)
        for d in loopback_devices() or ['/dev/video10']:
            self.vcam_dev.addItem(d)
        self.vcam_flip = QCheckBox('Sanal kamerada aynala')
        self.vcam_flip.setToolTip('Kapalıyken diğer taraf gerçek yönü görür (yazılar doğru okunur).\n'
                                  'Açıkken OBS/Zoom\'da aynalı (ayna gibi) görünürsün.')
        vrow.addWidget(self.vcam_cb); vrow.addWidget(self.vcam_dev); vrow.addWidget(self.vcam_flip); vrow.addStretch(1)
        v.addLayout(vrow)

        self.live_view = QLabel('Kamera kapalı')
        self.live_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_view.setMinimumSize(480, 360)
        self.live_view.setStyleSheet('background:#111; color:#888;')
        v.addWidget(self.live_view, 1)
        self.fps_label = QLabel('')
        v.addWidget(self.fps_label)
        return w

    def toggle_live(self):
        if self.live_worker is not None:
            self.live_worker.stop()
            self.b_live.setEnabled(False)
            return
        model = self._model()
        if not model:
            return
        if self.file_worker is not None:
            QMessageBox.information(self, 'İşlem sürüyor', 'Dosya işlemi bitmeden canlı başlatılamaz.')
            return
        self.live_worker = w = LiveWorker(self.mgr, model, self.cpu.isChecked(), self.make_cfg, self.cam_cb.currentData(),
                                          self.res_cb.currentData(), self.flip.isChecked(), self.swap_on.isChecked,
                                          self.vcam_dev.currentText().strip() if self.vcam_cb.isChecked() else None,
                                          self.vcam_flip.isChecked)
        w.frame.connect(self._on_live_frame)
        w.fps.connect(lambda f: self.fps_label.setText(f'{f:.1f} fps'))
        w.status.connect(self.statusBar().showMessage)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._live_finished)
        self.b_live.setText('Durdur')
        w.start()

    def _on_live_frame(self, img: QImage):
        self.live_view.setPixmap(QPixmap.fromImage(img).scaled(self.live_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.FastTransformation))

    def _live_finished(self):
        self.live_worker = None
        self.b_live.setText('Başlat'); self.b_live.setEnabled(True)
        self.live_view.setText('Kamera kapalı'); self.fps_label.setText('')
        self.statusBar().showMessage('Hazır')

    def closeEvent(self, e):
        for w in (self.file_worker, self.live_worker):
            if w is not None:
                (w.cancel if isinstance(w, FileWorker) else w.stop)()
                w.wait(5000)
        super().closeEvent(e)


def run():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
