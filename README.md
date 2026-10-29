# DeepFaceLive Reborn

A lean face swap tool for Linux: **photos**, **videos** and **live camera** (with virtual camera output).
It uses the `.dfm` models and the detection/alignment core of [DeepFaceLive](https://github.com/iperov/DeepFaceLive),
but replaces the old multi-process app with a single-process engine, a command line tool and a PyQt6 interface.

> Note: the GUI labels are currently in Turkish; the English names used below are translations.

## Features
- Photo, video and folder (batch) processing. Video audio is kept, and portrait (phone) videos keep their orientation.
- **High quality mode** (`--hq`): augmented face detection, averaged landmarks, two passes, Lanczos blending.
- **Face enhancer** (`--enhance`, GFPGAN 1.4) with adjustable strength.
- **Live mode** with **virtual camera** output (v4l2loopback): pick it as a camera in OBS, Zoom or Discord.
- NVIDIA GPU (CUDA 12, no system CUDA install needed) or CPU (slow).

## Installation
Requirements: Python 3.10–3.12, `ffmpeg`, and an NVIDIA driver for GPU use.

```bash
git clone https://github.com/Keremcm/DeepFaceLive-Reborn.git && cd DeepFaceLive-Reborn
./install.sh          # NVIDIA GPU
./install.sh --cpu    # CPU only
```

Put your `.dfm` models into the `models/` folder (e.g. `models/Joker.dfm`). Models are not included in the repository;
you can use ones from the [DeepFaceLive releases](https://github.com/iperov/DeepFaceLive/releases) or your own trained models.

For the face enhancer (optional, 340 MB):
```bash
mkdir -p models/enhancers
curl -L -o models/enhancers/gfpgan_1.4.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx
```

## Usage
```bash
./dfl.sh gui                                   # graphical interface
./dfl.sh models                                # list models
./dfl.sh swap photo.jpg -m Joker               # photo
./dfl.sh swap video.mp4 -m Joker -o out.mp4    # video
./dfl.sh swap folder/ -m Joker --all-faces     # folder, every face
./dfl.sh swap video.mp4 -m Joker --hq --enhance --enhance-strength 0.4
./dfl.sh live -m Joker --vcam /dev/video10     # live, without a window
```
All options: `./dfl.sh swap -h`. For a quick trial use `--limit 90` (first 90 frames only).

### Virtual camera (OBS / Zoom / Discord)
Create the device once (your kernel needs the `v4l2loopback` module):
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFaceLive" exclusive_caps=1
```
To make it persistent:
```bash
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 video_nr=10 card_label="DeepFaceLive" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
```
In the GUI, open the Live tab, tick "Send to virtual camera" (labelled *Sanal kameraya gönder*) and press Start (*Başlat*). **Then** add a
*Video Capture Device (V4L2)* source in OBS and select **DeepFaceLive**. If the picture is black, set Video Format to YUYV.
The "Mirror virtual camera" option (*Sanal kamerada aynala*) sends the output mirrored, like a mirror.

## Troubleshooting
- **`CUBLAS_STATUS_ALLOC_FAILED` / out-of-memory errors:** GPU memory is full. Close other programs that hold VRAM (e.g. `ollama`).
- **Face does not fit / looks broken:** The face should be close to the camera and frontal; try `--hq` and `--threshold`.
- **Enhancer adds artificial detail:** Lower its strength (0.2–0.4).
- **Camera does not open:** Check `ls /dev/video*`; your user must be in the `video` group.

## Layout
```
dfl/        engine (engine.py), media processing, CLI, PyQt6 interface, virtual camera, enhancer
modelhub/   DFM model loader, YOLOv5Face (detection) and FaceMesh (landmarks) ONNX models
xlib/       helper library taken from DeepFaceLive (face geometry, image processing, onnxruntime)
models/     .dfm models and models/enhancers/ (GFPGAN)
```

## License and credits
GPL-3.0 (see `LICENSE`). This project is derived from [iperov/DeepFaceLive](https://github.com/iperov/DeepFaceLive) (GPL-3.0);
`xlib` and `modelhub` come from there. GFPGAN and the other models have their own licenses; check them before use.
Use it only with faces you have permission to use, and in compliance with the law.
