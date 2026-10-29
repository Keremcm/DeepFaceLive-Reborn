# DeepFaceLive Reborn

Linux için sade bir yüz değiştirme aracı: **fotoğraf**, **video** ve **canlı kamera** (sanal kamera çıktısıyla).
[DeepFaceLive](https://github.com/iperov/DeepFaceLive)'ın `.dfm` modellerini ve algılama/hizalama çekirdeğini kullanır;
eski çok süreçli arayüz yerine tek süreçli bir motor, bir komut satırı aracı ve PyQt6 arayüzü sunar.

## Özellikler
- Fotoğraf, video ve klasör (toplu) işleme; video sesi korunur, dikey (telefon) videolar doğru yönde işlenir.
- **Yüksek kalite modu** (`--hq`): artırılmış yüz algılama, ortalanmış landmark'lar, iki geçiş, Lanczos birleştirme.
- **Yüz iyileştirici** (`--enhance`, GFPGAN 1.4): ayarlanabilir güç.
- **Canlı mod** + **sanal kamera** (v4l2loopback): OBS, Zoom, Discord'da kamera olarak seçilir.
- NVIDIA GPU (CUDA 12; sistem CUDA kurulumu gerekmez) veya yavaş da olsa CPU.

## Kurulum
Gereksinimler: Python 3.10–3.12, `ffmpeg`, (GPU için) NVIDIA sürücüsü.

```bash
git clone <bu-depo> && cd DeepFaceLive-Reborn
./install.sh          # NVIDIA GPU
./install.sh --cpu    # yalnızca CPU
```

`.dfm` modellerini `models/` klasörüne koy (örn. `models/Joker.dfm`). Modeller depoda yoktur;
[DeepFaceLive sürümlerinden](https://github.com/iperov/DeepFaceLive/releases) veya kendi eğittiklerinden kullanabilirsin.

Yüz iyileştirici için (isteğe bağlı, 340 MB):
```bash
mkdir -p models/enhancers
curl -L -o models/enhancers/gfpgan_1.4.onnx https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx
```

## Kullanım
```bash
./dfl.sh gui                                   # arayüz
./dfl.sh models                                # modelleri listele
./dfl.sh swap foto.jpg -m Joker                # fotoğraf
./dfl.sh swap video.mp4 -m Joker -o cikti.mp4  # video
./dfl.sh swap klasor/ -m Joker --all-faces     # klasör, tüm yüzler
./dfl.sh swap video.mp4 -m Joker --hq --enhance --enhance-strength 0.4
./dfl.sh live -m Joker --vcam /dev/video10     # pencere olmadan canlı
```
Tüm seçenekler: `./dfl.sh swap -h`. Önerilen hızlı deneme: `--limit 90` (ilk 90 kare).

### Sanal kamera (OBS / Zoom / Discord)
Bir kez cihazı oluştur (çekirdekte `v4l2loopback` modülü olmalı):
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFaceLive" exclusive_caps=1
```
Kalıcı yapmak için:
```bash
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 video_nr=10 card_label="DeepFaceLive" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
```
Arayüzde Canlı sekmesinde "Sanal kameraya gönder"i işaretle ve Başlat'a bas, **sonra** OBS'de
*Video Capture Device (V4L2)* kaynağı ekleyip **DeepFaceLive**'ı seç. Görüntü siyahsa Video Format'ı YUYV yap.
"Sanal kamerada aynala" kutusu, çıktıyı ayna gibi gönderir.

## Sorun giderme
- **`CUBLAS_STATUS_ALLOC_FAILED` / bellek hatası:** GPU belleği dolu. Başka programlar (örn. `ollama`) VRAM tutuyorsa kapat.
- **Yüz oturmuyor / bozuk:** Yüz kameraya yakın ve ön cepheden olmalı; `--hq` ve `--threshold` ile dene.
- **İyileştirici yapay detay üretiyor:** Gücü düşür (0.2–0.4).
- **Kamera açılmıyor:** `ls /dev/video*`; kullanıcı `video` grubunda olmalı.

## Yapı
```
dfl/        motor (engine.py), medya işleme, CLI, PyQt6 arayüzü, sanal kamera, iyileştirici
modelhub/   DFM model yükleyici, YOLOv5Face (algılama), FaceMesh (landmark) ONNX modelleri
xlib/       DeepFaceLive'dan alınan yardımcı kütüphane (yüz geometrisi, görüntü işleme, onnxruntime)
models/     .dfm modelleri ve models/enhancers/ (GFPGAN)
```

## Lisans ve atıflar
GPL-3.0 (bkz. `LICENSE`). Bu proje, [iperov/DeepFaceLive](https://github.com/iperov/DeepFaceLive) (GPL-3.0) kodundan türetilmiştir;
`xlib` ve `modelhub` oradan gelir. GFPGAN ve diğer modellerin kendi lisansları vardır; kullanmadan önce ilgili lisansı kontrol et.
Yalnızca izni olan yüzlerle ve yasalara uygun biçimde kullan.
