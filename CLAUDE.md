# standup-clipper

Stand-up videosundan seyirci kahkahasına göre en komik bitleri bulur, Claude ile seçip Türkçeye çevirir,
Türkçe karaoke altyazılı 1080x1920 dikey klipler üretir. Windows, Python 3.11, RTX 4060.

## Kurallar
- Komutları her zaman `.venv\Scripts\python` ile çalıştır. Kod yorumları ve README Türkçe.
- Tüm ayarlar `config.yaml`'da. Ara çıktılar `work/{video}/*.json` (cache: dosya varsa adım atlanır).
- `samples/` ve `out/` içindeki dosyaları asla silme (istisna: kullanıcının panelden "Sil" ile sildiği video). Değişiklikten sonra `python -m pytest`.

## Klasörler
- `main.py` — CLI. `clipper/` — adım modülleri (her biri `run(video, cfg, force)`), `review.html` onay sayfası.
- `models/` (git dışı): jrgillick laughter-detection, MediaPipe yüz/YAMNet modelleri, `realesrgan/` (ncnn exe + modeller).
- `tools/convert_realesrgan.py` — realesr-general-x4v3 .pth → ncnn (resmi ncnn paketinde yok).
- `fonts/Montserrat-ExtraBold.ttf`, `tests/`, `samples/`, `work/`, `out/`.

## Pipeline (`RUN_STEPS`)
audio → transcribe (faster-whisper large-v3, GPU, noktalama için initial_prompt) → laughter (jrgillick ResNet,
CUDA torch, yedek YAMNet) → candidates (cümle sınırı: noktalama veya 0.6 sn+ duraklama) → select (Claude, kelime
indeksiyle başlangıç/bitiş; her 5 dk'ya ≤1 bit, ≤12, puan ≥6) → gapfill (seçili bitlerdeki boşlukları VAD'sız
yeniden transkribe) → translate (Claude, sistem prompt'unda çeviri kuralları, kapsama kontrolü, notlar) →
proofread (Claude, `translations_checked.json`) → subtitles (ASS) → reframe (PySceneDetect çekimleri + MediaPipe
yüz → çekim başına crop x) → [review: onay sayfası] → render (crop|blur_bg, isteğe bağlı Real-ESRGAN, altyazı en son).

## Web arayüzü
`clipper/web.py` (FastAPI, yalnız 127.0.0.1:8770) + `clipper/web.html` (vanilla JS SPA) + `clipper/jobs.py`
(tek işçili kuyruk, `work/_jobs.json`, loglar `work/_jobs/{id}.log`, SSE ile canlı). `Baslat.bat` başlatır.
Pipeline'ı `clipper/pipeline.py` üzerinden CLI ile aynı fonksiyonlarla çağırır. Paylaşım takibi `work/_shares.json`.
Sunucu logu `work/server.log` (beklenmeyen istek hataları yakalanır, arayüze Türkçe 500 döner).
- SPA: her sekme `route()`'un açtığı yeni kaba (`view`) çizer; async sayfalar kabı en başta yakalar, zamanlayıcılar
  `every(kap, ...)` ile kap DOM'dan çıkınca durur. Global `app`'e geç yazmak sekme hatasına yol açmıştı.
- Yükleme: `PUT /api/videos/upload?...&offset&total`, 8 MB'lık sıralı parçalar `.part`'a eklenir, en fazla 3 GB.
- Silme: `DELETE /api/videos/{ad}` samples/ + work/ siler, out/ korunur; iş sıradaysa/çalışıyorsa 409.
- Onay önizlemesi: altyazı ve notlar istemcide anlık WebVTT (Blob) olarak `<track>`'a bağlanır, düzenleyince yenilenir.

## CLI
```
python main.py run <video> --comedian "Ad" [--no-review]
python main.py run-all <klasör>          # onay açmadan, komedyen adı dosya adından (ad__gosteri.mp4)
python main.py review <video>
python main.py render <video> [--layout crop|blur_bg] [--upscale auto|always|off]
python main.py step <adim> <video>       # tek adımı zorla yeniden çalıştırır
```

## Önemli config
`render.layout`, `upscale.mode/tile/keep_frames`, `laughter.threshold` (0.4), `laughter.stride` (GPU'da 1),
`select.per_minutes/max_bits/min_score`, `candidates.max_len` (90), `subs.size/note_size/margin_v`,
`reframe.scene_threshold/min_shot/max_face_y`, `claude.mode` (cli = `claude -p`, abonelik | api), `claude.model`.

## Alınan kararlar
- Claude çağrıları `claude_client.ask_json`: JSON şema doğrulama + 2 yeniden deneme; cli modunda boş geçici
  klasörde, araçlar kapalı çalışır. Zamanları LLM değil kod hesaplar (kelime indeksi → zaman).
- jrgillick orijinal scripti modern sürümlerde çalışmıyor; yalnız model tanımı + checkpoint kullanılıyor.
- torch CUDA 13.2 derlemesi (2.15 için cu128 yok). Whisper (ctranslate2) pip'ten gelen cuBLAS/cuDNN 12 DLL'lerini kullanır.
- Yüz: önce short range, bulamazsa full range; karenin alt %40'ı seyirci sayılır.
- Varsayılan: `render.layout: crop`, `upscale.mode: "off"` (telefonda fark görünmedi; ayarlardan açılabilir).
  YAML'da `off` tırnaklı olmalı (çıplak `off` PyYAML'da False); web ayar yazımı seçenekleri tırnaklı yazar.
- Seçim: 30-60 sn'lik bitler tercih edilir; 60 sn yalnız kurulum gerektiriyorsa aşılır, en fazla `candidates.max_len`.
- Upscale yalnız crop düzeninde ve yalnız onaylı kliplerde; JPEG kareler tam aralıktan tv aralığına çevrilir.
- Komedyen adı dosya adından standart büyük harfle (yabancı isimler: "ismo" → "Ismo").

## Bilinen sorunlar
- Upscale yavaş: RTX 4060'ta ~7.9 kare/sn (85 sn'lik klip ≈ 5.3 dk toplam), geçici kareler birkaç GB.
- Crop çekim içinde sabit; komedyen tek çekimde çok gezerse kadrajdan kayabilir.
- Whisper bazen kahkaha altındaki konuşmayı atlar; gapfill kısmen kurtarır.
- `claude -p` kullanım limitine takılırsa adım hata verir; limit sıfırlanınca aynı komut cache'ten devam eder.
