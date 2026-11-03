# standup-clipper

Stand-up videosundan seyirci kahkahasına göre en komik bitleri bulur, Claude ile seçip Türkçeye çevirir,
Türkçe karaoke altyazılı 1080x1920 dikey klipler üretir. Windows, Python 3.11, RTX 4060.

## Kurallar
- Komutları her zaman `.venv\Scripts\python` ile çalıştır. Kod yorumları ve README Türkçe.
- Tüm ayarlar `config.yaml`'da. Ara çıktılar `work/{video}/*.json` (cache: dosya varsa adım atlanır).
- `samples/` ve `out/` içindeki dosyaları asla silme (istisna: kullanıcının panelden "Sil" ile sildiği video). Değişiklikten sonra `python -m pytest`.

## Klasörler
- `main.py` — CLI. `clipper/` — adım modülleri (her biri `run(video, cfg, force)`), `review.html` onay sayfası.
- `models/` (git dışı): jrgillick laughter-detection, YuNet + SFace (OpenCV zoo .onnx), MediaPipe pose/YAMNet modelleri, `realesrgan/` (ncnn exe + modeller).
- `tools/convert_realesrgan.py` — realesr-general-x4v3 .pth → ncnn (resmi ncnn paketinde yok).
- `config/censor_tr.txt` — küfür sansürü regex listesi ([strong] / [mild] / [except]), `clipper/censor.py`.
- `fonts/Montserrat-ExtraBold.ttf`, `tests/`, `samples/`, `work/`, `out/`.

## Pipeline (`RUN_STEPS`)
audio → transcribe (faster-whisper large-v3, GPU, noktalama için initial_prompt) → laughter (jrgillick ResNet,
CUDA torch, yedek YAMNet) → candidates (cümle sınırı: noktalama veya 0.6 sn+ duraklama) → select (Claude, kelime
indeksiyle başlangıç/bitiş; her 5 dk'ya ≤1 bit, ≤12, puan ≥6) → gapfill (seçili bitlerdeki boşlukları VAD'sız
yeniden transkribe) → translate (Claude, sistem prompt'unda çeviri kuralları, kapsama kontrolü, notlar) →
proofread (Claude, `translations_checked.json` + grup başına `profanity`) → captions (Claude, tüm bitler tek çağrıda, `captions.json`) → subtitles (ASS) → reframe (PySceneDetect çekimleri + YuNet/SFace komedyen
tanıma + pose gövde → çekim tipi yakın/geniş/ara → kadraj yolu) → [review: onay sayfası] → render (crop|blur_bg, isteğe bağlı Real-ESRGAN, altyazı en son).

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
python main.py captions <video>          # açıklamaları yeniden yazar, yalnız out/*.txt (render yok)
python main.py step <adim> <video>       # tek adımı zorla yeniden çalıştırır
python main.py step reframe-debug <video> <klip_no>   # kadraj hata ayıklama videosu → work/debug/
```

## Önemli config
`render.layout`, `upscale.mode/tile/keep_frames`, `laughter.threshold` (0.4), `laughter.stride` (GPU'da 1),
`select.per_minutes/max_bits/min_score`, `candidates.max_len` (90), `subs.size/note_size/credit_y`, `safe_zone` (altyazı+not y 1050–1450, sol 60 / sağ 140 px),
`reframe.face_match/min_face/close_face/min_detect_rate/max_jump/cutaway_mode/tiny_face_ratio/tiny_edge/tiny_subject_mode/track_margin/end_shot/wide_margin/max_speed/dir_hold/min_move/center_band`, `censor.enabled/mode (full|first_letter)/include_mild`, `claude.mode` (cli = `claude -p`, abonelik | api), `claude.model`.

## Alınan kararlar
- Claude çağrıları `claude_client.ask_json`: JSON şema doğrulama + 2 yeniden deneme; cli modunda boş geçici
  klasörde, araçlar kapalı çalışır. Zamanları LLM değil kod hesaplar (kelime indeksi → zaman).
- jrgillick orijinal scripti modern sürümlerde çalışmıyor; yalnız model tanımı + checkpoint kullanılıyor.
- torch CUDA 13.2 derlemesi (2.15 için cu128 yok). Whisper (ctranslate2) pip'ten gelen cuBLAS/cuDNN 12 DLL'lerini kullanır.
- Kadraj "kameraman" gibi: yalnız komedyen. Referans yüz = tüm kliplerin yakın çekim yüzlerinin (SFace) en kalabalık
  benzerlik kümesi (`reframe.json` → `_ref`). Eşik altı yüzler (seyirci, tablo) ve `min_face`'ten küçük yüzler (uzak
  seyircinin vektörü güvenilmez, 0.36–0.39 ile yanlış eşleşiyordu) yok sayılır. Takip sürekliliği: çekimde son
  tanınan yüze konum (bir yüz boyu) ve boyutça (0.7–1.4 kat) yakın yüz `face_match - track_margin` ile kabul (baş
  dönünce düşen benzerlik). Yüz tanınmazsa sahne bölgesindeki (burun üst %60) pose kişisi yalnız çekimde tanınmış
  komedyene yakınsa ya da çekimde tanıma yokken tek belirgin kişiyse (tek pose / omuzları 2 kat geniş).
  Çekim içinde 3 sn kayan medyandan sapan tekil konumlar atılır.
- `blur` işaretli çekimler kırpılmaz (render: `overlay=enable='gte(t,a)*lt(t,b)'`, düzen tam çekim sınırında değişir):
  ara + `cutaway_mode: blur_bg`; geniş ve yüz < `tiny_face_ratio` (%4) ya da < %8 ve gövde kaynak karenin kenar
  %20'sinde (`tiny_edge`; Joe #5 51–60 sn yüzü %4.7, 0–11 sn ile aynı boyut, fark konum). Son çekim < `end_shot`
  (1.5 sn) ve orada yüz tanınmadıysa: kahkaha kesilmiyorsa `reframe.json` → `end` (render klibi kısaltır), kesiliyorsa blur.
- Çekim tipi: ara (tespit < %50 ya da zıplıyor) → sabit merkez | `cutaway_mode: blur_bg`;
  geniş → gövde (omuz-kalça) aralığı sığıyorsa tek sabit kadraj, yoksa sabit parçalar + cos ease geçiş (≤150 px/sn,
  yön değişimi öncesi 2 sn); yakın → orta %40 ölü bölge + 3 sn kayan medyan (1.5 sn'den kısa hareket yok) + hız sınırı
  (ileri/geri ortalamalı, gecikmesiz). 0.7 sn'den kısa çekim önceki kadrajı kullanır. Anahtar `[t, x, 1]` = ease.
- Titreme metriği (`reframe.jitter`, 1080p ölçeğinde px/sn ve 3 sn başına yön değişimi) her klip için loglanır ve
  `reframe.json`'da (`jitter`, `jitter_calm` = geniş+ara). `in_band` ara çekimleri saymaz.
  Hata ayıklama: `python main.py step reframe-debug <video> <klip>` → `work/debug/`. Analiz ~60 sn / 85 sn'lik klip (CPU).
- Yerleşim: hook üstte alttan hizalı, komedyen adı hemen altında (yarı saydam, hook ile birlikte kaybolur); altyazı alttan `text_bottom`'a,
  notlar üstten `text_top`'a hizalı. `test_rendered_text_stays_in_safe_zone` libass çıktısını pikselden ölçer.
- Varsayılan: `render.layout: crop`, `upscale.mode: "off"` (telefonda fark görünmedi; ayarlardan açılabilir).
  YAML'da `off` tırnaklı olmalı (çıplak `off` PyYAML'da False); web ayar yazımı seçenekleri tırnaklı yazar.
- Seçim: 30-60 sn'lik bitler tercih edilir; 60 sn yalnız kurulum gerektiriyorsa aşılır, en fazla `candidates.max_len`.
- Upscale yalnız crop düzeninde ve yalnız onaylı kliplerde; JPEG kareler tam aralıktan tv aralığına çevrilir.
- Açıklama (`captions.py`): Claude yalnız 1. satırı (hook, ≤80, punchline'ı ele vermez) ve konu etiketini yazar;
  `🎤 Komedyen – Gösteri` satırını ve sabit etiketleri (#standup #standupkomedi #türkçealtyazı #komedyen) kod ekler.
  Kurallar `make_check`'te (bağlaçla başlama, ≤1 emoji, reklam kalıbı, genel etiket yasak). Seçim artık açıklama üretmez.
- Gösteri adı: `meta.json`'daki `show`, yoksa dosya adındaki `__` sonrası (`common.video_meta`); panelde satıra tıklayıp düzenlenir.
- Küfür sansürü yalnız çıktıda (ASS altyazı/not/hook, out/*.txt); translations/captions JSON orijinal kalır. Kaynak:
  proofread'in `profanity` listesi (tam kelime biçimi) ∪ `config/censor_tr.txt` (kesmeden önceki kök, fullmatch, istisnalar
  önce) ∪ onay sayfasındaki ekle/kaldır (`review.json` → `censor: {n: {add, remove}}`). Yıldız sayısı = harf sayısı,
  bu yüzden karaoke zamanı değişmez. Panel önizlemesi `POST /api/videos/{ad}/censor` ile sunucuda hesaplanır.
  Eski videoların `translations_checked.json`'unda `profanity` yok → yalnız regex listesi.
- Komedyen adı dosya adından standart büyük harfle (yabancı isimler: "ismo" → "Ismo").

## Bilinen sorunlar
- Upscale yavaş: RTX 4060'ta ~7.9 kare/sn (85 sn'lik klip ≈ 5.3 dk toplam), geçici kareler birkaç GB.
- Komedyen kaynak karenin kenarındaysa crop daha öteye kayamaz; orta %40 kuralı orada sağlanamaz.
- Komedyen arkadan/profilden görünüyorsa yüz tanınmaz; pose ile takip edilir. Upscale yolunda blur işaretli çekimler uygulanmaz.
- Komedyen adı (y≈470) ilk 3 sn yakın çekimlerde başın üstüne denk gelebilir.
- Whisper bazen kahkaha altındaki konuşmayı atlar; gapfill kısmen kurtarır.
- `claude -p` kullanım limitine takılırsa adım hata verir; limit sıfırlanınca aynı komut cache'ten devam eder.
