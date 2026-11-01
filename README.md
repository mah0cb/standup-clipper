# standup-clipper

Bilgisayardaki bir stand-up videosundan en komik bitleri **seyirci kahkahasına** bakarak bulur,
Claude ile seçip Türkçeye çevirir ve **Türkçe karaoke altyazılı 9:16 dikey** kısa videolar üretir.

```
video.mp4 → ses → transkript → kahkaha → adaylar → Claude seçimi → Claude çevirisi → Claude çeviri kontrolü
          → ASS altyazı → yüz takipli crop → [onay sayfası] → render (mp4 + açıklama .txt)
```

## Gereksinimler

- Windows 10/11, **Python 3.11**
- **ffmpeg** PATH'te (libass içeren "full" sürüm; ör. `winget install Gyan.FFmpeg`)
- NVIDIA GPU önerilir (Whisper large-v3). GPU yoksa otomatik CPU + int8 + `medium`.
- Claude: varsayılan `cli` modu için [Claude Code](https://claude.com/claude-code) kurulu ve giriş yapılmış olmalı
  (`claude` komutu çalışmalı). `api` modu için Anthropic API anahtarı.

## Kurulum (adım adım)

```powershell
cd standup-clipper
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Modeller ve font (bir kez):

```powershell
mkdir models, fonts
curl.exe -L -o models\yamnet.tflite https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
curl.exe -L -o models\blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
curl.exe -L -o models\blaze_face_full_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite
curl.exe -L -o fonts\Montserrat-ExtraBold.ttf https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf
git clone --depth 1 https://github.com/jrgillick/laughter-detection models\laughter-detection
```

Whisper modeli (~3 GB) ilk çalıştırmada otomatik indirilir.

`api` modunu kullanacaksanız `.env.example` dosyasını `.env` olarak kopyalayıp `ANTHROPIC_API_KEY` girin
ve `config.yaml` içinde `claude.mode: api` yapın.

## Kullanım

```powershell
# 1–8. adımlar, sonunda tarayıcıda onay sayfası açılır
python main.py run samples\gosteri.mp4 --comedian "Komedyen Adı"

# Onay sayfasında: render edilecek klipleri işaretleyin, hook başlığını ve altyazı satırlarını düzeltin,
# "Kaydet"e basın, terminalde Ctrl+C ile sunucuyu kapatın. Sonra:
python main.py render samples\gosteri.mp4

# Tek bir adımı yeniden çalıştırmak (önceki adımlar cache'ten gelir):
python main.py step translate samples\gosteri.mp4
```

Adım adları: `audio, transcribe, laughter, candidates, select, translate, proofread, subtitles, reframe, render, review`.

### Çıktılar

| Yer | İçerik |
|---|---|
| `work/{video}/audio.wav` | 16 kHz mono ses |
| `work/{video}/words.json`, `segments.json` | Kelime zaman damgalı transkript |
| `work/{video}/laughter.json` | `[{start, end, prob}]` kahkaha aralıkları |
| `work/{video}/candidates.json` | En iyi 40 aday (cümle başından kahkaha sonuna) |
| `work/{video}/selected.json` | Claude'un seçtiği bitler (puan, gerekçe, hook, açıklama, hashtag) |
| `work/{video}/translations.json` | Zamana hizalı Türkçe ifade grupları, açıklama notları, ABD→TR uyarlama listesi |
| `work/{video}/translations_checked.json` | Çeviri kontrolünden geçmiş hali + yapılan değişiklikler (altyazı ve onay sayfası bunu kullanır) |
| `work/{video}/subs/{n}.ass` | Karaoke altyazı |
| `work/{video}/reframe.json` | 9:16 crop kutuları |
| `work/{video}/review.json` | Onaylanan klipler |
| `work/{video}/timings.json` | Her adımın süresi |
| `out/{video}/{n}_{hook}.mp4` + `.txt` | Dikey video + açıklama, 5 hashtag, komedyen adı |

**Cache:** Her adımın JSON çıktısı varsa yeniden hesaplanmaz. Bir adımı zorla yeniden yapmak için `step`
komutunu kullanın; sonraki adımları da güncellemek isterseniz onları da sırayla `step` ile çalıştırın.

## Ayarlar (`config.yaml`)

Tüm modeller, eşikler, klip uzunlukları, font, renkler ve Claude modu buradadır. Sık değiştirilenler:

- `laughter.threshold` — kahkaha bulunamıyorsa düşürün (ör. 0.35), fazla buluyorsa yükseltin.
- `select.per_minutes`, `select.max_bits`, `select.min_score` — her 5 dakikaya en fazla 1 bit, üst sınır 12, 6 puan altı elenir.
- `candidates.pause` — bundan uzun duraklamalar da cümle sınırı sayılır; klipler yalnızca cümle başında başlar.
- `laughter.stride` — GPU'da (CUDA torch) 1 önerilir (5 dk ses ~15 sn). CPU'da 4 yapın (~4 kat hızlı).
- `subs.margin_v`, `subs.size` — altyazının yeri ve boyutu.
- `claude.mode` — `cli` (aboneliğiniz, ek ücret yok) veya `api`.

## Notlar

- **Kahkaha modeli:** `jrgillick/laughter-detection` kullanılır. Orijinal script eski sürümlere bağlı
  (torch 1.3, librosa 0.8) olduğundan doğrudan çalışmaz; `clipper/laughter.py` yalnızca reponun model
  tanımını ve checkpoint'ini kullanır, özellik çıkarımını yeni librosa API'siyle yapar. Model yüklenemezse
  otomatik olarak YAMNet'in (MediaPipe) kahkaha sınıflarına geçer (`laughter.backend: yamnet` ile zorlanabilir).
- **Crop:** Her klip için yüz merkezlerinin yumuşatılmış medyanıyla sabit bir 9:16 kutu kullanılır.
  Önce yakın çekim, yüz bulunamazsa uzak mesafe (full range) modeli denenir; karenin alt kısmındaki
  yüzler seyirci sayılıp elenir (`reframe.max_face_y`). Hiç yüz yoksa merkez crop.
- **Loglar:** Konsol + `clipper.log`. Her Claude çağrısının süresi (api modunda token ve maliyet) loglanır.

## Testler

```powershell
python -m pytest
```
