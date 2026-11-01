# Sabah raporu — 2 Kasım 2026

## Kısaca
Bütün maddeler bitti; 46 test geçiyor. Web paneli çalışıyor: masaüstündeki **Standup Clipper** kısayolu ya da
`Baslat.bat` ile açılıyor (http://127.0.0.1:8770/). 5 dakikalık ISMO parçası panelden baştan sona işlendi:
ekle → başlat → onay → render → çıktılar. Gerçek test (madde 3) atlandı, çünkü `samples/` içinde ISMO dışında
video yok.

## Neler yapıldı
- **0) Kalite yükseltme karşılaştırması:** `work/compare.png` hazır. Klip 1'in aynı anı crop, crop+upscale ve
  blur_bg olarak yan yana; alt satırda 1:1 yakın görünüm var. Upscale'de göz, sakal ve saç kenarları belirgin
  biçimde daha net. Varsayılan düzene (crop) dokunmadım.
  - Bu sırada iki hata buldum ve düzelttim: upscale çıktısı `yuvj420p` (tam renk aralığı) çıkıyordu, artık
    standart `yuv420p`. Ayrıca bir hata olduğunda ara kareler diskte kalıyordu, artık siliniyor.
- **1) CLAUDE.md:** amaç, klasörler, pipeline, CLI, önemli ayarlar, kararlar, bilinen sorunlar.
- **2) Web arayüzü:** FastAPI, yalnızca 127.0.0.1'e bağlı, onay sayfasıyla aynı koyu tema.
  - **Kütüphane:** sürükle-bırak ile ekleme; dosya adı otomatik `komedyen__gosteri.mp4`. 720p altı videolarda
    kırmızı uyarı. Durum etiketleri: yeni / sırada / işleniyor / onay bekliyor / render edildi / hata.
  - **İşleme:** adım kutuları, süreler, SSE ile canlı log. Hata olursa Türkçe özet ve "Kaldığı yerden devam et".
    Claude limiti ayrıca tanınıyor ("Claude limiti doldu" + 30 dakikada bir otomatik yeniden deneme seçeneği).
  - **Onay:** her klibin yanında kaynak videonun o aralığını oynatan önizleme; altyazısız kelimeler kırmızı;
    "Kaydet ve render et" (düzen ve upscale seçilebilir).
  - **Çıktılar:** ızgara, tarayıcıda oynatma, Kopyala, Klasörde göster, YouTube/Instagram/TikTok işaretleri
    tarihleriyle, "henüz paylaşılmamış" filtresi.
  - **Ayarlar:** istenen 9 ayar form olarak; `config.yaml` yorumları korunarak kaydediliyor.
  - **İş kuyruğu:** aynı anda tek iş çalışıyor. Durum `work/_jobs.json`'da tutuluyor; sunucu kapanıp açılırsa
    yarıda kalan iş kaldığı yerden devam ediyor.
  - **CLI aynen çalışıyor:** arayüz ve CLI ortak `clipper/pipeline.py`'yi kullanıyor.

## Test sonuçları
- **pytest:** 46/46 geçti. Bunların 14'ü yeni API testi: yükleme, kütüphane, iş akışı, limit tanıma, otomatik
  yeniden deneme, yeniden başlatma sonrası devam, onay kaydetme + render, dosya yolu güvenliği, video Range
  isteği, paylaşım takibi, ayarlar (yorumlar korunuyor mu), SSE.
- **Arayüz uçtan uca (`tools/ui_e2e.py`, Edge ile tıklayarak):**

  | Aşama | Süre / sonuç |
  |---|---|
  | Yükleme | 0.3 sn |
  | İşleme toplam | **171 sn** |
  | Seçim | 1 bit (5 dakikalık video için hedef 1), 8/10 puan, 69 sn |
  | Render (upscale ile) | **273 sn** |
  | Önizleme oynatıcı | doğru aralığa atladı |
  | Paylaşım işareti + filtre | çalıştı |

  İşleme adımları: transcribe 65 sn, gapfill 32 sn, laughter 19.5 sn, çeviri 16.8 sn, seçim 15.4 sn,
  kontrol 11.2 sn, kadraj 9.2 sn.
- **Ekran görüntüleri:** `work/screenshots/` altında 10 sayfa.
- **Upscale hızı (RTX 4060):** ~7.9 kare/sn. 85 sn'lik klip ≈ 5.3 dk, 69 sn'lik klip ≈ 4.5 dk.

## Karşılaşılan sorunlar
- Upscale çıktısının renk aralığı yanlıştı; düzeltildi ve yeniden render edildi.
- Arayüzde iki görsel hata vardı ("null" yazısı, okunmayan bağlantı rengi); düzeltildi.
- Yüklemede yazılan komedyen adı ("ISMO") kaydedilmiyor, dosya adından "Ismo" türetiliyordu. Düzelttim, ama
  test videosunun çıktısındaki imza düzeltmeden önce render edildiği için "Ismo" yazıyor.
- Windows'ta tarayıcı canlı log bağlantısını kapatınca konsolda zararsız bir `ConnectionResetError` uyarısı
  çıkıyor; işlevi etkilemiyor.
- `ismo_test.mp4`'ün iş dosyaları eski sürümden kalma. Onay sayfası bu video için "henüz işlenmedi" diyor;
  panelden "Başlat" ile yeniden işlenirse düzelir.

## Senin karar vermen gerekenler
1. **Düzen:** `work/compare.png`'ye bakıp crop ya da blur_bg seç (Ayarlar → Düzen).
2. **Upscale varsayılanı `auto`:** 1080p kaynakta her klip büyütülüyor. ISMO'nun 12 klibi (toplam 12.5 dk) için yaklaşık 47 dakika sürer.
   Hız istersen `off` yap; kalite istersen `auto` kalsın.
3. **Klip uzunluğu:** `max_len` 90 olduğu için ISMO'nun 12 klibinin 8'i 60 saniyeden uzun. YouTube Shorts ve
   bazı platformlar 60 saniye sınırı uyguluyor; 60'a düşürmek isteyebilirsin.
4. **Test dosyası:** panel testi `samples/ismo__5_dk_test.mp4`'ü ekledi (`ismo_test.mp4`'ün kopyası). Gerekmiyorsa
   elle silebilirsin; talimat gereği silmedim.

## Önerilen sonraki adımlar
- Farklı bir komedyenin gösterisini panelden ekleyip yeni sürümü gerçek bir ikinci içerikte dene (madde 3).
- Çekim içinde komedyen çok geziyorsa kare kare kadraj takibi ekle; şu an her çekimde tek kesim konumu var.
- Upscale'i kuyrukta ayrı bir iş olarak çalıştır, böylece önizleme hızlı çıksın ve kaliteli sürüm sonradan gelsin.
- 60 saniye sınırı seçilirse Claude'a "en fazla 60 sn" kuralını ekle.
