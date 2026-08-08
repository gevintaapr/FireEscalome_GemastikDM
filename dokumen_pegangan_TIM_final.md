# DOKUMEN PEGANGAN TIM — FINAL
## Decoding the Fire Escalome — GEMASTIK XIX/2026 Divisi Penambangan Data
**Versi Tim (3 orang) — dokumen kerja harian. Baca dari atas ke bawah sebelum mulai.**

---

## 0. Ringkasan Ide (30 detik)

Kita tidak memprediksi "akan kebakaran atau tidak". Kita membongkar **mekanisme** kenapa sebagian kecil titik panas berkembang jadi kebakaran besar (eskalasi) sementara yang lain padam sendiri — lalu mengelompokkan mekanisme itu jadi beberapa **archetype** (peat-driven, wind-driven, drought-driven, human-induced), dan menghasilkan katalog rule yang bisa langsung dipakai BPBD.

> 🗣️ **Bahasa gampangnya:** Bukan bikin aplikasi ramalan cuaca kebakaran. Kita kayak detektif — cari tahu "modus operandi" tiap tipe kebakaran, terus bikin buku panduan "kalau ciri-cirinya begini, berarti tipe kebakarannya X, tanganinnya harus dengan cara Y".

---

## 1. UPDATE BESAR: Pendekatan Graph (Spatio-Temporal Graph)

### 1.1 Kenapa ini ditambahkan?

Selama ini "hubungan antar titik panas" cuma dihitung sebagai satu angka: "berapa banyak titik panas di sekitar sini". Pak Daniel minta itu diperjelas jadi: **titik panas A dan B ini terhubung KARENA APA** — angin? gambut yang menyambung di bawah tanah? lereng yang curam? jalan yang jadi jalur pembukaan lahan?

> 🗣️ **Bahasa gampangnya:** Sebelumnya kita cuma tahu "di sekitar sini rame titik api". Sekarang kita mau tahu "titik api A ini nular ke titik api B karena angin apa karena gambutnya nyambung di bawah tanah". Itu beda level penjelasan.

### 1.2 Konsep Graf ($G = (V, E)$) — dijelaskan pelan-pelan

- **Node (titik)** = satu titik panas FIRMS, di lokasi dan waktu tertentu, dengan "KTP" berisi: elevasi, kemiringan lahan, NDVI, jarak ke jalan, status gambut.
- **Edge (garis penghubung)** = dua titik panas dianggap "terhubung" HANYA JIKA: (a) jaraknya dekat (≤5-10 km), DAN (b) waktunya berurutan (0 sampai 3 hari setelahnya).

> 🗣️ **Bahasa gampangnya:** Node itu kayak akun media sosial orang, edge itu kayak garis "berteman". Dua titik api "berteman" (jadi satu graf) kalau mereka deket jaraknya DAN muncul berurutan waktunya — bukan asal deket doang, harus juga masuk akal urutannya.

### 1.3 Bobot Edge ($W_{ij}$) — kenapa tidak sekadar "ada/tidak ada"

Edge tidak cuma bernilai ada/tidak, tapi punya **kekuatan koneksi**, dibentuk dari 4 faktor:

| Faktor | Rumus / Logika | Bahasa Gampangnya |
|---|---|---|
| **Spatial-Temporal Decay** | $\exp(-\alpha \cdot d_{ij}) \times \exp(-\beta \cdot \Delta t_{ij})$ | Makin deket & makin cepat jaraknya nyusul, makin kuat koneksinya — kayak sinyal wifi yang makin lemah kalau makin jauh. |
| **Wind Alignment** | $\cos(\theta)$, $\theta$ = sudut arah angin vs arah node i→j | Kalau angin bertiup PERSIS dari titik A ke titik B, api gampang banget nyebrang ke B. Kalau anginnya ke arah lain, koneksinya lemah. |
| **Slope Direction** | Node di lereng lebih tinggi & curam = koneksi lebih kuat | Api itu kayak orang lari — lebih cepat "lari" naik ke atas bukit yang curam dibanding jalan datar. |
| **Peatland Continuity** | Kalau kedua titik sama-sama di atas gambut / vegetasi kering = hambatan nol | Gambut itu kayak sumbu yang nyambung di bawah tanah — api bisa merambat pelan-pelan di bawah tanpa keliatan di permukaan, jadi dua titik yang JAUH di permukaan bisa tetap "terhubung" lewat bawah tanah. |

> 🗣️ **Bahasa gampangnya secara keseluruhan:** $W_{ij}$ itu kayak skor "seberapa yakin kita api di titik A bakal nular ke titik B", dihitung dari campuran jarak, arah angin, kemiringan tanah, dan jenis tanahnya.

### 1.4 Graph Explainability — INI YANG PALING PENTING, DAN INI YANG PALING BERISIKO

Permintaan Pak Daniel: "faktor apa yang paling nentuin dua node terhubung?" Secara teori dijawab pakai **GNNExplainer** atau **GraphSHAP** — tapi ini artinya harus melatih model Graph Neural Network (GAT) dulu, baru "dibongkar" pakai explainer. Ini beban baru yang besar: butuh PyTorch Geometric, GPU/training time, dan skill yang belum pernah dipakai tim untuk bagian manapun sejauh ini.

> 🗣️ **Bahasa gampangnya:** GNNExplainer itu kayak nanya ke kotak hitam "kenapa lo mikir gitu?" — tapi kita harus bikin dulu kotak hitamnya (latih model GAT), baru nanya. Bikin kotak hitamnya ini yang makan waktu & effort besar, apalagi belum pernah dicoba sebelumnya.

**REKOMENDASI SAYA — Graph-Lite (bukan Graph-Full):**

Karena rumus $W_{ij}$ di atas **SUDAH TRANSPARAN SEJAK AWAL** (bukan hasil model black-box), kita sebenarnya TIDAK BUTUH GNNExplainer sama sekali untuk menjawab "faktor apa yang dominan". Caranya:

1. Bangun graf beneran pakai `networkx` (bukan PyTorch Geometric) — hitung $W_{ij}$ langsung dari rumus di atas untuk semua pasangan node yang memenuhi syarat jarak+waktu.
2. Untuk tiap edge, kita SUDAH TAHU kontribusi tiap faktor (spatial decay, wind, slope, peatland) karena itu memang komponen rumus yang kita hitung sendiri — tinggal dipecah (decompose) per komponen, tidak perlu "dijelaskan" oleh model lain.
3. Kelompokkan edge berdasarkan faktor mana yang paling dominan nilainya → ini otomatis jadi bahan archetype (wind-dominant edges = kandidat wind-driven, dst).
4. Community detection pakai **Louvain algorithm** (`python-louvain`, bukan GAT) untuk menemukan kluster graf alami — dipakai sebagai VALIDASI TAMBAHAN untuk archetype yang sudah ditemukan lewat clustering fitur (Bagian 2, Step 7 di bawah), bukan mekanisme baru dari nol.

| | Graph-Lite (DIREKOMENDASIKAN) | Graph-Full (opsional, kalau waktu sisa banyak) |
|---|---|---|
| Tools | networkx, python-louvain | PyTorch Geometric, GAT, GNNExplainer |
| Skill baru dibutuhkan | Rendah (masih Python biasa) | Tinggi (deep learning graph, belum pernah dicoba tim) |
| Explainability | Otomatis transparan (rumus sendiri) | Butuh training + explainer terpisah |
| Risiko waktu | Rendah | Tinggi |
| Nilai naratif ke juri | Tetap kuat — "kami membangun graf spasial-temporal dan mendekomposisi faktor pemicu edge" | Lebih "wow" kalau berhasil, tapi risiko gagal/tidak sempat besar |

> 🗣️ **Bahasa gampangnya:** Daripada bikin robot pintar buat nebak-nebak kenapa dua titik nyambung (yang ribet & makan waktu), kita CATET SENDIRI dari awal kenapa dua titik itu nyambung (karena angin sekian, gambut nyambung, dst) — jadi nggak perlu nebak, dari awal udah jelas.

### 1.5 Tabel Interpretasi Archetype (dari Analisis Edge, dipakai apa adanya)

| Tipe Eskalasi | Faktor Dominan pada Edge | Interpretasi |
|---|---|---|
| Wind-driven | Wind alignment tinggi, slope datar | Api menyambung cepat antar-node karena arah & kecepatan angin |
| Peat-driven | Kontinuitas gambut tinggi, jeda waktu antar-node agak panjang | Node tetap "terhubung" walau jarak permukaan tidak berdekatan — api merambat di bawah tanah (smoldering) |
| Human-induced | Jarak jalan <1km, pola edge membentuk garis lurus menyusuri jalan | Pola pembukaan lahan terencana sepanjang akses jalan |
| Drought-driven | NDVI rendah, kekeringan tinggi di semua arah | Edge terbentuk masif ke berbagai arah (dense subgraph) karena seluruh bahan bakar vegetasi di kawasan itu kering |

---

## 2. PIPELINE LENGKAP END-TO-END (step by step)

### Step 1 — Setup Google Earth Engine
Daftar akun GEE (gratis untuk riset), autentikasi Python API (`earthengine-api`).
> 🗣️ Ini kayak daftar akun baru buat bisa minjem "peta satelit" dari Google — sekali doang di awal.

### Step 2 — Uji H1 (Paradoks Pareto)
Pakai data FIRMS yang sudah ada (394.000+ baris). Hitung: berapa % cluster yang "eskalasi" (persentil ke-95 rasio pertumbuhan) vs berapa % dari total FRP yang mereka sumbang.
> 🗣️ Ini nyari bukti: "bener nggak sih, cuma segelintir titik api yang bikin masalah paling gede?" — kalau kebukti, itu jadi alasan kuat kenapa riset ini penting.

### Step 3 — Tarik Data Ground Truth & Konteks (Tier 1-2, prioritas)
MCD64A1 (validasi kejadian), peta gambut, SRTM DEM, land cover, NDVI — semua lewat GEE, dikonversi jadi titik koordinat (lat/lon + nilai).
> 🗣️ Ini nyari "bukti pendukung" — apakah titik api ini beneran jadi area terbakar (dari satelit lain, bukan cuma FIRMS), ada di gambut apa nggak, tanahnya miring apa nggak, dst.

### Step 4 — Sanitasi & Validasi (urutan penting!)
1. Spatial-temporal join FIRMS × MCD64A1 → validasi kejadian nyata (BUKAN filter tutupan lahan dulu!)
2. Baru setelah itu, filter Land Cover → buang titik di area urban/industri, sisakan hutan/gambut/semak
> 🗣️ Validasi dulu "ini beneran kebakaran nggak", baru habis itu cek "ini kebakaran hutan apa bukan". Urutannya kebalik dari draft awal — sudah diperbaiki di dokumen penjelasan dosen.

### Step 5 — Bentuk Label Eskalasi Final
Gabungkan heuristik FIRMS (persentil-95) + konfirmasi MCD64A1 (BurnDate per piksel) → Label 1 kalau dua-duanya terpenuhi.

### Step 6 — Feature Engineering (tabular, untuk model utama)
Lag temporal, densitas tetangga spasial, fitur interaksi (misal: curah hujan kering × gambut).

### Step 7 — Bangun Graf (Graph-Lite, lihat Bagian 1)
Bentuk node+edge pakai `networkx`, hitung $W_{ij}$ dari 4 faktor, decompose per faktor.
> 🗣️ Ini bikin "peta jaringan" hubungan antar titik api, lengkap dengan alasan kenapa mereka terhubung.

### Step 8 — Modeling Cost-Sensitive
LightGBM + `scale_pos_weight` pada fitur tabular (Step 6) — model prediktif dasar, bukan tujuan akhir.

### Step 9 — SHAP Interaction Values
Cari pasangan fitur yang paling kuat berinteraksi memicu eskalasi — jadi input untuk clustering archetype.

### Step 10 — Archetype Discovery (2 sumber bukti digabung)
- Clustering pada fitur interaksi (dari Step 9)
- Community detection pada graf (dari Step 7, Louvain)
- Cek apakah dua-duanya "sepakat" pada archetype yang sama — kalau iya, itu bukti kuat archetype-nya nyata, bukan kebetulan.
> 🗣️ Kita cek dari dua arah berbeda — kalau ternyata dua-duanya nunjuk ke kesimpulan yang sama, berarti temuannya makin bisa dipercaya.

### Step 11 — Association Rule Mining
Pakai `mlxtend` (Apriori) untuk hasilkan rule IF-THEN per archetype, dengan support/confidence/lift.

### Step 12 — Pattern Library
Susun katalog akhir: nama archetype, rule, kondisi lingkungan, rekomendasi intervensi, contoh kasus nyata (dilengkapi visual false-color Sentinel-2 untuk 2-3 contoh).

### Step 13 — Simulasi Near Real-Time
Jalankan seluruh pipeline terhadap data FIRMS NRT + Open-Meteo Forecast terbaru sebagai bukti operasional.

### Step 14 — Penulisan Technical Report
8 bagian sesuai format Gemastik.

---

## 3. PEMBAGIAN TUGAS TIM (3 orang)

Prinsip pembagian: minimalkan orang saling menunggu — Person C bisa mulai kerja HARI INI juga tanpa nunggu siapa pun.

### 🧑‍💻 Person A — Data & Infrastruktur Lead
**Tanggung jawab:** semua yang berhubungan dengan GEE dan data mentah.
- Step 1 (setup GEE) — dikerjakan duluan, di hari 1 pagi, jangan digabung kerjaan lain
- Step 3 (tarik MCD64A1, peta gambut, SRTM, land cover, NDVI)
- Step 4 (sanitasi & validasi spasial-temporal)
- Step 5 (bentuk label final)
- Kalau waktu sisa: Tier 3 (OSM road distance, WorldPop)

### 🧑‍💻 Person B — Modeling & Graph Lead
**Tanggung jawab:** semua pemodelan, termasuk bagian graf yang baru.
- Step 6 (feature engineering tabular) — bisa mulai kerangka kodenya dari hari 1 pakai data FIRMS+cuaca yang sudah ada, baru disempurnakan begitu Step 5 (Person A) selesai
- Step 7 (bangun graf, hitung edge weight) — bisa mulai dari hari 1-2 karena cuma butuh FIRMS+cuaca (wind vector), tidak perlu nunggu MCD64A1
- Step 8 (LightGBM cost-sensitive)
- Step 9 (SHAP interaction values)
- Step 10 (archetype discovery — clustering + community detection)

### 🧑‍💻 Person C — Knowledge Extraction, Report & Ops Lead
**Tanggung jawab:** dari hipotesis sampai naskah jadi.
- Step 2 (uji H1) — **MULAI HARI INI JUGA**, data FIRMS sudah ada, tidak perlu nunggu siapa pun
- Riset & tulis Kajian Terkait (Bab 4 laporan) — cari 2-3 sumber pembanding occurrence-based, bisa paralel dari hari 1
- Step 11 (Association Rule Mining) — setelah Step 9-10 dari Person B selesai
- Step 12 (Pattern Library curation)
- Step 13 (simulasi NRT)
- Step 14 (penulisan Technical Report penuh) — koordinasi kumpulkan hasil dari A & B
- Persiapan Q&A dosen & presentasi

---

## 4. TIMELINE TERINTEGRASI (4–14 Agustus)

| Tanggal | Person A | Person B | Person C |
|---|---|---|---|
| **4 Agu** | Setup GEE, mulai tarik MCD64A1 | Kerangka kode feature engineering + mulai bangun graf (FIRMS+cuaca dulu) | Uji H1 + mulai riset Kajian Terkait |
| **5 Agu** | Selesaikan MCD64A1 + peta gambut + SRTM, bentuk label final | Lanjut graf: hitung edge weight lengkap | Lanjut Kajian Terkait, siapkan kerangka Technical Report |
| **6 Agu** | Land cover + NDVI | Feature engineering final (setelah label dari A selesai) | Review progress A & B, siapkan template Pattern Library |
| **7 Agu** | Bantu B kalau ada kendala data | Modeling LightGBM + SHAP Interaction | Mulai draft Bab 1-4 laporan |
| **8 Agu** | Tier 3 (OSM/WorldPop) kalau sempat | Archetype discovery (clustering + community detection graf) | Lanjut draft laporan |
| **9 Agu** | — | Bantu C kalau ada kendala teknis | Association Rule Mining |
| **10 Agu** | — | Review hasil graf & archetype untuk laporan | Pattern Library curation |
| **11–12 Agu** | Bantu tulis bagian data & metodologi | Bantu tulis bagian model & hasil | Kompilasi Technical Report 8 bagian penuh |
| **13 Agu** | Simulasi NRT bareng-bareng | Simulasi NRT bareng-bareng | Uji similaritas, surat pernyataan, polish |
| **14 Agu** | Review akhir bareng-bareng | Review akhir bareng-bareng | Submit |

---

## 5. Definition of Done — per Tahap (biar nggak ambigu "udah selesai belum")

| Tahap | Selesai kalau... |
|---|---|
| H1 | Ada angka pasti (%), bukan perkiraan, dikutip dengan sumber datanya |
| Label Eskalasi | Setiap baris data punya label 0/1 yang jelas dasarnya (heuristik + MCD64A1) |
| Graf | Semua edge yang valid (jarak+waktu terpenuhi) punya nilai $W_{ij}$ dan breakdown 4 komponennya |
| Archetype | Minimal 2 metode (clustering fitur + community detection graf) menunjukkan kluster yang konsisten |
| Rule Mining | Ada minimal beberapa rule per archetype dengan angka support/confidence/lift, bukan cuma deskripsi kualitatif |
| Pattern Library | Tiap archetype punya: nama, rule, kondisi lingkungan, rekomendasi intervensi, 1 contoh visual |

---

## 6. Checklist Final Sebelum Submit

- [ ] H1 diuji dengan data riil, angka dikutip di Bab 1
- [ ] Label eskalasi final = heuristik + validasi MCD64A1 (bukan cuma satu)
- [ ] Urutan sanitasi benar: validasi kejadian dulu, baru filter land cover
- [ ] Graf dibangun (Graph-Lite minimal), breakdown faktor edge weight terdokumentasi
- [ ] Archetype tervalidasi dari 2 sumber bukti (fitur + graf)
- [ ] Association Rule Mining menghasilkan rule dengan metrik lengkap
- [ ] Pattern Library tersusun rapi sebagai bab/lampiran
- [ ] Simulasi NRT dijalankan minimal sekali
- [ ] Setiap anggota tim paham bagian orang lain secukupnya untuk jawab pertanyaan dosen/juri (jangan cuma paham bagian sendiri!)
