# AI Greenhouse Digital Twin Indonesia

> **Catatan:** Nama repository ini adalah `face-analysis-bot`, namun isi kode (`app.py`) saat ini adalah aplikasi **simulasi digital twin greenhouse (rumah kaca) pertanian**, bukan aplikasi analisis wajah. README ini dibuat berdasarkan isi kode yang sebenarnya ada di repo, bukan berdasarkan nama repo.

Aplikasi Streamlit untuk simulasi dan otomasi greenhouse pertanian di lokasi-lokasi Indonesia, menggunakan data cuaca real-time/near-real-time sebagai penggerak (forcing) model digital twin, lengkap dengan sistem kontrol otomatis (irigasi, ventilasi, shading, heating, injeksi CO2) dan beberapa jembatan (bridge) integrasi ke tool riset pertanian eksternal.

## Fitur Utama

- **Pengambilan data cuaca** dari beberapa sumber (dapat dipilih di sidebar):
  - OpenWeatherMap (butuh API key)
  - Meteostat (data historis/observasi terdekat)
  - Synthetic Fallback (data cuaca sintetis otomatis jika sumber lain gagal/tidak tersedia)
- **10 lokasi Indonesia** siap pakai: Jakarta, Bandung, Surabaya, Yogyakarta, Denpasar, Medan, Makassar, Palembang, Bogor, Malang.
- **Simulasi digital twin greenhouse** per jam: suhu dalam/luar, kelembapan, kelembapan tanah, CO2, biomassa, leaf area index, tahap pertumbuhan tanaman, indeks stres, risiko penyakit, dan perkiraan hasil panen (yield forecast).
- **Profil tanaman** yang bisa diatur: Tomato, Chili, Cucumber, Melon — dengan parameter suhu optimal dan target kelembapan tanah yang dapat disesuaikan.
- **Kontroler otomasi (AI controller)** dengan 3 mode:
  - *Rule-based expert* — aturan ambang batas untuk irigasi, ventilasi, fan, shading, heating, dan CO2.
  - *Predictive optimizer* — penyesuaian berdasarkan prediksi cuaca panas/hujan dan risiko penyakit.
  - *RL/ML-Agents ready* — kebijakan berbasis vektor observasi yang siap dihubungkan ke reinforcement learning.
- **Visualisasi analitik** (grafik suhu, water balance, pertumbuhan tanaman, risiko) menggunakan Matplotlib, serta log keputusan otomasi per jam.
- **Jembatan integrasi (connector bridges)** ke tool eksternal (opsional, terdeteksi otomatis jika package terpasang):
  - **GreenLightPlus (GLP)** — model greenhouse fidelitas tinggi.
  - **PCSE** — simulasi tanaman model WOFOST/LINGRA/LINTUL.
  - **FarmVibes.AI** — payload workflow (weather, NDVI, heatmap).
  - **Unity ML-Agents** — vektor observasi dan konfigurasi training PPO siap pakai.
- **Ekspor data**: tombol untuk mengunduh seluruh payload (cuaca, state twin, dan data konektor) sebagai file JSON.

## Tech Stack

- **Framework UI**: [Streamlit](https://streamlit.io/)
- **Data & komputasi**: `pandas`, `numpy`
- **Visualisasi**: `matplotlib`
- **HTTP client**: `requests` (untuk panggilan API OpenWeatherMap)
- **Opsional (terdeteksi otomatis, tidak wajib terpasang)**:
  - `meteostat` — sumber data cuaca alternatif
  - `GreenLightPlus` — model geometri/simulasi greenhouse
  - `pcse` — simulasi pertumbuhan tanaman
  - `mlagents_envs` — integrasi Unity ML-Agents

## Instalasi & Menjalankan

1. Clone repository:
   ```bash
   git clone https://github.com/Dhiyaahaq33/face-analysis-bot.git
   cd face-analysis-bot
   ```

2. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependency inti:
   ```bash
   pip install streamlit pandas numpy matplotlib requests
   ```

   Package opsional (jika ingin mengaktifkan fitur konektor terkait) dapat diinstall terpisah sesuai kebutuhan, misalnya:
   ```bash
   pip install meteostat
   # GreenLightPlus, pcse, dan mlagents_envs mengikuti dokumentasi masing-masing package
   ```

4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

5. Buka browser ke alamat lokal yang ditampilkan Streamlit (default `http://localhost:8501`).

## Konfigurasi Environment Variable

| Variable | Keterangan |
|---|---|
| `OPENWEATHER_API_KEY` | API key OpenWeatherMap (opsional). Bisa juga dimasukkan langsung lewat kolom input di sidebar aplikasi. Jika tidak diisi, aplikasi otomatis memakai data cuaca sintetis sebagai fallback. |

**Jangan pernah commit API key atau secret asli ke repository.** Simpan di environment variable lokal atau file `.env` (sudah masuk `.gitignore`).

## Struktur Repository

```
.
├── app.py                        # Aplikasi Streamlit utama (digital twin greenhouse)
├── LICENSE
├── FaceAnalysisBot_PRD_v1.docx   # Dokumen PRD (referensi terpisah)
└── bot context.pdf               # Dokumen konteks tambahan
```

## Lisensi

Lihat file [LICENSE](LICENSE) untuk detail lisensi proyek ini.
