# 📈 Bitcoin Price Prediction App

Aplikasi prediksi harga Bitcoin menggunakan model TCN-BiLSTM-GRU dengan data real-time dari CoinGecko API.

## 🚀 Fitur

- **Data Real-time**: Mengambil data OHLCV Bitcoin dari CoinGecko API
- **Prediksi Multi-time**: Prediksi untuk 1, 3, 7, 14, dan 30 hari ke depan
- **Model AI**: Menggunakan model TCN-BiLSTM-GRU dengan fallback LSTM
- **Visualisasi**: Grafik interaktif dengan Plotly
- **Mode Fallback**: Tetap berfungsi meskipun TensorFlow tidak tersedia

## 📦 Instalasi

### Opsi 1: Deployment Minimal (Direkomendasikan)
```bash
pip install -r requirements_minimal.txt
streamlit run streamlit_app.py
```

### Opsi 2: Deployment dengan TensorFlow
```bash
pip install -r requirements_deployment.txt
streamlit run streamlit_app.py
```

## 🔧 Konfigurasi

### File Requirements
- `requirements_minimal.txt` - Tanpa TensorFlow (untuk deployment)
- `requirements_deployment.txt` - Dengan TensorFlow (versi stabil)
- `requirements.txt` - Versi terbaru (mungkin ada konflik)

### File Model
- `model_tcn_bilstm_gru.h5` - Model TCN-BiLSTM-GRU
- `scaler_btc.save` - StandardScaler untuk normalisasi data

## 📊 Penggunaan

1. **Jalankan aplikasi**: `streamlit run streamlit_app.py`
2. **Pilih periode prediksi**: 1, 3, 7, 14, atau 30 hari
3. **Klik "Jalankan Prediksi Multi-time"** untuk prediksi
4. **Lihat hasil**: Tabel prediksi dan grafik visualisasi

## 🌐 Deployment

### Streamlit Cloud
1. Upload ke GitHub
2. Connect ke Streamlit Cloud
3. Pilih `requirements_minimal.txt` sebagai requirements file
4. Deploy

### Platform Lain
Lihat `DEPLOYMENT_GUIDE.md` untuk panduan lengkap.

## 📁 Struktur Proyek

```
├── streamlit_app.py          # Aplikasi utama
├── requirements_minimal.txt  # Dependencies minimal
├── requirements_deployment.txt # Dependencies dengan TensorFlow
├── requirements.txt          # Dependencies lengkap
├── model_tcn_bilstm_gru.h5  # Model AI
├── scaler_btc.save          # Scaler
├── .streamlit/config.toml   # Konfigurasi Streamlit
├── DEPLOYMENT_GUIDE.md      # Panduan deployment
├── .gitignore               # File yang diabaikan Git
└── README.md                # Dokumentasi ini
```

## 🔄 Mode Fallback

Jika TensorFlow tidak tersedia, aplikasi akan:
- Menggunakan prediksi sederhana berdasarkan trend
- Tetap menampilkan data Bitcoin real-time
- Menyediakan visualisasi dan grafik

## 📝 Catatan

- Data diambil dari CoinGecko API (rate limited)
- Model membutuhkan data 60 hari terakhir untuk prediksi
- Aplikasi di-cache selama 10 menit untuk performa

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur baru
3. Commit perubahan
4. Push ke branch
5. Buat Pull Request

## 📄 Lisensi

Proyek ini untuk tujuan akademis dan penelitian.
