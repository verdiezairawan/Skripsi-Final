# 🚀 Panduan Deployment Streamlit

## 📋 Opsi Deployment

### 1. **Deployment dengan TensorFlow (Lengkap)**
Gunakan `requirements.txt` untuk deployment dengan fitur prediksi lengkap:

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run streamlit_app.py
```

### 2. **Deployment Minimal (Tanpa TensorFlow)**
Gunakan `requirements_minimal.txt` untuk deployment tanpa TensorFlow:

```bash
# Install dependencies minimal
pip install -r requirements_minimal.txt

# Jalankan aplikasi (akan menggunakan mode fallback)
streamlit run streamlit_app.py
```

### 3. **Deployment dengan Versi Kompatibel**
Gunakan `requirements_deployment.txt` untuk versi yang lebih stabil:

```bash
# Install dependencies deployment
pip install -r requirements_deployment.txt

# Jalankan aplikasi
streamlit run streamlit_app.py
```

## 🔧 Konfigurasi Streamlit Cloud

### **File Konfigurasi: `.streamlit/config.toml`**
```toml
[server]
maxUploadSize = 200
enableXsrfProtection = false
enableCORS = false

[browser]
gatherUsageStats = false
```

### **File Requirements untuk Streamlit Cloud**
Pilih salah satu:
- `requirements.txt` - Untuk fitur lengkap
- `requirements_minimal.txt` - Untuk deployment minimal
- `requirements_deployment.txt` - Untuk versi stabil

## 🐛 Troubleshooting

### **Error: NumPy Version Conflict**
```
Because tensorflow-cpu==2.16.1 depends on numpy>=1.26.0,<2.0.0
```

**Solusi:**
1. Gunakan `requirements_minimal.txt` untuk deployment tanpa TensorFlow
2. Atau gunakan `requirements_deployment.txt` untuk versi yang kompatibel

### **Error: ModuleNotFoundError: No module named 'tensorflow'**
**Solusi:**
- Aplikasi akan otomatis menggunakan mode fallback
- Fitur prediksi akan menggunakan metode sederhana

### **Error: ModuleNotFoundError: No module named 'distutils'**
**Solusi:**
- Gunakan Python 3.11 atau 3.12
- Install `setuptools` terlebih dahulu

## 📊 Fitur yang Tersedia

### **Mode Lengkap (dengan TensorFlow):**
- ✅ Prediksi dengan model TCN-BiLSTM-GRU
- ✅ Prediksi multi-time (1, 3, 7, 14, 30 hari)
- ✅ Model fallback LSTM
- ✅ Data OHLCV dari CoinGecko

### **Mode Minimal (tanpa TensorFlow):**
- ✅ Tampilan data Bitcoin
- ✅ Prediksi sederhana berdasarkan trend
- ✅ Data OHLCV dari CoinGecko
- ✅ Grafik dan visualisasi

## 🌐 Deployment Platforms

### **1. Streamlit Cloud**
1. Upload ke GitHub
2. Connect ke Streamlit Cloud
3. Pilih requirements file
4. Deploy

### **2. Heroku**
1. Buat `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```
2. Deploy dengan requirements yang sesuai

### **3. Railway**
1. Connect repository
2. Pilih requirements file
3. Deploy otomatis

## 📝 Catatan Penting

1. **File Model**: Pastikan `model_tcn_bilstm_gru.h5` dan `scaler_btc.save` tersedia
2. **API Limits**: CoinGecko API memiliki rate limits
3. **Memory**: TensorFlow membutuhkan memory yang cukup
4. **Fallback**: Aplikasi selalu memiliki mode fallback

## 🔄 Update dan Maintenance

### **Update Dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

### **Test Lokal:**
```bash
python test_daily_data.py
python quick_coingecko_test.py
```

### **Monitor Logs:**
- Streamlit Cloud: Dashboard logs
- Heroku: `heroku logs --tail`
- Railway: Dashboard logs 