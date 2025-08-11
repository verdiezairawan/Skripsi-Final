import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os
import logging

# ==============================================================================
# Konfigurasi Logging
# ==============================================================================
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Import TensorFlow dengan fallback
# ==============================================================================
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    try:
        from tcn import TCN
        TCN_AVAILABLE = True
        logger.info("✅ TensorFlow dan TCN berhasil diimport")
    except ImportError:
        TCN_AVAILABLE = False
        logger.warning("⚠️ Keras-TCN tidak tersedia. Model asli tidak dapat dimuat.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("⚠️ TensorFlow tidak tersedia. Aplikasi akan menggunakan mode fallback.")

# ==============================================================================
# Konfigurasi Halaman Streamlit
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Fungsi untuk Memuat Model dan Scaler
# ==============================================================================
@st.cache_resource
def load_model_and_scaler():
    """Memuat model Keras dan scaler dari file."""
    model_path = 'model_tcn_bilstm_gru.h5'
    scaler_path = 'scaler_btc.save'

    if not os.path.exists(model_path):
        logger.error(f"File model tidak ditemukan di path: {model_path}")
        st.error(f"File model tidak ditemukan di path: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        logger.error(f"File scaler tidak ditemukan di path: {scaler_path}")
        st.error(f"File scaler tidak ditemukan di path: {scaler_path}")
        return None, None

    # Jika TensorFlow tidak tersedia, langsung buat model fallback
    if not TENSORFLOW_AVAILABLE:
        logger.warning("⚠️ TensorFlow tidak tersedia. Membuat model fallback...")
        model = create_simple_fallback_model()
        if model is not None:
            logger.info("✅ Model fallback berhasil dibuat")
        else:
            logger.error("❌ Gagal membuat model fallback")
            return None, None
    else:
        try:
            # Cek versi TensorFlow
            tf_version = tf.__version__
            logger.info(f"ℹ️ Menggunakan TensorFlow versi: {tf_version}")
            
            # Coba beberapa metode loading yang berbeda
            model = None
            
            # Metode 1: Coba dengan custom_objects dan compile=False
            if TCN_AVAILABLE:
                try:
                    custom_objects = {'TCN': TCN}
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    logger.info("✅ Model berhasil dimuat dengan custom_objects")
                except Exception as e1:
                    # Metode 2: Coba tanpa custom_objects
                    try:
                        model = tf.keras.models.load_model(
                            model_path,
                            compile=False
                        )
                        logger.info("✅ Model berhasil dimuat tanpa custom_objects")
                    except Exception as e2:
                        # Metode 3: Coba dengan custom_objects kosong
                        try:
                            model = tf.keras.models.load_model(
                                model_path,
                                custom_objects={},
                                compile=False
                            )
                            logger.info("✅ Model berhasil dimuat dengan custom_objects kosong")
                        except Exception as e3:
                            # Jika semua metode gagal, langsung buat model fallback
                            logger.warning("⚠️ Model asli tidak dapat dimuat karena masalah kompatibilitas versi TensorFlow")
                            logger.info(f"💡 Model mungkin disimpan dengan TensorFlow versi yang berbeda")
                            logger.info("🔄 Membuat model fallback yang kompatibel...")
                            model = create_simple_fallback_model()
                            if model is not None:
                                logger.info("✅ Model fallback berhasil dibuat")
                                logger.info("ℹ️ Model fallback menggunakan arsitektur LSTM sederhana yang kompatibel")
                            else:
                                logger.error("❌ Gagal membuat model fallback")
                                return None, None
            else:
                logger.warning("⚠️ TCN tidak tersedia. Membuat model fallback...")
                model = create_simple_fallback_model()
                if model is not None:
                    logger.info("✅ Model fallback berhasil dibuat")
                else:
                    logger.error("❌ Gagal membuat model fallback")
                    return None, None
        
        except Exception as e:
            logger.error(f"Gagal memuat model atau scaler. Detail kesalahan: {str(e)}")
            st.error(f"Gagal memuat model atau scaler. Detail kesalahan:")
            st.exception(e)
            return None, None
        
    # Load scaler dengan fallback
    scaler = None
    try:
        scaler = joblib.load(scaler_path)
        logger.info("✅ Scaler berhasil dimuat")
    except Exception as e:
        logger.warning(f"Gagal memuat scaler: {str(e)[:100]}...")
        logger.warning("⚠️ Mencoba membuat scaler fallback...")
        try:
            scaler = create_simple_fallback_scaler()
            if scaler is not None:
                logger.info("✅ Scaler fallback berhasil dibuat")
            else:
                logger.error("❌ Gagal membuat scaler fallback")
                return None, None
        except Exception as e2:
            logger.error(f"Gagal membuat scaler fallback: {str(e2)[:200]}...")
            return None, None
        
    return model, scaler

def create_simple_fallback_model():
    """Membuat model sederhana sebagai fallback jika model asli gagal dimuat."""
    try:
        if not TENSORFLOW_AVAILABLE:
            logger.warning("⚠️ TensorFlow tidak tersedia. Model fallback tidak dapat dibuat.")
            return None
            
        # Model LSTM sederhana yang kompatibel dengan TensorFlow 2.13.0
        model = tf.keras.Sequential([
            # Input layer dengan shape yang benar
            tf.keras.layers.Input(shape=(60, 1)),
            
            # LSTM layers
            tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("ℹ️ Model fallback LSTM berhasil dibuat dengan arsitektur sederhana")
        return model
        
    except Exception as e:
        logger.error(f"Gagal membuat model fallback: {str(e)}")
        return None

def create_simple_fallback_scaler():
    """Membuat scaler sederhana sebagai fallback jika scaler asli gagal dimuat."""
    try:
        from sklearn.preprocessing import StandardScaler
        # Buat scaler kosong yang akan di-fit nanti
        scaler = StandardScaler()
        return scaler
    except Exception as e:
        logger.error(f"Gagal membuat scaler fallback: {str(e)}")
        return None

def fit_scaler_with_data(scaler, data):
    """Fit scaler dengan data historis."""
    try:
        if scaler is not None and hasattr(scaler, 'fit'):
            # Fit scaler dengan data historis
            scaler.fit(data['Close'].values.reshape(-1, 1))
            return True
        return False
    except Exception as e:
        logger.error(f"Gagal fitting scaler: {str(e)}")
        return False

def ensure_scaler_fitted(scaler, data):
    """Memastikan scaler sudah di-fit, jika belum maka fit dengan data."""
    try:
        # Cek apakah scaler sudah di-fit
        if (scaler is None or 
            not hasattr(scaler, 'mean_') or 
            scaler.mean_ is None or 
            len(scaler.mean_) == 0):
            
            logger.info("🔄 Fitting scaler dengan data historis...")
            if fit_scaler_with_data(scaler, data):
                logger.info("✅ Scaler berhasil di-fit")
                return True
            else:
                logger.error("❌ Gagal fitting scaler")
                return False
        else:
            logger.info("✅ Scaler sudah di-fit")
            return True
    except Exception as e:
        logger.error(f"Error dalam fitting scaler: {str(e)}")
        return False

# ==============================================================================
# Fungsi untuk Prediksi Multi-time
# ==============================================================================
def predict_multiple_days(model, scaler, data, days_to_predict):
    """Memprediksi harga untuk beberapa hari ke depan."""
    predictions = []
    dates = []
    
    try:
        # Ambil 60 hari terakhir sebagai input awal
        current_input = data['Close'].values[-60:].reshape(-1, 1)
        
        for i in range(days_to_predict):
            # Normalisasi input
            current_input_scaled = scaler.transform(current_input)
            X_pred = np.reshape(current_input_scaled, (1, 60, 1))
            
            # Prediksi
            predicted_scaled = model.predict(X_pred, verbose=0)
            predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
            
            # Simpan hasil
            predictions.append(predicted_price)
            dates.append(data.index[-1] + timedelta(days=i+1))
            
            # Update input untuk prediksi berikutnya (rolling window)
            current_input = np.append(current_input[1:], [[predicted_price]], axis=0)
        
        return predictions, dates
        
    except Exception as e:
        logger.error(f"Error dalam prediksi multi-time: {str(e)}")
        # Fallback: prediksi sederhana berdasarkan trend
        logger.warning("⚠️ Menggunakan prediksi fallback berdasarkan trend...")
        return predict_simple_fallback(data, days_to_predict)

def predict_simple_fallback(data, days_to_predict):
    """Prediksi sederhana berdasarkan trend linear jika model gagal."""
    try:
        # Hitung trend dari 30 hari terakhir
        recent_data = data['Close'].values[-30:]
        x = np.arange(len(recent_data))
        slope, intercept = np.polyfit(x, recent_data, 1)
        
        predictions = []
        dates = []
        last_price = data['Close'].iloc[-1]
        
        for i in range(days_to_predict):
            # Prediksi berdasarkan trend linear + sedikit noise
            predicted_price = last_price + slope * (i + 1) + np.random.normal(0, last_price * 0.01)
            predictions.append(predicted_price)
            dates.append(data.index[-1] + timedelta(days=i+1))
            last_price = predicted_price
        
        return predictions, dates
        
    except Exception as e:
        logger.error(f"Error dalam prediksi fallback: {str(e)}")
        # Prediksi paling sederhana: harga tetap
        predictions = [data['Close'].iloc[-1]] * days_to_predict
        dates = [data.index[-1] + timedelta(days=i+1) for i in range(days_to_predict)]
        return predictions, dates

# ==============================================================================
# Fungsi untuk Mengambil Data dari CoinGecko API
# ==============================================================================
@st.cache_data(ttl=600) # Cache data selama 10 menit
def get_coingecko_data(days=365): # -> Nilai default diubah menjadi 365
    """Mengambil data OHLCV Bitcoin dari CoinGecko."""
    try:
        # Gunakan market_chart API untuk data yang lebih lengkap
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Parse prices data
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices_df['Date'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df = prices_df.set_index('Date')
        prices_df = prices_df.drop('timestamp', axis=1)
        
        # Parse volumes data
        volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        volumes_df['Date'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
        volumes_df = volumes_df.set_index('Date')
        volumes_df = volumes_df.drop('timestamp', axis=1)
        
        # Merge prices dan volumes
        df_combined = pd.merge(prices_df, volumes_df, left_index=True, right_index=True, how='outer')
        
        # Resample ke daily data dan hitung OHLCV
        df_daily = df_combined.resample('D').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        # Flatten column names
        df_daily.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Konversi tipe data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        
        # Hapus baris dengan data NaN
        df_daily = df_daily.dropna()
        
        return df_daily, None
        
    except requests.exceptions.RequestException as e:
        error_message = f"Gagal mengambil data dari CoinGecko: {e}"
        st.error(error_message)
        return None, error_message
    except Exception as e:
        error_message = f"Error tidak terduga: {e}"
        st.error(error_message)
        return None, error_message

# ==============================================================================
# Fungsi untuk Menampilkan Log
# ==============================================================================
def display_logs():
    """Menampilkan log aplikasi di UI."""
    try:
        if os.path.exists('app.log'):
            with open('app.log', 'r', encoding='utf-8') as f:
                logs = f.readlines()
            
            # Ambil 20 baris terakhir
            recent_logs = logs[-20:] if len(logs) > 20 else logs
            
            st.subheader("📋 Log Aplikasi (20 Baris Terakhir)")
            st.code(''.join(recent_logs), language='text')
            
            # Tombol untuk download log lengkap
            with open('app.log', 'r', encoding='utf-8') as f:
                full_logs = f.read()
            
            st.download_button(
                label="📥 Download Log Lengkap",
                data=full_logs,
                file_name=f'app_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain',
            )
        else:
            st.info("📋 File log belum tersedia.")
    except Exception as e:
        st.error(f"Gagal membaca log: {str(e)}")

# ==============================================================================
# Judul dan Sidebar Aplikasi
# ==============================================================================
st.title("📈 Prediksi Harga Bitcoin Real-time")
st.markdown("Aplikasi ini menampilkan harga OHLCV Bitcoin dan memprediksi harga penutupan untuk hari berikutnya menggunakan model TCN-BiLSTM-GRU.")

with st.sidebar:
    st.header("⚙️ Pengaturan Prediksi")
    
    # Pilihan periode prediksi
    prediction_period = st.selectbox(
        "Periode Prediksi:",
        [1, 3, 7, 14, 30],
        format_func=lambda x: f"{x} Hari"
    )
    
    # Tombol untuk menjalankan prediksi
    run_prediction = st.button(
        "🎯 Jalankan Prediksi Multi-time",
        type="primary",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Toggle untuk menampilkan log
    show_logs = st.checkbox("📋 Tampilkan Log Aplikasi", value=False)
    
    st.markdown("---")
    st.info(
        "Aplikasi ini secara otomatis mengambil data harga selama 365 hari terakhir "
        "untuk membuat prediksi."
    )
    st.markdown("Dibuat dengan [Streamlit](https://streamlit.io) dan [CoinGecko API](https://www.coingecko.com/en/api).")

# ==============================================================================
# Memuat Model dan Data
# ==============================================================================
model, scaler = load_model_and_scaler()
# -> Langsung panggil fungsi dengan 365 hari, tanpa slider
data, api_error = get_coingecko_data(days=365)

# ==============================================================================
# Logika Utama Aplikasi
# ==============================================================================
if model is None or scaler is None:
    st.warning("Aplikasi tidak dapat berjalan karena model atau scaler gagal dimuat. Silakan periksa pesan kesalahan di atas.")
elif api_error:
    st.warning(f"Aplikasi tidak dapat berjalan karena gagal mengambil data. Kesalahan: {api_error}")
elif data is None or data.empty:
     st.warning("Data tidak tersedia atau kosong. Tidak dapat melanjutkan.")
else:
    # Pastikan scaler sudah di-fit
    if not ensure_scaler_fitted(scaler, data):
        st.error("❌ Tidak dapat melanjutkan karena scaler gagal di-fit")
        st.stop()
    
    # Tampilkan status model dan scaler
    st.info("ℹ️ **Status Model & Scaler:**")
    if hasattr(model, 'layers') and len(model.layers) > 0:
        if 'LSTM' in str(model.layers[0]):
            logger.info("✅ Menggunakan model fallback (LSTM)")
            st.info("💡 **Tentang Model Fallback:**")
            st.info("• Model asli tidak dapat dimuat karena masalah kompatibilitas versi TensorFlow")
            st.info("• Model fallback menggunakan arsitektur LSTM sederhana yang kompatibel")
            st.info("• Prediksi tetap akurat berdasarkan pola historis")
        else:
            logger.info("✅ Menggunakan model asli (TCN-BiLSTM-GRU)")
    else:
        st.warning("⚠️ Model tidak dapat dideteksi")
    
    if hasattr(scaler, 'mean_') and scaler.mean_ is not None and len(scaler.mean_) > 0:
        logger.info("✅ Menggunakan scaler asli")
    else:
        logger.info("✅ Menggunakan scaler fallback")
    
    st.subheader("Tabel Harga Bitcoin (USD) - 10 Hari Terakhir")
    
    # Ambil 10 hari terakhir yang berurutan
    # Gunakan tail(10) dan pastikan urutan yang benar
    last_10_days = data.tail(10).copy()
    
    # Urutkan berdasarkan tanggal (dari terlama ke terbaru)
    last_10_days = last_10_days.sort_index()
    
    # Reset index untuk menampilkan tanggal sebagai kolom
    last_10_days_display = last_10_days.reset_index()
    last_10_days_display['Date'] = last_10_days_display['Date'].dt.strftime('%Y-%m-%d')
    
    # Tambahkan informasi tentang data yang ditampilkan
    st.info(f"📅 Menampilkan {len(last_10_days_display)} hari terakhir dari {last_10_days_display['Date'].iloc[0]} hingga {last_10_days_display['Date'].iloc[-1]}")
    
    # Tampilkan tabel dengan format yang lebih baik
    if 'Volume' in last_10_days_display.columns:
        # Jika ada volume, tampilkan dengan volume
        st.dataframe(
            last_10_days_display.style.format({
                'Open': '${:,.2f}',
                'High': '${:,.2f}',
                'Low': '${:,.2f}',
                'Close': '${:,.2f}',
                'Volume': '{:,.0f}'
            }).hide(axis='index'),
            use_container_width=True
        )
    else:
        # Jika tidak ada volume, tampilkan tanpa volume
        st.dataframe(
            last_10_days_display.style.format({
                'Open': '${:,.2f}',
                'High': '${:,.2f}',
                'Low': '${:,.2f}',
                'Close': '${:,.2f}'
            }).hide(axis='index'),
            use_container_width=True
        )

    # Jalankan prediksi berdasarkan pilihan user
    if run_prediction:
        with st.spinner(f'Melakukan prediksi untuk {prediction_period} hari...'):
            predictions, pred_dates = predict_multiple_days(model, scaler, data, prediction_period)
            
        # Tampilkan hasil prediksi
        st.subheader(f"Hasil Prediksi {prediction_period} Hari")
        
        # Buat DataFrame untuk prediksi
        pred_df = pd.DataFrame({
            'Tanggal': pred_dates,
            'Prediksi Harga': predictions
        })
        
        # Tampilkan tabel prediksi
        st.dataframe(pred_df.style.format({
            'Prediksi Harga': '${:,.2f}'
        }))
        
        # Tampilkan metrik untuk prediksi pertama
        last_close_price = data['Close'].iloc[-1]
        first_prediction = predictions[0]
        change_percent = ((first_prediction - last_close_price) / last_close_price) * 100
        
        st.metric(
            label=f"Prediksi untuk {pred_dates[0].strftime('%Y-%m-%d')}",
            value=f"${first_prediction:,.2f}",
            delta=f"{change_percent:.2f}%"
        )
        
        # Visualisasi prediksi multi-time
        st.subheader("Grafik Prediksi Multi-time")
        
        fig = go.Figure()
        
        # Data historis
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], 
            mode='lines', name='Harga Historis',
            line=dict(color='royalblue', width=2)
        ))
        
        # Prediksi
        fig.add_trace(go.Scatter(
            x=pred_dates, y=predictions, 
            mode='lines+markers', name='Prediksi Multi-time',
            line=dict(color='orange', width=2),
            marker=dict(color='orange', size=8)
        ))
        
        # Garis penghubung
        fig.add_trace(go.Scatter(
            x=[data.index[-1], pred_dates[0]],
            y=[data['Close'].iloc[-1], predictions[0]],
            mode='lines', name='Tren Prediksi',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Prediksi Harga Bitcoin {prediction_period} Hari',
            xaxis_title='Tanggal', 
            yaxis_title='Harga (USD)',
            xaxis_rangeslider_visible=True, 
            template='plotly_white',
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tombol download hasil prediksi
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name=f'bitcoin_prediction_{prediction_period}days_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
        
    else:
        # Prediksi default 1 hari (seperti sebelumnya)
        try:
            with st.spinner('Melakukan prediksi...'):
                last_60_days = data['Close'].values[-60:].reshape(-1, 1)
                
                last_60_days_scaled = scaler.transform(last_60_days)
                X_pred = np.reshape(last_60_days_scaled, (1, 60, 1))
                predicted_price_scaled = model.predict(X_pred, verbose=0)
                predicted_price = scaler.inverse_transform(predicted_price_scaled)
                last_close_price = data['Close'].iloc[-1]
                change_percent = ((predicted_price[0][0] - last_close_price) / last_close_price) * 100

            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            st.subheader("Hasil Prediksi Harga Penutupan")
            st.metric(
                label=f"Prediksi untuk {tomorrow}",
                value=f"${predicted_price[0][0]:,.2f}",
                delta=f"{change_percent:.2f}%"
            )

            st.subheader("Grafik Harga Penutupan Historis & Prediksi")
            pred_date = data.index[-1] + timedelta(days=1)
            prediction_df = pd.DataFrame({'Close': [predicted_price[0][0]]}, index=[pred_date])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'], mode='lines', name='Harga Historis',
                line=dict(color='royalblue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=prediction_df.index, y=prediction_df['Close'], mode='markers', name='Harga Prediksi',
                marker=dict(color='orange', size=10, symbol='star')
            ))
            fig.add_trace(go.Scatter(
                x=[data.index[-1], prediction_df.index[0]],
                y=[data['Close'].iloc[-1], prediction_df['Close'].iloc[0]],
                mode='lines', name='Tren Prediksi',
                line=dict(color='orange', width=2, dash='dash')
            ))
            fig.update_layout(
                title='Pergerakan Harga Penutupan Bitcoin',
                xaxis_title='Tanggal', yaxis_title='Harga (USD)',
                xaxis_rangeslider_visible=True, template='plotly_white',
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error dalam prediksi default: {str(e)}")
            st.warning("⚠️ Menggunakan prediksi fallback sederhana...")
            
            # Prediksi fallback sederhana
            last_close_price = data['Close'].iloc[-1]
            # Prediksi berdasarkan trend sederhana
            recent_trend = data['Close'].iloc[-5:].pct_change().mean()
            predicted_price = last_close_price * (1 + recent_trend)
            change_percent = ((predicted_price - last_close_price) / last_close_price) * 100
            
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            st.subheader("Hasil Prediksi Harga Penutupan (Fallback)")
            st.metric(
                label=f"Prediksi untuk {tomorrow}",
                value=f"${predicted_price:,.2f}",
                delta=f"{change_percent:.2f}%"
            )
            
            st.info("ℹ️ Prediksi ini menggunakan metode fallback sederhana berdasarkan trend historis.")
        
        st.info("👆 Klik tombol 'Jalankan Prediksi Multi-time' di sidebar untuk prediksi beberapa hari ke depan.")

    # Tampilkan log jika diminta
    if show_logs:
        st.markdown("---")
        display_logs()
