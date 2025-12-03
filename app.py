import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from PIL import Image, ImageOps
import numpy as np
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLING
# ==========================================
st.set_page_config(
    page_title="UAS Komputasi Paralel",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan lebih cantik
st.markdown("""
    <style>
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card Styling */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .info-card h4 {
        color: #2d3436;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .info-card ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .info-card li {
        color: #4a4a4a;
        font-size: 1rem;
        line-height: 1.8;
        margin-bottom: 0.5rem;
    }
    
    .info-card li b {
        color: #667eea;
        font-weight: 600;
    }
    
    .info-card p {
        color: #4a4a4a;
        font-size: 1rem;
        line-height: 1.7;
    }
    
    /* Metric Styling */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #2d3436;
        padding: 0.75rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        background-color: #636e72;
        color: #dfe6e9;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #74b9ff;
        color: white;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Upload Area */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Image Container */
    .image-container {
        border: 3px solid #e0e0e0;
        border-radius: 15px;
        padding: 1rem;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header Cantik
st.markdown("""
    <div class="main-header">
        <h1>🚀 Parallel Deep Learning Training System</h1>
        <p><b>Distributed Image Classification</b> | Komputasi Paralel dan Terdistribusi</p>
        <p>📡 TensorFlow MirroredStrategy • 🤖 MobileNetV2 Architecture • 🎯 Rock-Paper-Scissors Classifier</p>
        <p style='margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px;'>
            ☁️ <b>Trained on Google Colab</b> with <b>NVIDIA Tesla T4 GPU</b> Accelerator
        </p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI MEMBANGUN MODEL & LOAD WEIGHTS
# ==========================================
@st.cache_resource
def get_model():
    """Membangun arsitektur model dan memuat weights"""
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.4), 
        Dense(3, activation='softmax')
    ])
    
    weights_path = 'distributed_rps_model.weights.h5'
    
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            return model, True
        except Exception as e:
            return None, str(e)
    else:
        return None, "File weights tidak ditemukan"

# Load Model
model, load_status = get_model()

# Status Model dengan styling
col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
with col_status2:
    if isinstance(load_status, bool) and load_status:
        st.success("✅ **Model Loaded Successfully!** Siap untuk melakukan prediksi.")
    else:
        st.error(f"⚠️ **Model Error:** {load_status}")
        st.info("💡 Pastikan file `distributed_rps_model.weights.h5` ada di direktori yang sama.")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI PREDIKSI GAMBAR
# ==========================================
def import_and_predict(image_data, model):
    """Preprocessing dan prediksi gambar"""
    size = (224, 224)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    
    # Handle different image formats
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
        
    # Normalization
    img_reshape = img[np.newaxis, ...] / 255.0
    prediction = model.predict(img_reshape, verbose=0)
    
    return prediction

# ==========================================
# 4. TAMPILAN USER INTERFACE (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎯 Demo Prediksi", "📊 Analisis Performansi", "ℹ️ Info Sistem"])

# ============ TAB 1: DEMO ============
with tab1:
    st.markdown("### 🎯 Klasifikasi Citra Real-Time")
    st.markdown("Ambil foto atau upload gambar tangan dengan pose **Batu**, **Gunting**, atau **Kertas**.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Opsi Input: Upload atau Kamera
    input_option = st.radio("Pilih Metode Input:", ("📤 Upload Gambar", "📸 Ambil Foto (Kamera)"), horizontal=True)
    
    image = None
    
    if input_option == "📤 Upload Gambar":
        # Upload Section
        col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
        with col_upload2:
            file = st.file_uploader(
                "Pilih File Gambar",
                type=["jpg", "png", "jpeg"],
                help="Format yang didukung: JPG, PNG, JPEG",
                label_visibility="collapsed"
            )
        if file is not None:
            image = Image.open(file)
            
    elif input_option == "📸 Ambil Foto (Kamera)":
        col_cam1, col_cam2, col_cam3 = st.columns([1, 2, 1])
        with col_cam2:
            camera_file = st.camera_input("Ambil Foto Tangan Anda")
        if camera_file is not None:
            image = Image.open(camera_file)
    
    # JIKA ADA GAMBAR (DARI UPLOAD ATAU KAMERA), PROSES DILANJUTKAN
    if image is not None:
        try:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Layout dengan proporsi lebih baik
            col1, col2 = st.columns([1.2, 1.5], gap="large")
            
            with col1:
                st.markdown("""
                    <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                                box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                        <h4 style='color: #2d3436; margin-bottom: 1rem; text-align: center;'>
                            📷 Input Gambar
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Container untuk gambar dengan border
                st.markdown("""
                    <style>
                    .image-frame {
                        border: 3px solid #667eea;
                        border-radius: 12px;
                        padding: 0.5rem;
                        background: white;
                        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                st.image(image, use_container_width=True, caption="Gambar Input")
            
            with col2:
                st.markdown("""
                    <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                                box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                        <h4 style='color: #2d3436; margin-bottom: 1rem; text-align: center;'>
                            🤖 Hasil Prediksi
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                if model:
                    if st.button("🔍 **Analisis Gambar**", type="primary", use_container_width=True):
                        with st.spinner('🧠 Neural Network sedang memproses...'):
                            try:
                                predictions = import_and_predict(image, model)
                                
                                class_names = ['📄 Paper (Kertas)', '🪨 Rock (Batu)', '✂️ Scissors (Gunting)']
                                class_emojis = ['📄', '🪨', '✂️']
                                class_names_simple = ['Paper', 'Rock', 'Scissors']
                                
                                idx = np.argmax(predictions)
                                confidence = np.max(predictions) * 100
                                
                                # Gradient colors untuk setiap kelas
                                gradient_colors = [
                                    'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',  # Paper - Purple
                                    'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',  # Rock - Pink-Red
                                    'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'   # Scissors - Blue
                                ]
                                
                                # Display Result dengan styling
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 2.5rem 1.5rem; 
                                                background: {gradient_colors[idx]}; 
                                                border-radius: 15px; margin: 1rem 0;
                                                box-shadow: 0 8px 20px rgba(0,0,0,0.15);'>
                                        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>{class_emojis[idx]}</div>
                                        <h2 style='color: white; margin: 0; font-size: 1.8rem;'>{class_names_simple[idx]}</h2>
                                        <p style='color: #f0f0f0; font-size: 1.1rem; margin-top: 0.5rem;'>
                                            Confidence: <b>{confidence:.1f}%</b>
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Progress Bar dengan styling
                                st.markdown(f"""
                                    <div style='background: #e0e0e0; border-radius: 10px; height: 25px; overflow: hidden; margin: 1rem 0;'>
                                        <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                                    height: 100%; width: {confidence}%; 
                                                    display: flex; align-items: center; justify-content: center;
                                                    transition: width 0.5s ease;'>
                                            <span style='color: white; font-weight: 600; font-size: 0.9rem;'>{confidence:.1f}%</span>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Detail Probabilitas dengan design lebih baik
                                st.markdown("""
                                    <h5 style='color: #2d3436; margin-bottom: 1rem;'>
                                        📊 Detail Probabilitas Semua Kelas
                                    </h5>
                                """, unsafe_allow_html=True)
                                
                                for i, (name, emoji, simple) in enumerate(zip(class_names, class_emojis, class_names_simple)):
                                    prob = predictions[0][i] * 100
                                    
                                    # Color based on probability
                                    if prob > 50:
                                        color = "#28a745"
                                        bg_color = "#d4edda"
                                    elif prob > 20:
                                        color = "#ffc107"
                                        bg_color = "#fff3cd"
                                    else:
                                        color = "#6c757d"
                                        bg_color = "#f8f9fa"
                                    
                                    st.markdown(f"""
                                        <div style='margin: 0.7rem 0; padding: 1rem; 
                                                    background: {bg_color}; 
                                                    border-radius: 10px; 
                                                    border-left: 5px solid {color};
                                                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                                    display: flex; 
                                                    justify-content: space-between;
                                                    align-items: center;'>
                                            <div style='display: flex; align-items: center;'>
                                                <span style='font-size: 1.5rem; margin-right: 0.75rem;'>{emoji}</span>
                                                <b style='color: #2d3436;'>{simple}</b>
                                            </div>
                                            <span style='color: {color}; font-weight: 700; font-size: 1.1rem;'>{prob:.2f}%</span>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"❌ Error saat prediksi: {e}")
                else:
                    st.warning("⚠️ Model belum dimuat. Tidak dapat melakukan prediksi.")
                    
        except Exception as e:
            st.error(f"❌ File gambar tidak valid atau rusak. Error: {e}")

# ============ TAB 2: ANALISIS (PERBAIKAN ISI SESUAI GAMBAR) ============
with tab2:
    # --- CSS KHUSUS UNTUK MEMPERJELAS METRIC ---
    st.markdown("""
    <style>
    /* Styling khusus untuk box metric agar tidak samar */
    [data-testid="stMetric"] {
        background-color: #f8f9fa; /* Latar belakang putih abu terang */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333333; /* Warna teks utama gelap */
    }
    
    /* Label Metric (Judul kecil di atas angka) */
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #555555 !important; /* Abu tua jelas */
    }
    
    /* Value Metric (Angka besar) */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important; /* Biru gelap */
    }

    /* Delta Metric (Tulisan kecil di bawah angka) */
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #333333 !important;
    }
    
    /* Warna khusus untuk Delta Positif/Negatif */
    [data-testid="stMetricDelta"] svg {
        fill: currentColor;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Laporan Analisis Komputasi Paralel")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Grafik Speedup
    if os.path.exists('grafik_speedup.png'):
        st.markdown("#### 📈 Perbandingan Performa Training (Single GPU)")
        st.image("grafik_speedup.png", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Penjelasan Grafik (2 Kolom)
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.info("""
            **📊 Grafik Kiri: Akurasi Model**
            
            * **Training Accuracy (Biru):** Sangat stabil mencapai **100%**.
            * **Validation Accuracy (Oranye):** Mencapai **~99%**.
            * **Hasil:** Model sangat akurat (High Performance) berkat transfer learning MobileNetV2.
            """)
        
        with col_g2:
            st.warning("""
            **⚡ Grafik Kanan: Waktu Eksekusi (5 Epoch)**
            
            * **Single Node (Baseline):** 99.19 detik
            * **Distributed (Single GPU):** 121.98 detik
            * **Speedup Ratio:** 0.81x (Negative Speedup)
            """)
        
        st.divider()
        
        # --- ANALISIS MENDALAM ---
        st.subheader("🔍 Analisis Ilmiah: Fenomena Negative Speedup")
        
        st.markdown("""
        Hasil eksperimen menunjukkan **Distributed Training lebih lambat (0.81x)** dibandingkan Single Node. 
        Ini adalah hasil yang **valid secara ilmiah** karena kondisi hardware yang digunakan.
        """)

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.error("⚠️ **1. Limitasi Hardware (Single GPU)**")
            st.markdown("""
            Google Colab Free hanya menyediakan **1 unit GPU Tesla T4**.
            * **Distributed Requirement:** Strategi `MirroredStrategy` dirancang untuk membagi beban ke **minimal 2 device**.
            * **Efek:** Tidak terjadi pembagian tugas paralel secara fisik karena hanya ada 1 worker.
            """)

        with col_a2:
            st.warning("📉 **2. Overhead Strategi (Administration Cost)**")
            st.markdown("""
            Ketika kode distributed dijalankan di 1 GPU:
            * Sistem tetap menjalankan protokol **sinkronisasi & replikasi variabel**.
            * Hal ini menambah beban **'Overhead'** tanpa memberikan keuntungan kecepatan komputasi.
            * Akibatnya: Waktu total = Waktu Komputasi + Waktu Admin (Overhead).
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- KESIMPULAN UTAMA (DENGAN WARNA KONTRAS) ---
        st.subheader("📝 Kesimpulan Utama Eksperimen")
        
        # KITA GUNAKAN CONTAINER BACKGROUND PUTIH AGAR KONTRAS
        with st.container():
            k1, k2, k3 = st.columns(3)
            
            # Custom HTML cards
            k1.markdown("""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3; text-align: center;">
                    <p style="margin:0; font-size: 0.9rem; color: #555;">Arsitektur Kode</p>
                    <h3 style="margin:0; color: #1565c0;">Valid</h3>
                    <p style="margin:0; font-size: 0.8rem; color: #333; font-weight: bold;">✅ Data Parallelism Ready</p>
                </div>
            """, unsafe_allow_html=True)

            k2.markdown("""
                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50; text-align: center;">
                    <p style="margin:0; font-size: 0.9rem; color: #555;">Akurasi Validasi</p>
                    <h3 style="margin:0; color: #2e7d32;">98.8%</h3>
                    <p style="margin:0; font-size: 0.8rem; color: #333; font-weight: bold;">📈 Model Cerdas</p>
                </div>
            """, unsafe_allow_html=True)

            k3.markdown("""
                <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800; text-align: center;">
                    <p style="margin:0; font-size: 0.9rem; color: #555;">Speedup Ratio</p>
                    <h3 style="margin:0; color: #e65100;">0.81x</h3>
                    <p style="margin:0; font-size: 0.8rem; color: #333; font-weight: bold;">⚠️ Hardware Limit (1 GPU)</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        * **Validasi Sistem:** Kode telah berhasil mengimplementasikan `tf.distribute.MirroredStrategy`.
        * **Rekomendasi:** Untuk mendapatkan *Positive Speedup* (Linear Scaling), sistem ini harus dijalankan pada environment dengan **Multi-GPU** atau **TPU**.
        """)

    else:
        st.warning("⚠️ File grafik `grafik_speedup.png` tidak ditemukan. Harap upload file grafik ke folder ini.")

# ============ TAB 3: INFO SISTEM ============
with tab3:
    st.markdown("### ℹ️ Informasi Sistem & Teknologi")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
            <div class="info-card" style='background: linear-gradient(135deg, #fff3e0 0%, #ffffff 100%); border-left: 5px solid #FF9800;'>
                <h4>🔧 Spesifikasi Teknis</h4>
                <ul>
                    <li><b>Framework:</b> TensorFlow 2.x</li>
                    <li><b>Model Base:</b> MobileNetV2 (Pre-trained)</li>
                    <li><b>Input Size:</b> 224 × 224 × 3 (RGB)</li>
                    <li><b>Output:</b> 3 Classes (Softmax)</li>
                    <li><b>Optimizer:</b> Adam</li>
                    <li><b>Hidden Layer:</b> 512 Dense Units + Dropout 0.4</li>
                    <li><b>Training Platform:</b> Google Colab</li>
                    <li><b>Accelerator:</b> NVIDIA Tesla T4 GPU (1 Unit)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
            <div class="info-card" style='background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%); border-left: 5px solid #4CAF50;'>
                <h4>🚀 Strategi Paralel</h4>
                <ul>
                    <li><b>Metode:</b> Data Parallelism</li>
                    <li><b>Strategy:</b> TensorFlow MirroredStrategy</li>
                    <li><b>Distribusi:</b> Synchronous All-Reduce</li>
                    <li><b>Worker:</b> Single Host, Multi-Device Ready</li>
                    <li><b>Sinkronisasi:</b> Gradient Averaging</li>
                    <li><b>Komunikasi:</b> NCCL (GPU) / GRPC (CPU)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card" style='background: linear-gradient(135deg, #f3e5f5 0%, #ffffff 100%); border-left: 5px solid #9C27B0;'>
            <h4>📚 Tentang Dataset</h4>
            <p><b>Rock Paper Scissors Dataset</b> - Dataset klasifikasi citra tangan dengan tiga kelas:</p>
            <ul>
                <li>🪨 <b>Rock (Batu):</b> Tangan mengepal</li>
                <li>📄 <b>Paper (Kertas):</b> Telapak tangan terbuka</li>
                <li>✂️ <b>Scissors (Gunting):</b> Jari telunjuk dan tengah membentuk gunting</li>
            </ul>
            <p>Dataset digunakan untuk melatih model deep learning dengan teknik transfer learning dari MobileNetV2.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        <p><b>UAS Komputasi Paralel dan Terdistribusi</b></p>
        <p>Dibuat dengan teman teman yakan menggunakan Streamlit & TensorFlow</p>
        <p style='font-size: 0.8rem; color: #999;'>© 2024 | Parallel Deep Learning System</p>
    </div>
""", unsafe_allow_html=True)
