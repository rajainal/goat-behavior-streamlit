"""
Goat Behavior Monitoring System
Halaman Utama - Informasi Skripsi
"""

import streamlit as st
from pathlib import Path

# ========================================
# PAGE CONFIG
# ========================================

st.set_page_config(
    page_title="Goat Behavior Monitoring",
    page_icon="🐐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS
# ========================================

st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Center container */
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }

    /* Card styling */
    .home-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        max-width: 550px;
        margin: 1rem auto;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    .judul-skripsi {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    .info-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1.5rem;
    }

    .info-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.2rem;
    }

    .info-text {
        font-size: 1.1rem;
        color: #333;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
        margin: 1rem 0;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# MAIN CONTENT
# ========================================

# Logo and Title Section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Check if logo exists
    logo_path = Path("assets/logo_usu.png")
    if logo_path.exists():
        st.image(str(logo_path), width=150, use_container_width=False)
    else:
        # Placeholder if logo not found
        st.markdown("""
        <div style="width: 150px; height: 150px; border: 2px dashed #ccc; border-radius: 50%;
                    margin: 0 auto; display: flex; align-items: center; justify-content: center;
                    color: #999; font-size: 0.8rem; text-align: center;">
            Logo USU<br>(simpan di assets/logo_usu.png)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-card">
        <div class="judul-skripsi">
            Implementasi IoT dan CNN-GRU Dalam Monitoring Perilaku Kambing
            Untuk Mendukung Precision Livestock Farming
        </div>

        <div class="divider"></div>

        <div class="info-section">
            <div class="info-label">Program Studi</div>
            <div class="info-text">S1 Ilmu Komputer</div>

            <div class="info-label">Nama</div>
            <div class="info-text">M. Raja Inal Lubis</div>

            <div class="info-label">NIM</div>
            <div class="info-text" style="margin-bottom: 0;">211401134</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================================
# FEATURE SECTION
# ========================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🎯 Fitur Aplikasi")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Real-time Monitoring</div>
        <div class="feature-desc">Pantau perilaku kambing secara real-time dengan prediksi CNN-GRU</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🐐</div>
        <div class="feature-title">Deteksi 3 Perilaku</div>
        <div class="feature-desc">Aktif, Berbaring, dan Makan dengan tingkat akurasi tinggi</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📜</div>
        <div class="feature-title">Riwayat Data</div>
        <div class="feature-desc">Simpan dan unduh data prediksi dalam format CSV</div>
    </div>
    """, unsafe_allow_html=True)

# ========================================
# QUICK START
# ========================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🚀 Mulai Sekarang")

col1, col2 = st.columns(2)

with col1:
    st.info("👈 Pilih **Monitoring** di sidebar untuk memulai pemantauan real-time")

with col2:
    st.info("👈 Pilih **History** di sidebar untuk melihat riwayat data")

# ========================================
# SIDEBAR
# ========================================

with st.sidebar:
    st.markdown("### 📌 Navigasi")
    st.markdown("""
    - **Main** - Halaman utama
    - **Monitoring** - Real-time monitoring
    - **History** - Riwayat data
    """)

    st.markdown("---")
    st.markdown("### ℹ️ Tentang")
    st.caption("Sistem monitoring perilaku kambing menggunakan sensor IMU dan model CNN-GRU untuk mendukung precision livestock farming.")
