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
    /* Hide Streamlit default UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Logo center */
    .logo-center {
        display: flex;
        justify-content: center;
        margin-top: 40px;
        margin-bottom: 20px;
    }

    /* Card utama */
    .home-card {
        background-color: white;
        border-radius: 15px;
        padding: 3rem 2.5rem;
        max-width: 520px;
        margin: 0 auto;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }

    /* Judul skripsi */
    .judul-skripsi {
        font-size: 1.4rem;
        font-weight: 700;
        color: #222;
        line-height: 1.6;
        margin-bottom: 2rem;
    }

    /* Informasi mahasiswa */
    .info-text {
        font-size: 1.05rem;
        color: #333;
        margin: 0.4rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2d2d2d;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# MAIN CONTENT
# ========================================

# Logo (CENTER)
logo_path = Path("assets/logo_usu.png")
if logo_path.exists():
    st.markdown('<div class="logo-center">', unsafe_allow_html=True)
    st.image(str(logo_path), width=150)
    st.markdown('</div>', unsafe_allow_html=True)

# Card utama
st.markdown("""
<div class="home-card">
    <div class="judul-skripsi">
        Implementasi IoT dan CNN-GRU<br>
        Dalam Monitoring Perilaku Kambing<br>
        Untuk Mendukung Precision Livestock Farming
    </div>

    <div class="info-text">S1 Ilmu Komputer</div>
    <div class="info-text">M. Raja Inal Lubis</div>
    <div class="info-text">211401134</div>
</div>
""", unsafe_allow_html=True)
