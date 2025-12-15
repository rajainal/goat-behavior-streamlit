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

    /* Center card styling */
    .home-card {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 15px;
        padding: 3rem 2rem;
        max-width: 500px;
        margin: 2rem auto;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .judul-skripsi {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 2rem;
    }

    .info-text {
        font-size: 1.1rem;
        color: #333;
        margin: 0.5rem 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2d2d2d;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# MAIN CONTENT
# ========================================

# Logo centered above card
logo_path = Path("assets/logo_usu.png")
if logo_path.exists():
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        st.image(str(logo_path), width=150)

# Create centered card
st.markdown("""
<div class="home-card">
    <div class="judul-skripsi">
        Implementasi IoT dan CNN-GRU<br>
        Dalam Monitoring Perilaku Kambing Untuk Mendukung Precision Livestock Farming
    </div>

    <div class="info-text">S1 Ilmu Komputer</div>
    <div class="info-text">M. Raja Inal Lubis</div>
    <div class="info-text">211401134</div>
</div>
""", unsafe_allow_html=True)
