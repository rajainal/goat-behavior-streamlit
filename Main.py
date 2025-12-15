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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========================================
# MAIN CONTENT
# ========================================

# Center content using columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Logo centered
    logo_path = Path("assets/logo_usu.png")
    if logo_path.exists():
        subcol1, subcol2, subcol3 = st.columns([1, 1, 1])
        with subcol2:
            st.image(str(logo_path), width=150)

    st.markdown("<br>", unsafe_allow_html=True)

    # Title
    st.markdown("""
    <h3 style="text-align: center; color: #333;">
        Implementasi IoT dan CNN-GRU Dalam Monitoring Perilaku Kambing
        Untuk Mendukung Precision Livestock Farming
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Info
    st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">S1 Ilmu Komputer</p>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">M. Raja Inal Lubis</p>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">211401134</p>
    </div>
    """, unsafe_allow_html=True)
