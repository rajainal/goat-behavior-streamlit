"""
Goat Behavior Monitoring System
Halaman History - Riwayat Data
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# ========================================
# PAGE CONFIG
# ========================================

st.set_page_config(
    page_title="History - Goat Behavior",
    page_icon="📜",
    layout="wide"
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
# SIDEBAR - Only Refresh Button
# ========================================

with st.sidebar:
    st.header("⚙️ Kontrol")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# ========================================
# MAIN PAGE
# ========================================

st.title("📜 Riwayat Data")
st.markdown("---")

# ========================================
# GET DATA FROM SESSION STATE
# ========================================

if 'behavior_history' in st.session_state and len(st.session_state.behavior_history) > 0:
    df = pd.DataFrame(list(st.session_state.behavior_history))

    # Handle timestamp column
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['Waktu'] = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
    elif 'time' in df.columns:
        df['Waktu'] = df['time']
    else:
        df['Waktu'] = 'N/A'

    # Prepare display dataframe
    display_cols = ['Waktu']
    rename_map = {'behavior': 'Perilaku', 'confidence': 'Confidence (%)', 'subject': 'Subject'}

    for col, new_name in rename_map.items():
        if col in df.columns:
            display_cols.append(col)

    # Add sensor columns if available
    sensor_cols = ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']
    for col in sensor_cols:
        if col in df.columns:
            display_cols.append(col)

    df_display = df[display_cols].copy()

    # Rename columns
    col_names = {'behavior': 'Perilaku', 'confidence': 'Confidence (%)', 'subject': 'Subject',
                 'accX': 'Acc X', 'accY': 'Acc Y', 'accZ': 'Acc Z',
                 'gyrX': 'Gyr X', 'gyrY': 'Gyr Y', 'gyrZ': 'Gyr Z'}
    df_display = df_display.rename(columns=col_names)

    # Round numeric columns
    numeric_cols = df_display.select_dtypes(include=['float64', 'float32']).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(3)

    # ========================================
    # DATA TABLE
    # ========================================

    st.subheader(f"📋 Tabel Data ({len(df_display)} records)")

    # Show table (most recent first)
    st.dataframe(df_display.iloc[::-1], use_container_width=True, height=500)

    # ========================================
    # DOWNLOAD CSV
    # ========================================

    st.markdown("---")

    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"behavior_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("📭 Belum ada data riwayat. Mulai monitoring untuk mengumpulkan data.")