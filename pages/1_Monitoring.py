"""
Goat Behavior Monitoring System
Halaman Monitoring - Real-time Prediction

Firebase Setup:
- Firebase A: ESP32 sensor data (serviceAccountKey.json)
- Firebase B: Prediction results (REST API)
"""

import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
from tensorflow import keras
import requests
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import deque
import pickle

# Timezone for Indonesia (WIB = UTC+7)
WIB = timezone(timedelta(hours=7))

def get_current_time_wib():
    """Get current time in WIB (Waktu Indonesia Barat)"""
    return datetime.now(WIB)

# ========================================
# FIREBASE B CONFIG (for storing predictions)
# ========================================

FIREBASE_B_URL = "https://skripsic3web-default-rtdb.asia-southeast1.firebasedatabase.app"

# Firebase B Service Account JSON file
# Download from: Firebase Console (skripsic3web) -> Project Settings -> Service accounts -> Generate new private key
# Ganti dengan nama file JSON Anda

# ========================================
# PAGE CONFIG
# ========================================

st.set_page_config(
    page_title="Monitoring - Goat Behavior",
    page_icon="📊",
    layout="wide"
)

# ========================================
# CUSTOM CSS
# ========================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .behavior-aktif { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .behavior-berbaring { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .behavior-makan { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }

    .stat-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 0.8rem;
    }
    .stat-card-aktif {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #38ef7d;
        margin-bottom: 0.8rem;
        color: black;
    }
    .stat-card-berbaring {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4facfe;
        margin-bottom: 0.8rem;
        color: black;
    }
    .stat-card-makan {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #fee140;
        margin-bottom: 0.8rem;
        color: black;
    }
    .detection-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        background-color: #f0f2f6;
        border-radius: 5px;
        font-size: 0.9rem;
        color: black;
    }
    .section-spacer {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .chart-spacer {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# INITIALIZE SESSION STATE
# ========================================

if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

if 'sensor_history' not in st.session_state:
    st.session_state.sensor_history = deque(maxlen=200)

if 'behavior_history' not in st.session_state:
    st.session_state.behavior_history = deque(maxlen=500)

if 'behavior_durations' not in st.session_state:
    st.session_state.behavior_durations = {'Aktif': 0, 'Berbaring': 0, 'Makan': 0}

if 'behavior_detections' not in st.session_state:
    st.session_state.behavior_detections = {'Aktif': [], 'Berbaring': [], 'Makan': []}

if 'last_behavior' not in st.session_state:
    st.session_state.last_behavior = None

if 'last_behavior_time' not in st.session_state:
    st.session_state.last_behavior_time = None

if 'last_data_key' not in st.session_state:
    st.session_state.last_data_key = None

if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = {'behavior': '-', 'confidence': 0, 'subject': '-', 'probs': [0, 0, 0]}

if 'firebase_b_error' not in st.session_state:
    st.session_state.firebase_b_error = None

# ========================================
# FIREBASE A INITIALIZATION (Sensor Data from ESP32)
# ========================================

@st.cache_resource
def init_firebase_a():
    """Initialize Firebase A for reading sensor data from ESP32"""
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(st.secrets["firebase_a"])
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://skripsic3-b62fc-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
            return True
        except Exception as e:
            st.error(f"Firebase A Error: {e}")
            return False
    return True

# ========================================
# FIREBASE B INITIALIZATION (Prediction Results)
# ========================================

@st.cache_resource
def init_firebase_b():
    """Initialize Firebase B for storing prediction results (separate project)"""
    try:
        # Check if Firebase B app already exists
        try:
            firebase_admin.get_app('firebase_b')
            return True
        except ValueError:
            # App doesn't exist, create it
            pass

        cred_b = credentials.Certificate(st.secrets["firebase_b"])
        firebase_admin.initialize_app(cred_b, {
            'databaseURL': FIREBASE_B_URL
        }, name='firebase_b')
        return True
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: Firebase b")
        return False
    except Exception as e:
        st.error(f"Firebase B Error: {e}")
        return False


def save_prediction_to_firebase_b(prediction_data):
    """
    Save prediction result to Firebase B using firebase-admin SDK.
    """
    try:
        # Get Firebase B app (will raise ValueError if not initialized)
        try:
            firebase_b_app = firebase_admin.get_app('firebase_b')
        except ValueError:
            return False, "Firebase B belum diinisialisasi"

        # Get database reference for Firebase B
        ref = db.reference('prediction_results', app=firebase_b_app)

        # Push new prediction data
        ref.push(prediction_data)

        return True, "Success"
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[Firebase B Error] {error_msg}")
        return False, error_msg

# ========================================
# LOAD MODEL & PREPROCESSING
# ========================================

@st.cache_resource
def load_model():
    try:
        return keras.models.load_model("models/model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_preprocessing():
    try:
        with open("preprocessing_objects.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preprocessing: {e}")
        return None

# ========================================
# PREDICTION FUNCTION
# ========================================

def predict_behavior(sensor_data, model, preprocess):
    scaler = preprocess['scaler']
    label_encoder = preprocess['label_encoder']
    window_size = preprocess.get('window_size', 20)
    n_features = len(preprocess.get('sensor_columns', ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']))

    data_normalized = scaler.transform(sensor_data)
    input_data = data_normalized.reshape(1, window_size, n_features).astype(np.float32)
    prediction = model.predict(input_data, verbose=0)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    behavior = label_encoder.classes_[predicted_class]

    return behavior, confidence, prediction[0]

# ========================================
# FIREBASE A DATA FUNCTIONS (Read Sensor Data)
# ========================================

def get_latest_data():
    """Get latest sensor data from Firebase A (ESP32)"""
    try:
        ref = db.reference('sensor_readings')
        data = ref.order_by_key().limit_to_last(1).get()
        if data:
            key = list(data.keys())[0]
            return data[key], key
        return None, None
    except Exception as e:
        return None, None

def parse_sensor_data(data):
    """Parse sensor data from Firebase A"""
    sensor_data = data.get('sensor_data', [])
    parsed_data = []

    if isinstance(sensor_data, dict):
        sorted_keys = sorted(sensor_data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
        sensor_data = [sensor_data[k] for k in sorted_keys]

    for row in sensor_data:
        if isinstance(row, list):
            parsed_row = [float(val) if isinstance(val, (int, float, str)) else 0.0 for val in row]
            parsed_data.append(parsed_row)
        elif isinstance(row, dict):
            if all(str(k).isdigit() for k in row.keys()):
                sorted_keys = sorted(row.keys(), key=lambda x: int(x))
                parsed_row = [float(row[k]) if isinstance(row[k], (int, float, str)) else 0.0 for k in sorted_keys]
            else:
                parsed_row = [
                    float(row.get('accX', 0)), float(row.get('accY', 0)), float(row.get('accZ', 0)),
                    float(row.get('gyrX', 0)), float(row.get('gyrY', 0)), float(row.get('gyrZ', 0))
                ]
            parsed_data.append(parsed_row)

    return np.array(parsed_data) if parsed_data else np.array([])

def get_subject_from_data(data):
    """Extract subject from Firebase A data"""
    metadata = data.get('metadata', {})
    if isinstance(metadata, dict) and 'subject' in metadata:
        return metadata.get('subject', 'Unknown')
    return data.get('subject', 'Unknown')

# ========================================
# TRACKING FUNCTIONS
# ========================================

def update_behavior_tracking(behavior, current_time):
    """Update duration and detection tracking"""
    if st.session_state.last_behavior != behavior:
        st.session_state.behavior_detections[behavior].append(current_time)

        if st.session_state.last_behavior and st.session_state.last_behavior_time:
            duration = (current_time - st.session_state.last_behavior_time).total_seconds()
            st.session_state.behavior_durations[st.session_state.last_behavior] += duration

        st.session_state.last_behavior = behavior
        st.session_state.last_behavior_time = current_time
    else:
        if st.session_state.last_behavior_time:
            duration = (current_time - st.session_state.last_behavior_time).total_seconds()
            st.session_state.behavior_durations[behavior] += duration
            st.session_state.last_behavior_time = current_time

# ========================================
# CHART FUNCTIONS
# ========================================

def create_sensor_chart(sensor_history):
    """Create sensor data chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Accelerometer', 'Gyroscope'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    colors_acc = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    colors_gyr = ['#96CEB4', '#FFEAA7', '#DDA0DD']

    if len(sensor_history) >= 2:
        df = pd.DataFrame(list(sensor_history))
        x_col = 'timestamp' if 'timestamp' in df.columns else 'time' if 'time' in df.columns else None

        if x_col:
            for i, col in enumerate(['accX', 'accY', 'accZ']):
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df[x_col], y=df[col], name=col, line=dict(color=colors_acc[i], width=2)), row=1, col=1)

            for i, col in enumerate(['gyrX', 'gyrY', 'gyrZ']):
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df[x_col], y=df[col], name=col, line=dict(color=colors_gyr[i], width=2)), row=2, col=1)
    else:
        # Add empty traces to show the chart structure
        for i, col in enumerate(['accX', 'accY', 'accZ']):
            fig.add_trace(go.Scatter(x=[], y=[], name=col, line=dict(color=colors_acc[i], width=2)), row=1, col=1)
        for i, col in enumerate(['gyrX', 'gyrY', 'gyrZ']):
            fig.add_trace(go.Scatter(x=[], y=[], name=col, line=dict(color=colors_gyr[i], width=2)), row=2, col=1)

    fig.update_layout(
        height=600,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (g)", row=1, col=1)
    fig.update_yaxes(title_text="Angular Velocity (°/s)", row=2, col=1)

    return fig

def create_behavior_timeline(behavior_history):
    """Create behavior timeline chart"""
    fig = go.Figure()

    if len(behavior_history) >= 2:
        df = pd.DataFrame(list(behavior_history))
        x_col = 'timestamp' if 'timestamp' in df.columns else 'time' if 'time' in df.columns else None

        if x_col:
            color_map = {'Aktif': '#38ef7d', 'Berbaring': '#4facfe', 'Makan': '#fee140'}
            colors = [color_map.get(b, '#888888') for b in df['behavior']]

            fig.add_trace(go.Scatter(
                x=df[x_col], y=df['confidence'], mode='lines+markers',
                marker=dict(color=colors, size=10), line=dict(color='#888888', width=1),
                text=df['behavior'], hovertemplate='%{text}<br>Confidence: %{y:.1f}%<br>Time: %{x}'
            ))

    fig.update_layout(title="Timeline Perilaku", xaxis_title="Waktu", yaxis_title="Confidence (%)", yaxis_range=[0, 105], height=350, margin=dict(l=0, r=0, t=50, b=0))
    return fig

# ========================================
# MAIN PAGE
# ========================================

st.title("📊 Real-time Monitoring")
st.markdown("---")

# Load resources
firebase_a_ok = init_firebase_a()
firebase_b_ok = init_firebase_b()
model = load_model()
preprocess = load_preprocessing()

# ========================================
# SIDEBAR
# ========================================

with st.sidebar:
    st.header("⚙️ Kontrol Monitoring")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.monitoring = True
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.monitoring = False

    if st.session_state.monitoring:
        st.success("🟢 Monitoring Aktif")
    else:
        st.warning("🔴 Monitoring Berhenti")

    st.markdown("---")

    if st.button("🔄 Reset Data", use_container_width=True):
        st.session_state.sensor_history.clear()
        st.session_state.behavior_history.clear()
        st.session_state.behavior_durations = {'Aktif': 0, 'Berbaring': 0, 'Makan': 0}
        st.session_state.behavior_detections = {'Aktif': [], 'Berbaring': [], 'Makan': []}
        st.session_state.last_behavior = None
        st.session_state.last_behavior_time = None
        st.session_state.current_prediction = {'behavior': '-', 'confidence': 0, 'subject': '-', 'probs': [0, 0, 0]}
        st.rerun()

    st.markdown("---")
    st.header("🔗 Status Koneksi")

    if firebase_a_ok:
        st.success("✅ Firebase A (Sensor)")
    else:
        st.error("❌ Firebase A (Sensor)")

    # Show Firebase B status
    if firebase_b_ok:
        if st.session_state.firebase_b_error:
            st.warning(f"⚠️ Firebase B (Prediksi)")
            st.caption(f"Last error: {st.session_state.firebase_b_error}")
        else:
            st.success("✅ Firebase B (Prediksi)")
    else:
        st.error("❌ Firebase B (Prediksi)")
        st.caption(f"File: Firebase B")

    if model:
        st.success("✅ Model")
    else:
        st.error("❌ Model")

    if preprocess:
        st.success("✅ Preprocessing")
    else:
        st.error("❌ Preprocessing")

# Check resources
if not all([firebase_a_ok, firebase_b_ok, model, preprocess]):
    st.error("⚠️ Beberapa komponen belum siap. Periksa file credentials JSON, model.h5, dan preprocessing_objects.pkl")
    st.stop()

# ========================================
# MONITORING FRAGMENT
# ========================================

@st.fragment(run_every=timedelta(seconds=2) if st.session_state.monitoring else None)
def monitoring_fragment():
    emoji_map = {'Aktif': '🟢', 'Berbaring': '🔵', 'Makan': '🟡'}

    # Get current prediction data
    pred = st.session_state.current_prediction

    # If monitoring, fetch and process data
    if st.session_state.monitoring:
        current_time = get_current_time_wib()
        data, key = get_latest_data()

        if data and 'sensor_data' in data and key != st.session_state.last_data_key:
            st.session_state.last_data_key = key

            sensor_data = parse_sensor_data(data)
            subject = get_subject_from_data(data)
            last_reading = sensor_data[-1] if len(sensor_data) > 0 else [0]*6

            # Calculate mean sensor values for the window
            if len(sensor_data) > 0:
                mean_acc_x = float(np.mean(sensor_data[:, 0]))
                mean_acc_y = float(np.mean(sensor_data[:, 1]))
                mean_acc_z = float(np.mean(sensor_data[:, 2]))
                mean_gyr_x = float(np.mean(sensor_data[:, 3]))
                mean_gyr_y = float(np.mean(sensor_data[:, 4]))
                mean_gyr_z = float(np.mean(sensor_data[:, 5]))
            else:
                mean_acc_x = mean_acc_y = mean_acc_z = 0.0
                mean_gyr_x = mean_gyr_y = mean_gyr_z = 0.0

            st.session_state.sensor_history.append({
                'timestamp': current_time, 'time_str': current_time.strftime("%H:%M:%S"),
                'accX': float(last_reading[0]), 'accY': float(last_reading[1]), 'accZ': float(last_reading[2]),
                'gyrX': float(last_reading[3]), 'gyrY': float(last_reading[4]), 'gyrZ': float(last_reading[5])
            })

            behavior, confidence, probs = predict_behavior(sensor_data, model, preprocess)
            update_behavior_tracking(behavior, current_time)

            # ========================================
            # SAVE TO FIREBASE B (with sensor values)
            # ========================================
            prediction_data = {
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S WIB"),
                'subject': subject,
                'behavior': behavior,
                'confidence': float(round(confidence, 2)),
                'sensor_data': {
                    'accX': float(round(mean_acc_x, 4)),
                    'accY': float(round(mean_acc_y, 4)),
                    'accZ': float(round(mean_acc_z, 4)),
                    'gyrX': float(round(mean_gyr_x, 4)),
                    'gyrY': float(round(mean_gyr_y, 4)),
                    'gyrZ': float(round(mean_gyr_z, 4))
                }
            }
            success, message = save_prediction_to_firebase_b(prediction_data)
            if not success:
                st.session_state.firebase_b_error = message
            else:
                st.session_state.firebase_b_error = None

            st.session_state.behavior_history.append({
                'timestamp': current_time, 'time_str': current_time.strftime("%H:%M:%S"),
                'behavior': behavior, 'confidence': confidence, 'subject': subject,
                'accX': float(last_reading[0]), 'accY': float(last_reading[1]), 'accZ': float(last_reading[2]),
                'gyrX': float(last_reading[3]), 'gyrY': float(last_reading[4]), 'gyrZ': float(last_reading[5])
            })

            st.session_state.current_prediction = {'behavior': behavior, 'confidence': confidence, 'subject': subject, 'probs': probs}

    # ========================================
    # ALWAYS SHOW ALL COMPONENTS
    # ========================================

    # 1. Sensor Chart
    st.subheader("📈 Data Sensor Real-time")
    fig_sensor = create_sensor_chart(st.session_state.sensor_history)
    st.plotly_chart(fig_sensor, use_container_width=True, key="sensor_chart")

    # 2. Prediction Results
    st.subheader("🎯 Hasil Prediksi")

    pred = st.session_state.current_prediction
    col1, col2, col3 = st.columns(3)

    behavior_class = f"behavior-{pred['behavior'].lower()}" if pred['behavior'] != '-' else ""

    with col1:
        st.markdown(f"""
        <div class="metric-card {behavior_class}">
            <h3>{emoji_map.get(pred['behavior'], '⚪')} Behavior</h3>
            <h1>{pred['behavior']}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Confidence</h3>
            <h1>{pred['confidence']:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🐐 Subject</h3>
            <h1>{pred['subject']}</h1>
        </div>
        """, unsafe_allow_html=True)

    # 3. Probability bars
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown("#### Probabilitas per Kelas")
    labels = preprocess['label_encoder'].classes_
    prob_cols = st.columns(len(labels))

    for i, (label, prob) in enumerate(zip(labels, pred['probs'])):
        with prob_cols[i]:
            st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

    # 4. Statistics
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.subheader("📊 Statistik Sesi")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⏱️ Durasi per Perilaku")
        for beh, duration in st.session_state.behavior_durations.items():
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            time_str = f"{minutes} menit {seconds} detik" if minutes > 0 else f"{seconds} detik"
            emoji = emoji_map.get(beh, '⚪')
            card_class = f"stat-card-{beh.lower()}"
            st.markdown(f'<div class="{card_class}">{emoji} <strong>{beh}</strong>: {time_str}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🔢 Jumlah Deteksi")
        for beh, detections in st.session_state.behavior_detections.items():
            count = len(detections)
            emoji = emoji_map.get(beh, '⚪')
            with st.expander(f"{emoji} {beh}: {count} kali"):
                if detections:
                    recent = detections[-5:]
                    for det_time in reversed(recent):
                        st.markdown(f'<div class="detection-item">🕐 {det_time.strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
                    if len(detections) > 5:
                        st.caption(f"...dan {len(detections) - 5} deteksi lainnya")
                else:
                    st.caption("Belum ada deteksi")

    # 5. Behavior Timeline
    st.markdown('<div class="chart-spacer"></div>', unsafe_allow_html=True)
    fig_timeline = create_behavior_timeline(st.session_state.behavior_history)
    st.plotly_chart(fig_timeline, use_container_width=True, key="behavior_timeline")

    # Status message
    if not st.session_state.monitoring:
        st.info("👆 Klik **Start** di sidebar untuk memulai monitoring")

# Run fragment
monitoring_fragment()