import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os
import queue
import threading
from collections import deque

from pose_classifier import classify_pose, POSE_COLORS, SKELETON_CONNECTIONS

# ─────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────
INFER_SIZE = 320
FRAME_SKIP = 2
WINDOW_SEC = 5

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "pose_window" not in st.session_state:
    st.session_state.pose_window = deque()
if "last_display" not in st.session_state:
    st.session_state.last_display = {}

# ─────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Pose Tubuh Realtime",
    page_icon="🤸",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤸 Deteksi Pose Tubuh Realtime</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Proyek Akhir Jaringan Syaraf Tiruan &nbsp;|&nbsp; YOLOv8 Pose + Streamlit</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "yolov8n-pose.pt"),
        os.path.join(base_dir, "models", "yolov8n-pose.pt"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)

    m = YOLO(model_path) if model_path else YOLO("yolov8n-pose.pt")
    dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
    m(dummy, verbose=False)
    return m

model = load_model()

# ─────────────────────────────────────────────
# Mapping
# ─────────────────────────────────────────────
COLOR_MAP = {
    "Berdiri"          : "#00C853",
    "Duduk"            : "#2979FF",
    "Jongkok"          : "#FF6D00",
    "Berbaring"        : "#D500F9",
    "Tidak Terdeteksi" : "#616161",
}
EMOJI_MAP = {
    "Berdiri"          : "🟢",
    "Duduk"            : "🔵",
    "Jongkok"          : "🟠",
    "Berbaring"        : "🟣",
    "Tidak Terdeteksi" : "⚫",
}
ALL_POSES = ["Berdiri", "Duduk", "Jongkok", "Berbaring"]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.3, max_value=0.8,
        value=0.5, step=0.05,
    )
    show_skeleton  = st.checkbox("Tampilkan Skeleton",    value=True)
    show_keypoints = st.checkbox("Tampilkan Keypoint",    value=True)
    show_angles    = st.checkbox("Tampilkan Sudut Sendi", value=True)

    st.divider()
    st.subheader("🎨 Legenda Warna")
    for emoji, label in [("🟢","Berdiri"),("🔵","Duduk"),("🟠","Jongkok"),("🟣","Berbaring"),("⚫","Tidak Terdeteksi")]:
        st.write(f"{emoji} {label}")

    st.divider()
    st.subheader("ℹ️ Informasi Model")
    st.info("**Model:** YOLOv8n-pose\n\n**Keypoint:** 17 titik tubuh\n\n**Kelas:** 4 pose\n\n**Dataset:** COCO Keypoints")

    st.divider()
    if st.button("🔄 Reset Statistik", use_container_width=True):
        st.session_state.pose_window  = deque()
        st.session_state.last_display = {}
        st.rerun()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_window_stats():
    now    = time.time()
    window = st.session_state.pose_window
    while window and (now - window[0][0]) > WINDOW_SEC:
        window.popleft()
    counts = {p: 0 for p in ALL_POSES}
    for _, label in window:
        if label in counts:
            counts[label] += 1
    total = sum(counts.values()) or 1
    return {p: (counts[p] / total) * 100 for p in ALL_POSES}

def build_pose_html(label):
    color_hex = COLOR_MAP.get(label, "#616161")
    emoji     = EMOJI_MAP.get(label, "❓")
    return (
        f'<div style="background-color:{color_hex};padding:10px;'
        f'border-radius:10px;text-align:center;margin:0.4rem 0;">'
        f'<h2 style="color:white;margin:0;">{emoji} {label}</h2></div>'
    )

def build_metrics_html(knee, hip, n_persons, fps):
    knee_str = f"{knee:.1f}°" if knee is not None else "N/A"
    hip_str  = f"{hip:.1f}°"  if hip  is not None else "N/A"
    return (
        f'<div style="background:#1e1e2e;padding:0.8rem 1rem;border-radius:10px;'
        f'margin:0.4rem 0;color:#fff;font-size:0.95rem;">'
        f'<b>📐 Sudut Lutut:</b> {knee_str}<br>'
        f'<b>📐 Sudut Pinggul:</b> {hip_str}<br>'
        f'<b>👥 Orang Terdeteksi:</b> {n_persons}<br>'
        f'<b>⚡ FPS:</b> {fps}</div>'
    )

def build_stats_html(pcts):
    rows = ""
    for pose_name in ALL_POSES:
        pct   = pcts.get(pose_name, 0.0)
        c_hex = COLOR_MAP.get(pose_name, "#616161")
        em    = EMOJI_MAP.get(pose_name, "")
        rows += (
            f'<div style="margin:6px 0;">'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span style="color:#fff;">{em} {pose_name}</span>'
            f'<span style="color:#ccc;font-size:0.85rem;">{pct:.1f}%</span></div>'
            f'<div style="background:#444;border-radius:5px;height:12px;margin-top:3px;">'
            f'<div style="width:{pct:.1f}%;background:{c_hex};height:12px;border-radius:5px;"></div>'
            f'</div></div>'
        )
    return (
        f'<div style="background:#1e1e2e;padding:0.8rem 1rem;border-radius:10px;margin:0.4rem 0;">'
        f'<b style="color:#fff;">📊 Statistik pose:</b><br><br>{rows}</div>'
    )

# ─────────────────────────────────────────────
# Fungsi Proses Frame
# ─────────────────────────────────────────────
def process_frame(frame, conf_threshold, draw_skeleton, draw_keypoints, draw_angles):
    h_orig, w_orig = frame.shape[:2]

    scale = INFER_SIZE / max(h_orig, w_orig)
    w_inf = int(w_orig * scale)
    h_inf = int(h_orig * scale)
    small = cv2.resize(frame, (w_inf, h_inf), interpolation=cv2.INTER_LINEAR)

    results      = model(small, conf=conf_threshold, verbose=False, imgsz=INFER_SIZE)
    pose_results = []
    annotated    = frame.copy()
    num_persons  = 0
    sx           = w_orig / w_inf
    sy           = h_orig / h_inf

    for result in results:
        if result.keypoints is None:
            continue

        keypoints_data = result.keypoints.xy.cpu().numpy()
        boxes_data     = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        num_persons    = len(keypoints_data)

        for i, kps in enumerate(keypoints_data):
            kps_orig        = kps.copy()
            kps_orig[:, 0] *= sx
            kps_orig[:, 1] *= sy

            pose_label, angle_info = classify_pose(kps_orig)
            color = POSE_COLORS.get(pose_label, (128, 128, 128))

            if len(boxes_data) > i:
                x1   = int(boxes_data[i][0] * sx)
                y1   = int(boxes_data[i][1] * sy)
                x2   = int(boxes_data[i][2] * sx)
                y2   = int(boxes_data[i][3] * sy)
                bg_y = max(y1 - 35, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(annotated, (x1, bg_y), (x1 + 200, y1), color, -1)
                cv2.putText(annotated, pose_label, (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if draw_angles and angle_info.get("knee_angle") is not None:
                    cv2.putText(annotated,
                                f"Lutut: {angle_info['knee_angle']:.1f}deg",
                                (x1 + 5, max(bg_y - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if draw_keypoints:
                for kp in kps_orig:
                    x, y = int(kp[0]), int(kp[1])
                    if x > 0 and y > 0:
                        cv2.circle(annotated, (x, y), 5, (0, 255, 255), -1)
                        cv2.circle(annotated, (x, y), 5, (0, 0, 0), 1)

            if draw_skeleton:
                for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
                    if pt1_idx < len(kps_orig) and pt2_idx < len(kps_orig):
                        pt1 = (int(kps_orig[pt1_idx][0]), int(kps_orig[pt1_idx][1]))
                        pt2 = (int(kps_orig[pt2_idx][0]), int(kps_orig[pt2_idx][1]))
                        if all(v > 0 for v in [pt1[0], pt1[1], pt2[0], pt2[1]]):
                            cv2.line(annotated, pt1, pt2, color, 2)

            pose_results.append({
                "pose"       : pose_label,
                "knee_angle" : angle_info.get("knee_angle"),
                "hip_angle"  : angle_info.get("hip_angle"),
            })

    return annotated, pose_results, num_persons

# ─────────────────────────────────────────────
# Video Processor (WebRTC)
# FIX: semua config disimpan sebagai atribut agar
#      recv() tidak baca variabel dari scope global
# ─────────────────────────────────────────────
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_queue    = queue.Queue(maxsize=3)
        self._frame_counter  = 0
        self._last_annotated = None
        self._lock           = threading.Lock()
        self._fps_buf        = []
        self._t_prev         = time.time()

        # ← Config default; di-sync dari main thread tiap rerun
        self.confidence     = 0.5
        self.show_skeleton  = True
        self.show_keypoints = True
        self.show_angles    = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        self._frame_counter += 1

        now = time.time()
        dt  = max(now - self._t_prev, 1e-9)
        self._t_prev = now
        self._fps_buf.append(1.0 / dt)
        if len(self._fps_buf) > 15:
            self._fps_buf.pop(0)
        stream_fps = round(sum(self._fps_buf) / len(self._fps_buf), 1)

        # Baca config dari atribut sendiri (thread-safe, tidak baca global)
        conf       = self.confidence
        skeleton   = self.show_skeleton
        keypoints  = self.show_keypoints
        angles     = self.show_angles

        if self._frame_counter % FRAME_SKIP == 0:
            annotated, pose_results, num_persons = process_frame(
                img, conf, skeleton, keypoints, angles
            )
            with self._lock:
                self._last_annotated = annotated

            if pose_results:
                data = pose_results[0].copy()
                data["fps"]         = stream_fps
                data["num_persons"] = num_persons
                data["timestamp"]   = now
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put_nowait(data)
        else:
            with self._lock:
                annotated = self._last_annotated if self._last_annotated is not None else img

        cv2.putText(
            annotated, f"FPS: {stream_fps}",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2, cv2.LINE_AA
        )

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(rgb, format="rgb24")

# ─────────────────────────────────────────────
# Layout Utama
# ─────────────────────────────────────────────
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("📷 Feed Kamera")
    ctx = webrtc_streamer(
        key="pose-detection",
        video_processor_factory=PoseProcessor,
        media_stream_constraints={
            "video": {
                "width"    : {"ideal": 640},
                "height"   : {"ideal": 480},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

with col_info:
    st.subheader("📊 Hasil Deteksi")
    pose_placeholder    = st.empty()
    metrics_placeholder = st.empty()
    stats_placeholder   = st.empty()

# ─────────────────────────────────────────────
# Tampilan Default
# ─────────────────────────────────────────────
pose_placeholder.markdown(build_pose_html("Tidak Terdeteksi"), unsafe_allow_html=True)
metrics_placeholder.markdown(build_metrics_html(None, None, 0, 0), unsafe_allow_html=True)
stats_placeholder.markdown(build_stats_html({p: 0.0 for p in ALL_POSES}), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Loop Update Panel Info
# ─────────────────────────────────────────────
if ctx.state.playing and ctx.video_processor:
    # Sync config ke processor (main thread → recv thread)
    ctx.video_processor.confidence     = confidence
    ctx.video_processor.show_skeleton  = show_skeleton
    ctx.video_processor.show_keypoints = show_keypoints
    ctx.video_processor.show_angles    = show_angles

    # Non-blocking poll — tidak ada while True!
    try:
        result = ctx.video_processor.result_queue.get_nowait()
        st.session_state.last_display = {
            "label"    : result.get("pose", "Tidak Terdeteksi"),
            "knee"     : result.get("knee_angle"),
            "hip"      : result.get("hip_angle"),
            "fps"      : result.get("fps", 0.0),
            "n_persons": result.get("num_persons", 0),
        }
        ts = result.get("timestamp", time.time())
        if st.session_state.last_display["label"] in ALL_POSES:
            st.session_state.pose_window.append((ts, st.session_state.last_display["label"]))
    except queue.Empty:
        pass

    # Render dari session state (pakai data terakhir jika queue kosong)
    disp = st.session_state.last_display
    if disp:
        pcts = get_window_stats()
        pose_placeholder.markdown(build_pose_html(disp["label"]), unsafe_allow_html=True)
        metrics_placeholder.markdown(
            build_metrics_html(disp["knee"], disp["hip"], disp["n_persons"], disp["fps"]),
            unsafe_allow_html=True
        )
        stats_placeholder.markdown(build_stats_html(pcts), unsafe_allow_html=True)

    time.sleep(0.05)  # ~20 Hz UI refresh
    st.rerun()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#aaa;font-size:0.8rem;">'
    '🤸 Deteksi Pose Tubuh Realtime &nbsp;|&nbsp; YOLOv8 + Streamlit &nbsp;|&nbsp; Proyek Akhir JST'
    '</div>',
    unsafe_allow_html=True
)
