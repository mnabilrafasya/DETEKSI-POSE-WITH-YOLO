import numpy as np

# ─────────────────────────────────────────────
# Index Keypoint COCO (YOLOv8 Pose)
# ─────────────────────────────────────────────
NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16


def get_keypoint(keypoints, index, min_conf=0.0):
    """Return (x, y) atau None. Juga cek confidence kalau array punya kolom ke-3."""
    if keypoints is None or len(keypoints) <= index:
        return None
    kp = keypoints[index]
    x, y = float(kp[0]), float(kp[1])
    if x < 1 and y < 1:   # koordinat (0,0) = tidak terdeteksi
        return None
    if len(kp) >= 3 and float(kp[2]) < min_conf:
        return None
    return (x, y)


def calculate_angle(a, b, c):
    """Sudut di titik b (a-b-c), dalam derajat [0, 180]."""
    if a is None or b is None or c is None:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    angle  = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angle)


def midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def avg_angle(*angles):
    valid = [a for a in angles if a is not None]
    return float(np.mean(valid)) if valid else None


def classify_pose(keypoints):
    """
    Klasifikasi pose menggunakan voting multi-sinyal:
      - knee_angle  : sudut lutut (hip-knee-ankle)
      - hip_angle   : sudut pinggul (shoulder-hip-knee)
      - torso_angle : kemiringan badan (shoulder_mid → hip_mid)
      - hip_y vs knee_y : posisi relatif
    
    Return: (label, {"knee_angle": float|None, "hip_angle": float|None})
    """
    # ── Ambil keypoint ────────────────────────────────────────────────
    l_sh  = get_keypoint(keypoints, LEFT_SHOULDER)
    r_sh  = get_keypoint(keypoints, RIGHT_SHOULDER)
    l_hip = get_keypoint(keypoints, LEFT_HIP)
    r_hip = get_keypoint(keypoints, RIGHT_HIP)
    l_kn  = get_keypoint(keypoints, LEFT_KNEE)
    r_kn  = get_keypoint(keypoints, RIGHT_KNEE)
    l_an  = get_keypoint(keypoints, LEFT_ANKLE)
    r_an  = get_keypoint(keypoints, RIGHT_ANKLE)

    # ── Midpoints ────────────────────────────────────────────────────
    sh_mid  = midpoint(l_sh,  r_sh)
    hip_mid = midpoint(l_hip, r_hip)
    kn_mid  = midpoint(l_kn,  r_kn)

    # ── Hitung sudut utama ───────────────────────────────────────────
    knee_l = calculate_angle(l_hip, l_kn, l_an)
    knee_r = calculate_angle(r_hip, r_kn, r_an)
    knee_angle = avg_angle(knee_l, knee_r)

    hip_l = calculate_angle(l_sh, l_hip, l_kn)
    hip_r = calculate_angle(r_sh, r_hip, r_kn)
    hip_angle = avg_angle(hip_l, hip_r)

    angle_info = {"knee_angle": knee_angle, "hip_angle": hip_angle}

    # ── Sinyal bantu ─────────────────────────────────────────────────
    # 1. Kemiringan torso (0° = vertikal, 90° = horizontal)
    torso_angle = None
    if sh_mid and hip_mid:
        dy = hip_mid[1] - sh_mid[1]   # positif = hip di bawah shoulder (normal)
        dx = hip_mid[0] - sh_mid[0]
        torso_angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))
        # torso_angle mendekati 0 = tegak, mendekati 90 = berbaring

    # 2. Apakah hip lebih tinggi dari knee di layar? (y lebih kecil = lebih tinggi)
    hip_above_knee = False
    hip_below_knee = False
    if hip_mid and kn_mid:
        if hip_mid[1] < kn_mid[1] - 10:   # margin 10px agar tidak flip karena noise
            hip_above_knee = True
        elif hip_mid[1] > kn_mid[1] + 10:
            hip_below_knee = True

    # 3. Jarak vertikal hip–ankle (tubuh berdiri = jarak besar)
    vertical_span = None
    if sh_mid and (l_an or r_an):
        an = l_an or r_an
        vertical_span = abs(sh_mid[1] - an[1])

    # ── Voting ───────────────────────────────────────────────────────
    scores = {"Berdiri": 0, "Duduk": 0, "Jongkok": 0, "Berbaring": 0}

    # -- Sinyal torso_angle --
    if torso_angle is not None:
        if torso_angle > 50:              # badan sangat miring → berbaring
            scores["Berbaring"] += 4
        elif torso_angle > 30:            # agak miring
            scores["Berbaring"] += 2
        else:                             # badan tegak
            scores["Berdiri"]   += 1
            scores["Duduk"]     += 1

    # -- Sinyal knee_angle --
    if knee_angle is not None:
        if knee_angle >= 160:
            scores["Berdiri"]   += 4
        elif knee_angle >= 140:
            scores["Berdiri"]   += 2
            scores["Duduk"]     += 1
        elif knee_angle >= 110:
            scores["Duduk"]     += 4
        elif knee_angle >= 80:
            scores["Duduk"]     += 2
            scores["Jongkok"]   += 2
        elif knee_angle >= 55:
            scores["Jongkok"]   += 4
        else:
            scores["Jongkok"]   += 3
            scores["Berbaring"] += 1   # bisa juga berbaring dengan kaki ditekuk

    # -- Sinyal hip_angle --
    if hip_angle is not None:
        if hip_angle >= 160:
            scores["Berdiri"]   += 2
        elif hip_angle >= 130:
            scores["Duduk"]     += 1
            scores["Berdiri"]   += 1
        elif hip_angle >= 80:
            scores["Duduk"]     += 3
        elif hip_angle >= 50:
            scores["Jongkok"]   += 2
            scores["Duduk"]     += 1
        else:
            scores["Jongkok"]   += 2
            scores["Berbaring"] += 1

    # -- Sinyal posisi hip vs knee --
    if hip_above_knee:
        scores["Berdiri"]   += 2
        scores["Berbaring"] += 1
    if hip_below_knee:
        scores["Jongkok"]   += 2
        scores["Duduk"]     += 1

    # ── Guard: kalau hampir semua keypoint tidak ada → tidak terdeteksi ──
    detected_kps = sum(1 for kp in [l_sh, r_sh, l_hip, r_hip, l_kn, r_kn, l_an, r_an]
                       if kp is not None)
    if detected_kps < 4:
        return "Tidak Terdeteksi", angle_info

    # ── Ambil pemenang voting ─────────────────────────────────────────
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # Kalau skor terlalu rendah = tidak yakin
    if best_score < 3:
        return "Tidak Terdeteksi", angle_info

    return best_label, angle_info


POSE_COLORS = {
    "Berdiri"          : (0, 200, 80),
    "Duduk"            : (255, 140, 0),
    "Jongkok"          : (0, 140, 255),
    "Berbaring"        : (200, 0, 220),
    "Tidak Terdeteksi" : (120, 120, 120),
}

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),    # kepala
    (5, 6),                              # bahu-bahu
    (5, 7), (7, 9),                      # lengan kiri
    (6, 8), (8, 10),                     # lengan kanan
    (5, 11), (6, 12),                    # bahu-pinggul
    (11, 12),                            # pinggul-pinggul
    (11, 13), (13, 15),                  # kaki kiri
    (12, 14), (14, 16),                  # kaki kanan
]
