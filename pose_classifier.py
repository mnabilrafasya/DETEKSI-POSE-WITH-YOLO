import numpy as np

# ─────────────────────────────────────────────
# Index Keypoint COCO (YOLOv8 Pose)
# ─────────────────────────────────────────────
NOSE           = 0
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16


def get_keypoint(keypoints, index):
    if keypoints is None or len(keypoints) <= index:
        return None
    kp = keypoints[index]
    if len(kp) >= 2:
        x, y = float(kp[0]), float(kp[1])
        if x == 0 and y == 0:
            return None
        return (x, y)
    return None


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle


def midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def classify_pose(keypoints):
    l_shoulder = get_keypoint(keypoints, LEFT_SHOULDER)
    r_shoulder = get_keypoint(keypoints, RIGHT_SHOULDER)
    l_hip      = get_keypoint(keypoints, LEFT_HIP)
    r_hip      = get_keypoint(keypoints, RIGHT_HIP)
    l_knee     = get_keypoint(keypoints, LEFT_KNEE)
    r_knee     = get_keypoint(keypoints, RIGHT_KNEE)
    l_ankle    = get_keypoint(keypoints, LEFT_ANKLE)
    r_ankle    = get_keypoint(keypoints, RIGHT_ANKLE)

    shoulder_mid = midpoint(l_shoulder, r_shoulder)
    hip_mid      = midpoint(l_hip, r_hip)
    knee_mid     = midpoint(l_knee, r_knee)

    knee_angle_left  = calculate_angle(l_hip, l_knee, l_ankle)
    knee_angle_right = calculate_angle(r_hip, r_knee, r_ankle)
    angles = [a for a in [knee_angle_left, knee_angle_right] if a is not None]
    knee_angle = np.mean(angles) if angles else None

    hip_angle_left  = calculate_angle(l_shoulder, l_hip, l_knee)
    hip_angle_right = calculate_angle(r_shoulder, r_hip, r_knee)
    hip_angles = [a for a in [hip_angle_left, hip_angle_right] if a is not None]
    hip_angle = np.mean(hip_angles) if hip_angles else None

    body_horizontal = False
    if shoulder_mid and hip_mid:
        dy = abs(shoulder_mid[1] - hip_mid[1])
        dx = abs(shoulder_mid[0] - hip_mid[0])
        if dx > dy * 0.8:
            body_horizontal = True

    hip_above_knee = False
    hip_below_knee = False
    if hip_mid and knee_mid:
        if hip_mid[1] < knee_mid[1]:
            hip_above_knee = True
        else:
            hip_below_knee = True

    if body_horizontal:
        return "Berbaring", {"knee_angle": knee_angle, "hip_angle": hip_angle}

    if knee_angle is not None:
        if knee_angle > 150:
            return "Berdiri", {"knee_angle": knee_angle, "hip_angle": hip_angle}
        if knee_angle < 90 and (hip_below_knee or not hip_above_knee):
            return "Jongkok", {"knee_angle": knee_angle, "hip_angle": hip_angle}
        if 80 <= knee_angle <= 150:
            return "Duduk", {"knee_angle": knee_angle, "hip_angle": hip_angle}
        if knee_angle < 80:
            return "Jongkok", {"knee_angle": knee_angle, "hip_angle": hip_angle}

    return "Tidak Terdeteksi", {"knee_angle": None, "hip_angle": None}


POSE_COLORS = {
    "Berdiri"          : (0, 255, 0),
    "Duduk"            : (255, 165, 0),
    "Jongkok"          : (0, 165, 255),
    "Berbaring"        : (255, 0, 255),
    "Tidak Terdeteksi" : (128, 128, 128),
}

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
