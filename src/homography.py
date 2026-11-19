import cv2
import numpy as np


def ensure_thermal_orientation(thermal_frame, rgb_shape):
    """
    Ensures the radiometric .npy frame matches the exact (H, W) orientation
    of the RGB video frame (480, 640).
    """
    h_rgb, w_rgb = rgb_shape[:2]
    h_th, w_th = thermal_frame.shape[:2]

    if (h_th, w_th) == (h_rgb, w_rgb):
        return thermal_frame.astype(np.float32)
    elif (w_th, h_th) == (h_rgb, w_rgb):
        bottom_warmth_orig = np.percentile(thermal_frame[-40:, :], 90)
        right_warmth_orig = np.percentile(thermal_frame[:, -40:], 90)
        if bottom_warmth_orig >= right_warmth_orig:
            return cv2.resize(thermal_frame.astype(np.float32), (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)
        else:
            return thermal_frame.T.astype(np.float32)
    else:
        return cv2.resize(thermal_frame.astype(np.float32), (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)


def segment_thermal_hand(thermal_frame):
    """
    Segments the warm hand silhouette from the radiometric temperature array (°C).
    """
    t_min, t_max = float(np.min(thermal_frame)), float(np.max(thermal_frame))
    if t_max - t_min < 1.5:
        return np.ones_like(thermal_frame, dtype=np.uint8) * 255

    norm_u8 = np.clip((thermal_frame - t_min) / (t_max - t_min) * 255.0, 0, 255).astype(np.uint8)
    norm_blur = cv2.GaussianBlur(norm_u8, (7, 7), 0)
    _, mask = cv2.threshold(norm_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_idx, 255, 0).astype(np.uint8)


def compute_self_calibrated_homography(rgb_frame, thermal_frame, landmarks_px=None):
    """
    Aspect-Locked & Web-Valley Calibrated Homography Registration:
    Accounts for the (640, 480) -> (480, 640) aspect ratio transformation between
    the native FLIR Lepton .npy array and OpenCV RGB coordinate space so that
    PIP, MCP, and Thumb markers sit directly on the center of the thermal joints.
    """
    thermal_oriented = ensure_thermal_orientation(thermal_frame, rgb_frame.shape)
    therm_mask = segment_thermal_hand(thermal_oriented)
    ys_t, xs_t = np.where(therm_mask > 0)

    if len(xs_t) < 100 or landmarks_px is None or len(landmarks_px) != 21:
        return np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    # 1. RGB skeletal bounds from MediaPipe 21 landmarks
    r_xmin_skel = float(np.min(landmarks_px[:, 0]))
    r_xmax_skel = float(np.max(landmarks_px[:, 0]))
    r_ymin_skel = float(np.min(landmarks_px[:, 1]))
    r_ymax_skel = float(np.max(landmarks_px[:, 1]))  # Wrist landmark 0

    r_width = max(10.0, r_xmax_skel - r_xmin_skel)
    r_height = max(10.0, r_ymax_skel - r_ymin_skel)

    # 2. Top fingertip crown of the thermal hand mask
    t_ymin = float(np.percentile(ys_t, 1.0))
    t_ymax_all = float(np.max(ys_t))

    # Restrict horizontal span measurement to the upper 55% of the mask (fingers + thumb)
    # so the lower forearm cannot skew horizontal bounds
    upper_finger_limit_y = t_ymin + 0.55 * (t_ymax_all - t_ymin)
    upper_mask = ys_t <= upper_finger_limit_y
    xs_upper = xs_t[upper_mask] if np.any(upper_mask) else xs_t

    t_xmin = float(np.percentile(xs_upper, 1.2))
    t_xmax = float(np.percentile(xs_upper, 98.8))
    t_width = max(20.0, t_xmax - t_xmin)

    # Inward finger-pad margin (~4.5% of hand width) to center markers inside the digits
    pad_x = 0.045 * t_width
    pad_y_top = 0.035 * t_width

    t_xmin_skel = t_xmin + pad_x
    t_xmax_skel = t_xmax - pad_x
    t_ymin_skel = t_ymin + pad_y_top

    # 3. Aspect-Corrected Vertical Scale:
    # Because the native (640, 480) thermal array is mapped into a (480, 640) frame,
    # vertical pixel scale is compressed relative to horizontal scale by ~0.62
    s_x = (t_xmax_skel - t_xmin_skel) / r_width
    s_y = s_x * 0.62
    t_ymax_skel = t_ymin_skel + s_y * r_height

    src_therm_pts = np.array([
        [t_xmin_skel, t_ymin_skel],
        [t_xmax_skel, t_ymin_skel],
        [t_xmax_skel, t_ymax_skel],
        [t_xmin_skel, t_ymax_skel],
    ], dtype=np.float32)

    dst_rgb_pts = np.array([
        [r_xmin_skel, r_ymin_skel],
        [r_xmax_skel, r_ymin_skel],
        [r_xmax_skel, r_ymax_skel],
        [r_xmin_skel, r_ymax_skel],
    ], dtype=np.float32)

    H_therm_to_rgb = cv2.getPerspectiveTransform(src_therm_pts, dst_rgb_pts)
    H_rgb_to_therm = cv2.getPerspectiveTransform(dst_rgb_pts, src_therm_pts)

    return H_therm_to_rgb, H_rgb_to_therm


def apply_cross_modal_registration(thermal_frame, H_therm_to_rgb, target_shape=(480, 640)):
    """
    Warps the thermal radiometric array (°C) onto the RGB pixel coordinate space.
    """
    h_rgb, w_rgb = target_shape[:2]
    thermal_oriented = ensure_thermal_orientation(thermal_frame, (h_rgb, w_rgb))
    bg_temp = float(np.percentile(thermal_oriented, 10))

    aligned_thermal = cv2.warpPerspective(
        thermal_oriented,
        H_therm_to_rgb,
        (w_rgb, h_rgb),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_temp,
    )
    return aligned_thermal


def map_points_homography(points_xy, H):
    """
    Maps an array of (N, 2) [x, y] coordinates using a 3x3 homography matrix H.
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H)
    return mapped.reshape(-1, 2)