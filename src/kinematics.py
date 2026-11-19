import cv2
import numpy as np
import mediapipe as mp
from homography import (
    ensure_thermal_orientation,
    compute_self_calibrated_homography,
    apply_cross_modal_registration,
    map_points_homography,
)

# 10 Target Articular Joints:
# (joint_name, mp_landmark_idx, parent_idx, child_idx, tip_idx, is_mcp, natural_splay_deg)
TARGET_JOINTS = [
    ("Thumb MCP",  2,  1,  3,  4, True,  18.0),
    ("Thumb IP",   3,  2,  4,  4, False, 15.0),
    ("Index MCP",  5,  0,  6,  8, True,  10.0),
    ("Index PIP",  6,  5,  7,  8, False,  8.0),
    ("Middle MCP", 9,  0, 10, 12, True,   8.0),
    ("Middle PIP", 10, 9, 11, 12, False,  8.0),
    ("Ring MCP",   13, 0, 14, 16, True,  12.0),
    ("Ring PIP",   14, 13, 15, 16, False, 10.0),
    ("Pinky MCP",  17, 0, 18, 20, True,  22.0),
    ("Pinky PIP",  18, 17, 19, 20, False, 14.0),
]


class KinematicEngine:
    def __init__(self, roi_size=64):
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.roi_size = roi_size
        self.half_roi = roi_size // 2

    @staticmethod
    def _compute_3d_flexion_angle(lm_array_3d, idx_joint, idx_parent, idx_child, idx_tip, is_mcp):
        """
        Computes the anatomical joint flexion angle (degrees, ~175°-180° open, ~45°-95° closed fist)
        using 3D vectors combined with fingertip-to-wrist foreshortening compensation.
        """
        v1 = lm_array_3d[idx_parent] - lm_array_3d[idx_joint]
        v2 = lm_array_3d[idx_child] - lm_array_3d[idx_joint]
        n1 = np.linalg.norm(v1) + 1e-8
        n2 = np.linalg.norm(v2) + 1e-8
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle_3d = float(np.degrees(np.arccos(cos_theta)))

        # Distal tip foreshortening ratio relative to palm size
        wrist_xy = lm_array_3d[0, :2]
        palm_len = np.linalg.norm(lm_array_3d[9, :2] - wrist_xy) + 1e-6
        tip_dist = np.linalg.norm(lm_array_3d[idx_tip, :2] - wrist_xy)
        ext_ratio = np.clip((tip_dist - 0.70 * palm_len) / (1.15 * palm_len), 0.0, 1.0)

        if is_mcp:
            # Anatomically, MCP flexes from ~178° (open) to ~85° (closed fist)
            foreshortened_angle = 82.0 + 96.0 * float(ext_ratio)
            return 0.35 * angle_3d + 0.65 * foreshortened_angle
        else:
            # PIP flexes more sharply from ~178° (open) to ~40° (closed fist)
            foreshortened_angle = 38.0 + 140.0 * float(ext_ratio)
            return 0.30 * angle_3d + 0.70 * foreshortened_angle

    @staticmethod
    def _compute_ulnar_deviation_deg(lm_px, idx_joint, idx_parent, idx_child, natural_splay_deg):
        """
        Computes lateral structural deviation (ulnar drift) in degrees on an open hand,
        subtracting normal physiological finger abduction splay (e.g. spread pinky/thumb).
        """
        v_prox = lm_px[idx_joint] - lm_px[idx_parent]
        v_dist = lm_px[idx_child] - lm_px[idx_joint]
        n1 = np.linalg.norm(v_prox) + 1e-8
        n2 = np.linalg.norm(v_dist) + 1e-8
        cos_dev = np.clip(np.dot(v_prox, v_dist) / (n1 * n2), -1.0, 1.0)
        raw_dev_deg = float(np.degrees(np.arccos(cos_dev)))

        # Subtract natural anatomical finger abduction splay
        corrected_dev = max(0.0, raw_dev_deg - natural_splay_deg)
        return float(np.clip(corrected_dev, 0.0, 60.0))

    def process_sequence(self, rgb_seq, thermal_seq):
        T = len(rgb_seq)
        h_rgb, w_rgb = rgb_seq[0].shape[:2]

        all_lm_px, all_lm_3d, valid_mask = [], [], []

        for t in range(T):
            res = self.mp_hands.process(rgb_seq[t])
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                px = np.array([[p.x * w_rgb, p.y * h_rgb] for p in lm], dtype=np.float32)
                p3d = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                all_lm_px.append(px)
                all_lm_3d.append(p3d)
                valid_mask.append(True)
            else:
                all_lm_px.append(None)
                all_lm_3d.append(None)
                valid_mask.append(False)

        # Interpolate any missing frames
        last_px = np.zeros((21, 2), dtype=np.float32)
        last_3d = np.zeros((21, 3), dtype=np.float32)
        for idx in range(T):
            if valid_mask[idx]:
                last_px, last_3d = all_lm_px[idx], all_lm_3d[idx]
                break
        for t in range(T):
            if not valid_mask[t]:
                all_lm_px[t], all_lm_3d[t] = last_px.copy(), last_3d.copy()
            else:
                last_px, last_3d = all_lm_px[t], all_lm_3d[t]

        # Identify open-hand frames (top 35% most extended frames) for ulnar deviation measurement
        # so closed-fist frames during open_close tasks never corrupt ulnar deviation!
        finger_spreads = np.array([
            np.mean(np.linalg.norm(all_lm_px[t][[8, 12, 16, 20]] - all_lm_px[t][0], axis=1))
            for t in range(T)
        ])
        open_threshold = np.percentile(finger_spreads, 65)
        open_frame_indices = np.where(finger_spreads >= open_threshold)[0]
        if len(open_frame_indices) == 0:
            open_frame_indices = np.arange(T)

        flexion_curves = np.zeros((10, T), dtype=np.float32)
        ulnar_curves = np.zeros((10, T), dtype=np.float32)

        for j_idx, (_, lm_id, parent_id, child_id, tip_id, is_mcp, splay_tol) in enumerate(TARGET_JOINTS):
            # Compute open-hand baseline ulnar deviation on extended frames only
            open_devs = [
                self._compute_ulnar_deviation_deg(all_lm_px[t], lm_id, parent_id, child_id, splay_tol)
                for t in open_frame_indices
            ]
            baseline_ulnar_dev = float(np.percentile(open_devs, 25))

            for t in range(T):
                flexion_curves[j_idx, t] = self._compute_3d_flexion_angle(
                    all_lm_3d[t], lm_id, parent_id, child_id, tip_id, is_mcp
                )
                ulnar_curves[j_idx, t] = baseline_ulnar_dev

        median_rom = float(np.median(np.ptp(flexion_curves[2:], axis=1)))
        is_dynamic = 1.0 if median_rom > 25.0 else 0.0

        # Select representative fully open-hand frame for visualization & calibration
        rep_frame_idx = int(np.argmax(finger_spreads)) if any(valid_mask) else 0

        H_therm_to_rgb_ref, H_rgb_to_therm_ref = compute_self_calibrated_homography(
            rgb_seq[rep_frame_idx],
            thermal_seq[rep_frame_idx],
            all_lm_px[rep_frame_idx] if valid_mask[rep_frame_idx] else None,
        )

        joint_rois = np.zeros((10, T, self.roi_size, self.roi_size, 4), dtype=np.float32)
        joint_temps_t = np.zeros((10, T), dtype=np.float32)
        palm_ref_temps_t = np.zeros(T, dtype=np.float32)

        for t in range(T):
            # Use open-hand calibrated homography on closed-fist frames to avoid fist bounding drift
            if valid_mask[t] and (t in open_frame_indices):
                H_t2r, _ = compute_self_calibrated_homography(
                    rgb_seq[t], thermal_seq[t], all_lm_px[t]
                )
                H_therm_to_rgb = 0.6 * H_t2r + 0.4 * H_therm_to_rgb_ref
            else:
                H_therm_to_rgb = H_therm_to_rgb_ref

            aligned_therm = apply_cross_modal_registration(
                thermal_seq[t], H_therm_to_rgb, target_shape=(h_rgb, w_rgb)
            )

            lm_px = all_lm_px[t]
            palm_cx = int(np.clip(np.mean(lm_px[[0, 5, 9, 13, 17], 0]), 20, w_rgb - 21))
            palm_cy = int(np.clip(np.mean(lm_px[[0, 5, 9, 13, 17], 1]), 20, h_rgb - 21))
            palm_patch = aligned_therm[palm_cy - 18 : palm_cy + 18, palm_cx - 18 : palm_cx + 18]
            t_palm_ref = float(np.percentile(palm_patch, 80)) if palm_patch.size > 0 else float(np.percentile(aligned_therm, 85))
            palm_ref_temps_t[t] = t_palm_ref

            pad = self.half_roi
            rgb_padded = np.pad(rgb_seq[t], ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            therm_padded = np.pad(aligned_therm, ((pad, pad), (pad, pad)), mode="edge")

            for j_idx, (_, lm_id, _, _, _, _, _) in enumerate(TARGET_JOINTS):
                jx = int(np.clip(lm_px[lm_id, 0], 0, w_rgb - 1)) + pad
                jy = int(np.clip(lm_px[lm_id, 1], 0, h_rgb - 1)) + pad

                rgb_crop = rgb_padded[
                    jy - self.half_roi : jy + self.half_roi,
                    jx - self.half_roi : jx + self.half_roi,
                ].astype(np.float32) / 255.0

                therm_crop = therm_padded[
                    jy - self.half_roi : jy + self.half_roi,
                    jx - self.half_roi : jx + self.half_roi,
                ].astype(np.float32)

                # Sample joint tissue temperature using the 85th percentile of warm tissue pixels
                # inside the 20x20 center patch so cold background next to pinky/thumb is ignored
                c0, c1 = self.half_roi - 10, self.half_roi + 10
                center_patch = therm_crop[c0:c1, c0:c1]
                warm_tissue = center_patch[center_patch > (t_palm_ref - 4.0)]
                if warm_tissue.size > 0:
                    joint_temp_c = float(np.percentile(warm_tissue, 85))
                else:
                    joint_temp_c = float(np.percentile(center_patch, 90))
                joint_temps_t[j_idx, t] = joint_temp_c

                # Clamp background pixels (< t_palm_ref - 2.0°C) in Channel 4 so boundary edges
                # on pinky/thumb do not mimic focal synovial gradients inside the 3D CNN
                therm_fg = np.maximum(therm_crop, t_palm_ref - 1.5)
                delta_t_crop = np.clip((therm_fg - t_palm_ref) / 4.0, -0.5, 1.5)

                joint_rois[j_idx, t, :, :, :3] = rgb_crop
                joint_rois[j_idx, t, :, :, 3] = delta_t_crop

        joint_kinematics = np.zeros((10, T, 4), dtype=np.float32)
        for j_idx in range(10):
            vel = np.gradient(flexion_curves[j_idx])
            joint_kinematics[j_idx, :, 0] = flexion_curves[j_idx] / 180.0
            joint_kinematics[j_idx, :, 1] = np.abs(vel) / 45.0
            joint_kinematics[j_idx, :, 2] = ulnar_curves[j_idx] / 45.0
            joint_kinematics[j_idx, :, 3] = is_dynamic

        rep_rgb_lm = all_lm_px[rep_frame_idx]
        rep_therm_oriented = ensure_thermal_orientation(thermal_seq[rep_frame_idx], (h_rgb, w_rgb))
        rep_therm_lm = map_points_homography(rep_rgb_lm, H_rgb_to_therm_ref)

        # Evaluate representative open-hand joint temperatures cleanly at the mapped thermal coordinates
        open_joint_temps = np.percentile(joint_temps_t[:, open_frame_indices], 75, axis=1)
        open_palm_ref = float(np.median(palm_ref_temps_t[open_frame_indices]))

        metadata = {
            "rep_frame_idx": rep_frame_idx,
            "rep_rgb_frame": rgb_seq[rep_frame_idx],
            "rep_thermal_frame": rep_therm_oriented,
            "rep_aligned_thermal": apply_cross_modal_registration(
                thermal_seq[rep_frame_idx], H_therm_to_rgb_ref, (h_rgb, w_rgb)
            ),
            "rep_rgb_landmarks": rep_rgb_lm,
            "rep_thermal_landmarks": rep_therm_lm,
            "flexion_curves": flexion_curves,
            "ulnar_curves": ulnar_curves,
            "joint_temps_c": open_joint_temps,
            "palm_ref_temp_c": open_palm_ref,
            "joint_delta_t_c": open_joint_temps - open_palm_ref,
            "is_dynamic": bool(is_dynamic > 0.5),
            "joint_names": [j[0] for j in TARGET_JOINTS],
            "joint_landmark_ids": [j[1] for j in TARGET_JOINTS],
        }

        return joint_rois, joint_kinematics, metadata