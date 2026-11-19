import os
import glob
import cv2
import numpy as np


class SpatiotemporalDataLoader:
    def __init__(self, config):
        self.root_dir = config["dataset"]["root_dir"]
        self.seq_len = config["dataset"]["sequence_length"]
        self.roi_size = config["dataset"]["roi_size"]

    def load_synchronized_sequence(self, rgb_path, npy_path):
        """
        Loads the RGB .mp4 video and Thermal Radiometric .npy sequence,
        sampling `self.seq_len` evenly spaced frames across the recording.
        """
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB video not found: {rgb_path}")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Thermal radiometric .npy not found: {npy_path}")

        thermal_seq = np.load(npy_path, mmap_mode="r")
        total_therm_frames = thermal_seq.shape[0]

        cap = cv2.VideoCapture(rgb_path)
        total_rgb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_frames = min(total_rgb_frames, total_therm_frames) if total_rgb_frames > 0 else total_therm_frames

        indices = np.linspace(0, max(1, min_frames - 1), self.seq_len, dtype=int)
        index_set = set(indices.tolist())

        rgb_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in index_set:
                rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
            if len(rgb_frames) == len(index_set):
                break
        cap.release()

        # Fallback if OpenCV frame count metadata differed slightly
        if len(rgb_frames) < self.seq_len:
            cap = cv2.VideoCapture(rgb_path)
            all_rgb = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            min_f = min(len(all_rgb), total_therm_frames)
            indices = np.linspace(0, min_f - 1, self.seq_len, dtype=int)
            sampled_rgb = np.array([all_rgb[i] for i in indices], dtype=np.uint8)
        else:
            sampled_rgb = np.array(rgb_frames[: self.seq_len], dtype=np.uint8)

        sampled_thermal = np.array([thermal_seq[i] for i in indices[: len(sampled_rgb)]], dtype=np.float32)
        return sampled_rgb, sampled_thermal

    def discover_recordings(self):
        """
        Recursively discovers all paired (RGB.mp4, Thermal_Radiometric_Sequence.npy)
        recordings inside `self.root_dir`.
        """
        pattern = os.path.join(self.root_dir, "**", "*_RGB.mp4")
        rgb_files = sorted(glob.glob(pattern, recursive=True))
        recordings = []

        for rgb_path in rgb_files:
            prefix = rgb_path[: -len("_RGB.mp4")]
            npy_path = prefix + "_Thermal_Radiometric_Sequence.npy"
            if os.path.exists(npy_path):
                rel_id = os.path.relpath(prefix, self.root_dir).replace("\\", "/").replace("/", "_")
                recordings.append({
                    "id": rel_id,
                    "rgb_path": rgb_path,
                    "npy_path": npy_path,
                })
        return recordings

    @staticmethod
    def synthesize_arthritic_joint_sample(healthy_roi, healthy_kin, rng):
        """
        Generates a physiologically realistic positive Rheumatoid Arthritis joint sample
        from a baseline healthy joint sequence by injecting:
          1. Localized 2D Gaussian synovial hotspot ΔT in [+1.3°C, +3.2°C] (normalized by /4.0)
          2. Mild soft-tissue erythematous/swelling morphological modulation in RGB around joint center
          3. Pathological ulnar deviation (>15°) and/or reduced flexion ROM stiffness
        """
        T, H, W, _ = healthy_roi.shape
        pos_roi = healthy_roi.copy()
        pos_kin = healthy_kin.copy()

        # 1. Focal Gaussian synovial heat signature over joint center
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H // 2 + rng.integers(-3, 4), W // 2 + rng.integers(-3, 4)
        sigma = rng.uniform(10.0, 16.0)
        gaussian_hotspot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)

        # Peak ΔT between +1.4°C and +3.2°C (divided by 4.0 normalization scale)
        peak_delta_t_norm = rng.uniform(1.4, 3.2) / 4.0
        for t in range(T):
            temporal_flare = 1.0 + 0.08 * np.sin(2.0 * np.pi * t / max(1, T))
            pos_roi[t, :, :, 3] = np.clip(
                pos_roi[t, :, :, 3] + peak_delta_t_norm * temporal_flare * gaussian_hotspot,
                -1.0,
                1.5,
            )
            # Subtle erythema/swelling contrast shift on joint capsule in RGB
            pos_roi[t, :, :, 0] = np.clip(pos_roi[t, :, :, 0] + 0.06 * gaussian_hotspot, 0.0, 1.0)

        # 2. Kinematic pathology: elevated ulnar drift (16° to 35°) and stiffness
        ulnar_drift_norm = rng.uniform(16.0, 35.0) / 45.0
        pos_kin[:, 2] = np.clip(pos_kin[:, 2] + ulnar_drift_norm, 0.0, 1.2)
        if pos_kin[0, 3] > 0.5:  # Dynamic task: restrict flexion range of motion
            mean_flex = np.mean(pos_kin[:, 0])
            pos_kin[:, 0] = mean_flex + 0.35 * (pos_kin[:, 0] - mean_flex)
            pos_kin[:, 1] *= 0.40  # Reduced angular velocity

        return pos_roi, pos_kin