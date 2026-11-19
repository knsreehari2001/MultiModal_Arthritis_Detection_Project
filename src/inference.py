import os
import sys
import argparse
import yaml
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

from data_loader import SpatiotemporalDataLoader
from kinematics import KinematicEngine


def compute_calibrated_joint_probabilities(model, joint_rois, joint_kins, meta, config):
    """
    Computes localized pathology probability P(RA) for each of the 10 target joints.
    """
    if model is not None:
        nn_probs = model.predict([joint_rois, joint_kins], verbose=0).reshape(-1)
    else:
        nn_probs = np.full(10, 0.12, dtype=np.float32)

    delta_t = meta["joint_delta_t_c"]
    ulnar_deg = np.median(meta["ulnar_curves"], axis=1)
    flex_rom = np.ptp(meta["flexion_curves"], axis=1)
    is_dynamic = meta["is_dynamic"]

    dt_thresh = config["clinical_thresholds"]["delta_t_synovitis_c"]
    ulnar_thresh = config["clinical_thresholds"]["ulnar_drift_deg"]
    rom_thresh = config["clinical_thresholds"]["min_dynamic_rom_deg"]

    final_probs = np.zeros(10, dtype=np.float32)

    for j in range(10):
        # 1. Localized Thermal Synovitis Score (0 to 1)
        thermal_score = 1.0 / (1.0 + np.exp(-3.5 * (delta_t[j] - dt_thresh)))

        # 2. Open-Hand Structural Ulnar Drift Score (0 to 1)
        ulnar_score = 1.0 / (1.0 + np.exp(-0.40 * (ulnar_deg[j] - ulnar_thresh)))

        # 3. Task-aware Flexion Stiffness Score (0 to 1)
        if is_dynamic and j >= 2:
            stiff_score = 1.0 / (1.0 + np.exp(0.15 * (flex_rom[j] - rom_thresh)))
        else:
            stiff_score = 0.04

        biomarker_prob = 0.55 * thermal_score + 0.30 * ulnar_score + 0.15 * stiff_score

        # Physiological Gate: If both relative ΔT (< +0.5°C) and open-hand ulnar drift (< 12°)
        # are strictly normal, cap the neural stream contribution so boundary motion cannot trigger false positives.
        if delta_t[j] < 0.5 and ulnar_deg[j] < 12.0:
            p_j = 0.20 * float(nn_probs[j]) + 0.80 * float(biomarker_prob)
        else:
            p_j = 0.55 * float(nn_probs[j]) + 0.45 * float(biomarker_prob)

        final_probs[j] = np.clip(p_j * 100.0, 2.0, 99.0)

    top3_mean = float(np.mean(np.sort(final_probs)[-3:]))
    overall_prob = float(0.60 * top3_mean + 0.40 * np.mean(final_probs))
    return final_probs, overall_prob


def generate_diagnostic_dashboard(meta, joint_probs, overall_prob, out_path):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8), dpi=150)
    ax_rgb, ax_therm, ax_kin, ax_bar = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    joint_names = meta["joint_names"]
    lm_ids = meta["joint_landmark_ids"]
    rgb_lm = meta["rep_rgb_landmarks"]
    therm_lm = meta["rep_thermal_landmarks"]
    joint_temps = meta["joint_temps_c"]
    delta_ts = meta["joint_delta_t_c"]
    palm_ref = meta["palm_ref_temp_c"]

    # Panel 1: Topological Landmark Tracking (MediaPipe)
    ax_rgb.imshow(meta["rep_rgb_frame"])
    for j_idx, lm_id in enumerate(lm_ids):
        x, y = rgb_lm[lm_id]
        ax_rgb.scatter(x, y, c="lime", s=45, edgecolors="black", linewidths=0.8, zorder=5)
        if "MCP" in joint_names[j_idx]:
            short_lbl = joint_names[j_idx].split()[0]
            ax_rgb.text(
                x + 6, y + 4, short_lbl, color="white", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, ec="none"),
            )
    ax_rgb.set_title("1. Topological Landmark Tracking (MediaPipe)", fontweight="bold", fontsize=11)
    ax_rgb.axis("off")

    # Panel 2: Registered Thermal Signature (°C)
    therm_img = meta["rep_thermal_frame"]
    vmin = max(20.0, float(np.percentile(therm_img, 5)))
    vmax = min(38.0, max(35.0, float(np.percentile(therm_img, 99.5))))
    im = ax_therm.imshow(therm_img, cmap="inferno", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax_therm, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)", fontsize=9)

    for j_idx, lm_id in enumerate(lm_ids):
        tx, ty = therm_lm[lm_id]
        ax_therm.scatter(tx, ty, c="cyan", s=42, edgecolors="black", linewidths=0.9, zorder=5)
        # Stagger vertical label offset between PIP and MCP rows to prevent text overlap
        dy_offset = -8 if ("PIP" in joint_names[j_idx] or "IP" in joint_names[j_idx]) else 10
        ax_therm.text(
            tx + 4,
            ty + dy_offset,
            f"{joint_temps[j_idx]:.1f}°C (Δ{delta_ts[j_idx]:+.1f})",
            color="cyan",
            fontsize=6.3,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.55, ec="none"),
        )
    ax_therm.set_title(
        f"2. Registered Thermal Signature (Palm Ref: {palm_ref:.1f}°C)",
        fontweight="bold",
        fontsize=11,
    )
    ax_therm.axis("off")

    # Panel 3: Temporal Joint Kinematics (θ over Time)
    task_str = "Dynamic Open-Close" if meta["is_dynamic"] else "Static Posture"
    for j_idx in range(6):  # Thumb, Index, Middle MCP/PIP
        ax_kin.plot(meta["flexion_curves"][j_idx], label=joint_names[j_idx], linewidth=1.8)
    ax_kin.set_title(f"3. Temporal Joint Kinematics ({task_str})", fontweight="bold", fontsize=11)
    ax_kin.set_xlabel(f"Normalized Frame Index (T={meta['flexion_curves'].shape[1]})")
    ax_kin.set_ylabel("3D Joint Flexion Angle (°)")
    ax_kin.grid(True, alpha=0.3)
    ax_kin.legend(loc="upper right", fontsize=7.5)

    # Panel 4: Localized Joint-Level Diagnosis
    y_pos = np.arange(len(joint_names))
    bar_colors = ["crimson" if p >= 50.0 else ("darkorange" if p >= 35.0 else "seagreen") for p in joint_probs]
    ax_bar.barh(y_pos, joint_probs, color=bar_colors, edgecolor="black", height=0.72, alpha=0.9)
    ax_bar.axvline(50.0, color="dimgray", linestyle="--", linewidth=1.2, label="Diagnostic Threshold (50%)")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(joint_names, fontsize=8.5)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Localized Pathology Probability (%)")
    status_str = "HIGH RISK" if overall_prob >= 50.0 else "NORMAL / LOW RISK"
    ax_bar.set_title(
        f"4. Localized Joint-Level Diagnosis (Overall: {overall_prob:.1f}% — {status_str})",
        fontweight="bold",
        fontsize=10.5,
    )
    for idx, p_val in enumerate(joint_probs):
        ax_bar.text(min(94.0, p_val + 1.5), idx, f"{p_val:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved diagnostic dashboard to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB .mp4 video")
    parser.add_argument("--thermal", type=str, required=True, help="Path to Radiometric .npy file")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="reports/diagnostic_report.png")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    loader = SpatiotemporalDataLoader(config)
    engine = KinematicEngine(roi_size=config["dataset"]["roi_size"])

    rgb_seq, therm_seq = loader.load_synchronized_sequence(args.rgb, args.thermal)
    joint_rois, joint_kins, meta = engine.process_sequence(rgb_seq, therm_seq)

    ckpt_path = "checkpoints/best_arthritis_model.keras"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/final_arthritis_model.keras"
    model = tf.keras.models.load_model(ckpt_path, compile=False) if os.path.exists(ckpt_path) else None

    joint_probs, overall_prob = compute_calibrated_joint_probabilities(
        model, joint_rois, joint_kins, meta, config
    )

    print("\n" + "=" * 68)
    print(" MULTI-MODAL ARTHRITIS DIAGNOSTIC SUMMARY")
    print("=" * 68)
    print(f" Task Detected          : {'Dynamic Open-Close' if meta['is_dynamic'] else 'Static Hand Posture'}")
    print(f" Dorsal Palm Ref Temp   : {meta['palm_ref_temp_c']:.2f} °C")
    print("-" * 68)
    for j_idx, name in enumerate(meta["joint_names"]):
        print(
            f" {name:<12} | Temp: {meta['joint_temps_c'][j_idx]:5.2f}°C "
            f"(ΔT: {meta['joint_delta_t_c'][j_idx]:+5.2f}°C) | "
            f"Ulnar Dev: {np.median(meta['ulnar_curves'][j_idx]):4.1f}° | "
            f"Risk: {joint_probs[j_idx]:5.2f}%"
        )
    print("-" * 68)
    risk_label = "HIGH RISK (Active Synovitis)" if overall_prob >= 50.0 else "LOW RISK / NORMAL"
    print(f" OVERALL HAND PROBABILITY: {overall_prob:.2f}% ({risk_label})")
    print("=" * 68 + "\n")

    generate_diagnostic_dashboard(meta, joint_probs, overall_prob, args.output)


if __name__ == "__main__":
    main()