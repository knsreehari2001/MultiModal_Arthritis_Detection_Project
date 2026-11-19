import os
import sys
import argparse
import yaml
import json
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from data_loader import SpatiotemporalDataLoader
from kinematics import KinematicEngine
from inference import compute_calibrated_joint_probabilities


def generate_subject_combined_dashboard(
    subj_id,
    hand_name,
    static_meta,
    static_probs,
    static_overall,
    dynamic_meta,
    dynamic_probs,
    dynamic_overall,
    combined_joint_probs,
    combined_overall,
    out_path,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.2), dpi=150)
    (ax_s_therm, ax_d_rgb, ax_d_therm), (ax_d_kin, ax_dt_comp, ax_prob_comp) = axes

    joint_names = static_meta["joint_names"]
    lm_ids = static_meta["joint_landmark_ids"]

    # -------------------------------------------------------------
    # 1. Static Registered Thermal Signature
    # -------------------------------------------------------------
    s_therm_img = static_meta["rep_thermal_frame"]
    s_vmin = max(20.0, float(np.percentile(s_therm_img, 5)))
    s_vmax = min(38.0, max(35.0, float(np.percentile(s_therm_img, 99.5))))
    im1 = ax_s_therm.imshow(s_therm_img, cmap="inferno", vmin=s_vmin, vmax=s_vmax)
    cbar1 = fig.colorbar(im1, ax=ax_s_therm, fraction=0.046, pad=0.04)
    cbar1.set_label("°C", fontsize=8.5)

    s_therm_lm = static_meta["rep_thermal_landmarks"]
    s_temps = static_meta["joint_temps_c"]
    s_dts = static_meta["joint_delta_t_c"]
    s_palm = static_meta["palm_ref_temp_c"]

    for j_idx, lm_id in enumerate(lm_ids):
        tx, ty = s_therm_lm[lm_id]
        ax_s_therm.scatter(tx, ty, c="cyan", s=38, edgecolors="black", linewidths=0.8, zorder=5)
        dy_off = -7 if ("PIP" in joint_names[j_idx] or "IP" in joint_names[j_idx]) else 9
        ax_s_therm.text(
            tx + 3,
            ty + dy_off,
            f"{s_temps[j_idx]:.1f}° (Δ{s_dts[j_idx]:+.1f})",
            color="cyan",
            fontsize=5.8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.55, ec="none"),
        )
    ax_s_therm.set_title(
        f"1. Static Thermal Signature (Palm: {s_palm:.1f}°C)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax_s_therm.axis("off")

    # -------------------------------------------------------------
    # 2. Dynamic Motion RGB Representative Frame
    # -------------------------------------------------------------
    ax_d_rgb.imshow(dynamic_meta["rep_rgb_frame"])
    d_rgb_lm = dynamic_meta["rep_rgb_landmarks"]
    for j_idx, lm_id in enumerate(lm_ids):
        x, y = d_rgb_lm[lm_id]
        ax_d_rgb.scatter(x, y, c="lime", s=40, edgecolors="black", linewidths=0.8, zorder=5)
        if "MCP" in joint_names[j_idx]:
            short_lbl = joint_names[j_idx].split()[0]
            ax_d_rgb.text(
                x + 5, y + 4, short_lbl, color="white", fontsize=7.0, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.5, ec="none"),
            )
    ax_d_rgb.set_title("2. Dynamic Open-Close Skeletal Tracking", fontweight="bold", fontsize=10.5)
    ax_d_rgb.axis("off")

    # -------------------------------------------------------------
    # 3. Dynamic Registered Thermal Signature
    # -------------------------------------------------------------
    d_therm_img = dynamic_meta["rep_thermal_frame"]
    d_vmin = max(20.0, float(np.percentile(d_therm_img, 5)))
    d_vmax = min(38.0, max(35.0, float(np.percentile(d_therm_img, 99.5))))
    im2 = ax_d_therm.imshow(d_therm_img, cmap="inferno", vmin=d_vmin, vmax=d_vmax)
    cbar2 = fig.colorbar(im2, ax=ax_d_therm, fraction=0.046, pad=0.04)
    cbar2.set_label("°C", fontsize=8.5)

    d_therm_lm = dynamic_meta["rep_thermal_landmarks"]
    d_temps = dynamic_meta["joint_temps_c"]
    d_dts = dynamic_meta["joint_delta_t_c"]
    d_palm = dynamic_meta["palm_ref_temp_c"]

    for j_idx, lm_id in enumerate(lm_ids):
        tx, ty = d_therm_lm[lm_id]
        ax_d_therm.scatter(tx, ty, c="cyan", s=38, edgecolors="black", linewidths=0.8, zorder=5)
        dy_off = -7 if ("PIP" in joint_names[j_idx] or "IP" in joint_names[j_idx]) else 9
        ax_d_therm.text(
            tx + 3,
            ty + dy_off,
            f"{d_temps[j_idx]:.1f}° (Δ{d_dts[j_idx]:+.1f})",
            color="cyan",
            fontsize=5.8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.55, ec="none"),
        )
    ax_d_therm.set_title(
        f"3. Dynamic Thermal Signature (Palm: {d_palm:.1f}°C)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax_d_therm.axis("off")

    # -------------------------------------------------------------
    # 4. Dynamic Flexion Range of Motion (Kinematics)
    # -------------------------------------------------------------
    for j_idx in range(6):  # Thumb, Index, Middle MCP/PIP
        ax_d_kin.plot(dynamic_meta["flexion_curves"][j_idx], label=joint_names[j_idx], linewidth=1.7)
    ax_d_kin.set_title("4. Dynamic Flexion Range of Motion Curves (θ)", fontweight="bold", fontsize=10.5)
    ax_d_kin.set_xlabel("Sequence Frame Index (T=30)", fontsize=9)
    ax_d_kin.set_ylabel("Flexion Angle (°)", fontsize=9)
    ax_d_kin.grid(True, linestyle=":", alpha=0.6)
    ax_d_kin.legend(loc="upper right", fontsize=7.2)

    # -------------------------------------------------------------
    # 5. Joint Relative Hyperthermia (ΔT) Comparison: Static vs Dynamic
    # -------------------------------------------------------------
    y_pos = np.arange(len(joint_names))
    bar_width = 0.38
    ax_dt_comp.barh(y_pos - bar_width / 2, s_dts, height=bar_width, color="#3498db", label="Static ΔT", edgecolor="k", linewidth=0.5)
    ax_dt_comp.barh(y_pos + bar_width / 2, d_dts, height=bar_width, color="#e67e22", label="Dynamic ΔT", edgecolor="k", linewidth=0.5)
    ax_dt_comp.axvline(0.8, color="#e74c3c", linestyle="--", linewidth=1.2, label="Synovitis Cutoff (+0.8°C)")
    ax_dt_comp.axvline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)
    ax_dt_comp.set_yticks(y_pos)
    ax_dt_comp.set_yticklabels(joint_names, fontsize=8)
    ax_dt_comp.set_xlabel("Relative Temperature ΔT (°C relative to dorsal palm)", fontsize=9)
    ax_dt_comp.set_title("5. Joint Relative Hyperthermia (ΔT): Static vs Dynamic", fontweight="bold", fontsize=10.5)
    ax_dt_comp.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax_dt_comp.legend(loc="lower right", fontsize=7.5)

    # -------------------------------------------------------------
    # 6. Joint Pathology Probability P(RA): Static vs Dynamic vs Combined
    # -------------------------------------------------------------
    ax_prob_comp.barh(y_pos - bar_width / 2, static_probs, height=bar_width, color="#2980b9", alpha=0.7, label="Static Risk", edgecolor="k", linewidth=0.5)
    ax_prob_comp.barh(y_pos + bar_width / 2, dynamic_probs, height=bar_width, color="#d35400", alpha=0.7, label="Dynamic Risk", edgecolor="k", linewidth=0.5)
    ax_prob_comp.plot(combined_joint_probs, y_pos, "ko-", linewidth=1.8, markersize=5.5, label="Combined Joint Risk")
    ax_prob_comp.axvline(50.0, color="#c0392b", linestyle="--", linewidth=1.3, label="Cutoff (50%)")
    ax_prob_comp.set_yticks(y_pos)
    ax_prob_comp.set_yticklabels(joint_names, fontsize=8)
    ax_prob_comp.set_xlim(0, 100)
    ax_prob_comp.set_xlabel("Pathology Probability (%)", fontsize=9)

    diag_label = "HIGH RISK (Active Synovitis)" if combined_overall >= 50.0 else "NORMAL / LOW RISK"
    ax_prob_comp.set_title(
        f"6. Calibrated P(RA) (Static: {static_overall:.1f}%, Dyn: {dynamic_overall:.1f}% -> Combined: {combined_overall:.1f}%)",
        fontweight="bold",
        fontsize=9.8,
    )
    ax_prob_comp.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax_prob_comp.legend(loc="lower right", fontsize=7.2)

    diag_color = "#c0392b" if combined_overall >= 50.0 else "#27ae60"
    fig.suptitle(
        f"MultiModal Arthritis Diagnostic Assessment — Subject {subj_id} ({hand_name})\n"
        f"Combined Clinical Risk: {combined_overall:.2f}% [{diag_label}]",
        fontsize=14,
        fontweight="bold",
        color=diag_color,
        y=0.99,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_all_17_subjects(config_path="config.yaml", hand="left hand"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    reports_dir = "reports/subject_reports"
    os.makedirs(reports_dir, exist_ok=True)

    loader = SpatiotemporalDataLoader(config)
    engine = KinematicEngine(roi_size=config["dataset"]["roi_size"])

    ckpt_path = "checkpoints/best_arthritis_model.keras"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/final_arthritis_model.keras"
    model = tf.keras.models.load_model(ckpt_path, compile=False) if os.path.exists(ckpt_path) else None

    all_subject_summaries = []

    for subj_idx in range(1, 18):
        subj_id = f"S{subj_idx}"
        static_rgb = f"data/{subj_id}/{hand}/static_RGB.mp4"
        static_npy = f"data/{subj_id}/{hand}/static_Thermal_Radiometric_Sequence.npy"
        dyn_rgb = f"data/{subj_id}/{hand}/open_close_RGB.mp4"
        dyn_npy = f"data/{subj_id}/{hand}/open_close_Thermal_Radiometric_Sequence.npy"

        if not (os.path.exists(static_rgb) and os.path.exists(static_npy) and
                os.path.exists(dyn_rgb) and os.path.exists(dyn_npy)):
            print(f"Skipping {subj_id}: missing required files.")
            continue

        # 1. Process Static Condition
        s_rgb_seq, s_therm_seq = loader.load_synchronized_sequence(static_rgb, static_npy)
        s_rois, s_kins, s_meta = engine.process_sequence(s_rgb_seq, s_therm_seq)
        s_probs, s_overall = compute_calibrated_joint_probabilities(model, s_rois, s_kins, s_meta, config)

        # 2. Process Dynamic Condition
        d_rgb_seq, d_therm_seq = loader.load_synchronized_sequence(dyn_rgb, dyn_npy)
        d_rois, d_kins, d_meta = engine.process_sequence(d_rgb_seq, d_therm_seq)
        d_probs, d_overall = compute_calibrated_joint_probabilities(model, d_rois, d_kins, d_meta, config)

        # 3. Fused Multimodal Joint & Hand Probability
        # Both conditions contribute equally to the final clinical assessment
        combined_joint_probs = 0.50 * s_probs + 0.50 * d_probs
        top3_comb = float(np.mean(np.sort(combined_joint_probs)[-3:]))
        combined_overall = float(0.60 * top3_comb + 0.40 * np.mean(combined_joint_probs))

        diagnosis = "HIGH RISK (Active Synovitis)" if combined_overall >= 50.0 else "NORMAL / LOW RISK"

        dashboard_file = os.path.join(reports_dir, f"diagnostic_report_{subj_id}_combined.png")
        generate_subject_combined_dashboard(
            subj_id=subj_id,
            hand_name=hand.capitalize(),
            static_meta=s_meta,
            static_probs=s_probs,
            static_overall=s_overall,
            dynamic_meta=d_meta,
            dynamic_probs=d_probs,
            dynamic_overall=d_overall,
            combined_joint_probs=combined_joint_probs,
            combined_overall=combined_overall,
            out_path=dashboard_file,
        )

        # Build detailed table for stdout
        joint_names = s_meta["joint_names"]
        joint_details = []
        for j in range(10):
            rom = float(np.ptp(d_meta["flexion_curves"][j]))
            s_ulnar = float(np.median(s_meta["ulnar_curves"][j]))
            d_ulnar = float(np.median(d_meta["ulnar_curves"][j]))
            joint_details.append({
                "joint": joint_names[j],
                "s_temp": float(s_meta["joint_temps_c"][j]),
                "s_dt": float(s_meta["joint_delta_t_c"][j]),
                "s_ulnar": s_ulnar,
                "s_risk": float(s_probs[j]),
                "d_temp": float(d_meta["joint_temps_c"][j]),
                "d_dt": float(d_meta["joint_delta_t_c"][j]),
                "d_rom": rom,
                "d_ulnar": d_ulnar,
                "d_risk": float(d_probs[j]),
                "comb_risk": float(combined_joint_probs[j]),
            })

        subj_summary = {
            "subject": subj_id,
            "hand": hand,
            "static_palm_temp_c": round(float(s_meta["palm_ref_temp_c"]), 2),
            "dynamic_palm_temp_c": round(float(d_meta["palm_ref_temp_c"]), 2),
            "static_mean_joint_temp_c": round(float(np.mean(s_meta["joint_temps_c"])), 2),
            "dynamic_mean_joint_temp_c": round(float(np.mean(d_meta["joint_temps_c"])), 2),
            "static_max_delta_t_c": round(float(np.max(s_meta["joint_delta_t_c"])), 2),
            "dynamic_max_delta_t_c": round(float(np.max(d_meta["joint_delta_t_c"])), 2),
            "dynamic_mean_flexion_rom_deg": round(float(np.mean([np.ptp(d_meta["flexion_curves"][j]) for j in range(2, 10)])), 1),
            "static_overall_risk_pct": round(float(s_overall), 2),
            "dynamic_overall_risk_pct": round(float(d_overall), 2),
            "combined_overall_risk_pct": round(float(combined_overall), 2),
            "diagnosis": diagnosis,
            "dashboard_path": dashboard_file,
            "joint_details": joint_details,
        }
        all_subject_summaries.append(subj_summary)

        # Print detailed report for this subject
        print_subject_full_report(subj_summary)

    # Save summary CSV and JSON
    save_cohort_summaries(all_subject_summaries)
    return all_subject_summaries


def print_subject_full_report(s):
    print("\n" + "=" * 98)
    print(f" MULTIMODAL ARTHRITIS DIAGNOSTIC REPORT: SUBJECT {s['subject']} ({s['hand'].upper()})")
    print("=" * 98)
    print(
        f" Baseline Static Palm Temp  : {s['static_palm_temp_c']:5.2f} °C   | "
        f"Active Dynamic Palm Temp   : {s['dynamic_palm_temp_c']:5.2f} °C"
    )
    print(
        f" Static Mean Joint Temp     : {s['static_mean_joint_temp_c']:5.2f} °C   | "
        f"Dynamic Mean Joint Temp    : {s['dynamic_mean_joint_temp_c']:5.2f} °C"
    )
    print(
        f" Static Max Relative ΔT     : {s['static_max_delta_t_c']:+5.2f} °C   | "
        f"Dynamic Max Relative ΔT    : {s['dynamic_max_delta_t_c']:+5.2f} °C"
    )
    print(
        f" Dynamic Flexion Mean ROM   : {s['dynamic_mean_flexion_rom_deg']:5.1f} °    | "
        f"Kinematic Stiffness Score  : {'NORMAL' if s['dynamic_mean_flexion_rom_deg'] >= 40.0 else 'ELEVATED STIFFNESS'}"
    )
    print("-" * 98)
    print(
        f" {'JOINT NAME':<11} | "
        f"{'STATIC TEMP':<11} | {'STATIC ΔT':<9} | {'STAT RISK':<9} | "
        f"{'DYN TEMP':<9} | {'DYN ΔT':<8} | {'ROM (°)':<7} | {'DYN RISK':<8} | "
        f"{'COMB RISK'}"
    )
    print("-" * 98)
    for jd in s["joint_details"]:
        print(
            f" {jd['joint']:<11} | "
            f"{jd['s_temp']:5.2f} °C   | {jd['s_dt']:+5.2f} °C | {jd['s_risk']:5.1f} %  | "
            f"{jd['d_temp']:5.2f}°C | {jd['d_dt']:+5.2f}°C | {jd['d_rom']:5.1f}° | {jd['d_risk']:5.1f} % | "
            f"{jd['comb_risk']:5.1f} %"
        )
    print("-" * 98)
    print(
        f" OVERALL RISK EVALUATION: Static={s['static_overall_risk_pct']:.2f}% | "
        f"Dynamic={s['dynamic_overall_risk_pct']:.2f}% | "
        f"COMBINED SUBJECT RISK={s['combined_overall_risk_pct']:.2f}%"
    )
    print(f" CLINICAL VERDICT       : {s['diagnosis']}")
    print(f" Visual Dashboard       : {s['dashboard_path']}")
    print("=" * 98 + "\n")


def save_cohort_summaries(summaries):
    csv_path = "reports/all_17_subjects_dual_condition_summary.csv"
    json_path = "reports/all_17_subjects_dual_condition_summary.json"

    fieldnames = [
        "subject",
        "hand",
        "static_palm_temp_c",
        "dynamic_palm_temp_c",
        "static_mean_joint_temp_c",
        "dynamic_mean_joint_temp_c",
        "static_max_delta_t_c",
        "dynamic_max_delta_t_c",
        "dynamic_mean_flexion_rom_deg",
        "static_overall_risk_pct",
        "dynamic_overall_risk_pct",
        "combined_overall_risk_pct",
        "diagnosis",
        "dashboard_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {k: s[k] for k in fieldnames}
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Saved cohort dual-condition summary CSV: {csv_path}")
    print(f"Saved cohort dual-condition summary JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--hand", default="left hand", help="Hand to evaluate (default: left hand)")
    args = parser.parse_args()

    run_all_17_subjects(args.config, args.hand)
