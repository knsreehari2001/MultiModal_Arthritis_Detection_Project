import os
import sys
import argparse
import yaml
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


def run_batch_inference(config_path="config.yaml", output_csv="reports/all_datasets_diagnostic_summary.csv"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    loader = SpatiotemporalDataLoader(config)
    engine = KinematicEngine(roi_size=config["dataset"]["roi_size"])

    recordings = loader.discover_recordings()
    if not recordings:
        print("No recordings found in dataset!")
        return

    # Natural sorting helper for subject IDs (e.g. S1 before S10)
    def sort_key(rec):
        parts = rec["id"].split("_")
        subj_str = parts[0]
        subj_num = int(subj_str[1:]) if subj_str.startswith("S") and subj_str[1:].isdigit() else 999
        return (subj_num, rec["id"])

    recordings = sorted(recordings, key=sort_key)
    print(f"\nDiscovered {len(recordings)} paired recordings across all subjects.")

    ckpt_path = "checkpoints/best_arthritis_model.keras"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/final_arthritis_model.keras"
    model = tf.keras.models.load_model(ckpt_path, compile=False) if os.path.exists(ckpt_path) else None
    if model is not None:
        print(f"Loaded trained neural network from {ckpt_path}\n")
    else:
        print("Warning: Trained checkpoint not found, using rule-based calibrated scoring.\n")

    results = []

    print("=" * 105)
    print(f"{'REC ID':<28} | {'TASK':<12} | {'PALM (°C)':<9} | {'MAX ΔT':<8} | {'MAX ULNAR':<9} | {'RISK %':<7} | {'DIAGNOSIS'}")
    print("=" * 105)

    for idx, rec in enumerate(recordings, 1):
        rec_id = rec["id"]
        rgb_path = rec["rgb_path"]
        npy_path = rec["npy_path"]

        # Parse subject, hand, task from id
        parts = rec_id.split("_")
        subject = parts[0]
        hand = parts[1] if len(parts) > 1 else "unknown"
        task = "_".join(parts[2:]) if len(parts) > 2 else "unknown"

        try:
            rgb_seq, therm_seq = loader.load_synchronized_sequence(rgb_path, npy_path)
            joint_rois, joint_kins, meta = engine.process_sequence(rgb_seq, therm_seq)
            joint_probs, overall_prob = compute_calibrated_joint_probabilities(
                model, joint_rois, joint_kins, meta, config
            )

            palm_temp = float(meta["palm_ref_temp_c"])
            joint_temps = meta["joint_temps_c"]
            delta_ts = meta["joint_delta_t_c"]
            ulnar_devs = np.median(meta["ulnar_curves"], axis=1)

            mean_joint_temp = float(np.mean(joint_temps))
            max_joint_temp = float(np.max(joint_temps))
            max_delta_t = float(np.max(delta_ts))
            mean_delta_t = float(np.mean(delta_ts))
            max_ulnar = float(np.max(ulnar_devs))

            top_joint_idx = int(np.argmax(joint_probs))
            top_joint_name = meta["joint_names"][top_joint_idx]
            top_joint_risk = float(joint_probs[top_joint_idx])

            diagnosis = "HIGH RISK (Synovitis)" if overall_prob >= 50.0 else "NORMAL / LOW RISK"

            row = {
                "id": rec_id,
                "subject": subject,
                "hand": hand,
                "task": task,
                "palm_ref_temp_c": round(palm_temp, 2),
                "mean_joint_temp_c": round(mean_joint_temp, 2),
                "max_joint_temp_c": round(max_joint_temp, 2),
                "max_delta_t_c": round(max_delta_t, 2),
                "mean_delta_t_c": round(mean_delta_t, 2),
                "max_ulnar_dev_deg": round(max_ulnar, 1),
                "overall_risk_pct": round(overall_prob, 2),
                "diagnosis": diagnosis,
                "top_affected_joint": top_joint_name,
                "top_joint_risk_pct": round(top_joint_risk, 2),
            }
            results.append(row)

            task_display = "Dynamic" if meta["is_dynamic"] else "Static"
            print(
                f"[{idx:02d}/{len(recordings):02d}] {rec_id:<23} | {task_display:<12} | "
                f"{palm_temp:6.2f} °C | {max_delta_t:+6.2f}°C | {max_ulnar:6.1f}° | "
                f"{overall_prob:5.2f}% | {diagnosis}"
            )

        except Exception as e:
            print(f"[{idx:02d}/{len(recordings):02d}] {rec_id:<23} | ERROR: {e}")

    print("=" * 105)

    # Save to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved complete diagnostic summary CSV to: {output_csv}")

    # Generate Overview Cohort Visualization
    generate_cohort_visualization(results, "reports/cohort_diagnostic_overview.png")

    # Subject-level summary
    print_subject_summary(results)


def generate_cohort_visualization(results, out_path):
    if not results:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    rec_ids = [r["id"] for r in results]
    risks = [r["overall_risk_pct"] for r in results]
    palm_temps = [r["palm_ref_temp_c"] for r in results]
    max_dts = [r["max_delta_t_c"] for r in results]
    max_ulnars = [r["max_ulnar_dev_deg"] for r in results]
    tasks = [r["task"] for r in results]

    # Panel 1: Overall Risk Probability across all recordings
    colors = ["#e74c3c" if risk >= 50.0 else "#27ae60" for risk in risks]
    x_indices = np.arange(len(results))
    ax1.bar(x_indices, risks, color=colors, width=0.8, edgecolor="black", linewidth=0.5)
    ax1.axhline(50.0, color="#c0392b", linestyle="--", linewidth=1.5, label="Clinical Threshold (50%)")
    ax1.set_title("Cohort Joint Pathology Probability P(RA) Across All Recordings", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recording Index", fontsize=10)
    ax1.set_ylabel("Overall Hand Risk (%)", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel 2: Palm vs Joint Temperatures
    mean_joint_temps = [r["mean_joint_temp_c"] for r in results]
    ax2.scatter(palm_temps, mean_joint_temps, c="#2980b9", edgecolors="k", s=45, alpha=0.85, label="Recordings")
    diag_line = np.linspace(min(palm_temps), max(palm_temps), 50)
    ax2.plot(diag_line, diag_line, "k--", alpha=0.5, label="Isothermal Line (Joint = Palm)")
    ax2.set_title("Mean Articular Temperature vs Palm Reference Temperature", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Palm Reference Temperature (°C)", fontsize=10)
    ax2.set_ylabel("Mean Articular Joint Temperature (°C)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()

    # Panel 3: Max Joint ΔT Distribution
    ax3.hist(max_dts, bins=16, color="#8e44ad", edgecolor="black", alpha=0.75)
    ax3.axvline(0.8, color="#e74c3c", linestyle="--", linewidth=1.5, label="Synovitis Cutoff (+0.8°C)")
    ax3.set_title("Distribution of Peak Joint Relative Hyperthermia (Max ΔT)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Max ΔT (°C relative to dorsal palm)", fontsize=10)
    ax3.set_ylabel("Number of Recordings", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend()

    # Panel 4: Subject-wise Mean Risk
    subjects = sorted(list(set(r["subject"] for r in results)), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)
    subj_means = []
    for s in subjects:
        s_risks = [r["overall_risk_pct"] for r in results if r["subject"] == s]
        subj_means.append(np.mean(s_risks))

    ax4.bar(subjects, subj_means, color="#16a085", edgecolor="black", width=0.65)
    ax4.set_title("Mean Arthritis Risk Probability by Subject (S1 - S17)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Subject ID", fontsize=10)
    ax4.set_ylabel("Mean Hand Risk (%)", fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved cohort diagnostic overview visualization to: {out_path}\n")


def print_subject_summary(results):
    if not results:
        return

    subjects = sorted(list(set(r["subject"] for r in results)), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)

    print("\n" + "=" * 80)
    print(" SUBJECT-LEVEL AGGREGATE DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"{'SUBJECT':<10} | {'RECS':<6} | {'MEAN PALM (°C)':<15} | {'MAX ΔT (°C)':<12} | {'MEAN RISK':<11} | {'DIAGNOSIS'}")
    print("-" * 80)

    total_low = 0
    total_high = 0

    for s in subjects:
        s_rows = [r for r in results if r["subject"] == s]
        mean_palm = np.mean([r["palm_ref_temp_c"] for r in s_rows])
        max_dt = np.max([r["max_delta_t_c"] for r in s_rows])
        mean_risk = np.mean([r["overall_risk_pct"] for r in s_rows])
        diag = "HIGH RISK" if mean_risk >= 50.0 else "NORMAL / LOW RISK"

        if mean_risk >= 50.0:
            total_high += 1
        else:
            total_low += 1

        print(
            f" {s:<9} | {len(s_rows):<6} | {mean_palm:11.2f} °C   | {max_dt:+8.2f} °C    | "
            f"{mean_risk:7.2f} %  | {diag}"
        )

    print("=" * 80)
    print(f" Cohort Breakdown: {total_low} Normal/Low Risk Subjects | {total_high} High Risk Subjects")
    print(f" Total Recordings Processed: {len(results)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference across all paired recordings in dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output_csv", default="reports/all_datasets_diagnostic_summary.csv", help="Output summary CSV")
    args = parser.parse_args()

    run_batch_inference(args.config, args.output_csv)
