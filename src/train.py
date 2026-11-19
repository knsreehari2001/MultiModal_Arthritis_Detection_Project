import os
import sys
import argparse
import yaml
import numpy as np

# Use Agg backend for headless execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_loader import SpatiotemporalDataLoader
from kinematics import KinematicEngine
from models import build_dual_stream_model


def get_compiled_model(config):
    model = build_dual_stream_model(
        seq_len=config["dataset"]["sequence_length"],
        roi_size=config["dataset"]["roi_size"],
        channels=config["model"]["channels"],
        kin_features=config["model"]["kinematic_features"],
    )

    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True,
        alpha=config["training"]["focal_loss_alpha"],
        gamma=config["training"]["focal_loss_gamma"],
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"],
        clipnorm=1.0,  # Prevents gradient explosion
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_joint_level_dataset(config, max_recordings=None):
    loader = SpatiotemporalDataLoader(config)
    engine = KinematicEngine(roi_size=config["dataset"]["roi_size"])
    recordings = loader.discover_recordings()

    if not recordings:
        raise RuntimeError(f"No valid paired recordings found in {config['dataset']['root_dir']}")

    limit = max_recordings or config.get("dataset", {}).get("max_recordings", None)
    if limit:
        recordings = recordings[:limit]

    rng = np.random.default_rng(config["training"]["random_seed"])
    X_rois, X_kins, y_labels, groups = [], [], [], []

    print(f"Ingesting {len(recordings)} synchronized recordings. Extracting 10-joint spatiotemporal tensors...")
    for rec in recordings:
        print(f"  -> Processing {rec['id']} ...")
        rgb_seq, therm_seq = loader.load_synchronized_sequence(rec["rgb_path"], rec["npy_path"])
        joint_rois, joint_kins, meta = engine.process_sequence(rgb_seq, therm_seq)

        for j_idx in range(10):
            roi_j = joint_rois[j_idx]
            kin_j = joint_kins[j_idx]
            delta_t_j = meta["joint_delta_t_c"][j_idx]
            ulnar_j = float(np.median(meta["ulnar_curves"][j_idx]))

            # Ground-Truth Clinical Heuristic:
            # Healthy baseline FLIR One subjects have ΔT < 1.0°C and normal ulnar alignment (Label = 0)
            is_pathological = (delta_t_j >= config["clinical_thresholds"]["delta_t_synovitis_c"]) or (
                ulnar_j >= config["clinical_thresholds"]["ulnar_drift_deg"]
            )
            label = 1.0 if is_pathological else 0.0

            X_rois.append(roi_j)
            X_kins.append(kin_j)
            y_labels.append(label)
            groups.append(rec["id"])

            # Generate a paired physiologically calibrated positive synovitis sample (Label = 1)
            # so the model learns true localized ΔT hotspots & ulnar drift rather than baseline body heat
            if label == 0.0:
                syn_roi, syn_kin = loader.synthesize_arthritic_joint_sample(roi_j, kin_j, rng)
                X_rois.append(syn_roi)
                X_kins.append(syn_kin)
                y_labels.append(1.0)
                groups.append(rec["id"] + "_syn")

    return (
        np.array(X_rois, dtype=np.float32),
        np.array(X_kins, dtype=np.float32),
        np.array(y_labels, dtype=np.float32),
    )


def plot_training_curves(history, out_path="reports/training_curves.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)

    epochs = range(1, len(history.history["loss"]) + 1)
    ax1.plot(epochs, history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history.history["val_loss"], "--", label="Val Loss", linewidth=2)
    ax1.set_title("Focal Loss Convergence", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Focal Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history.history["accuracy"], label="Train Accuracy", color="forestgreen", linewidth=2)
    ax2.plot(epochs, history.history["val_accuracy"], "--", label="Val Accuracy", color="firebrick", linewidth=2)
    if "val_auc" in history.history:
        ax2.plot(epochs, history.history["val_auc"], ":", label="Val AUC", color="royalblue", linewidth=2.2)
    ax2.set_title("Diagnostic Performance Metrics", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max_recordings", type=int, default=None, help="Override max recordings limit")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    tf.random.set_seed(config["training"]["random_seed"])
    np.random.seed(config["training"]["random_seed"])

    X_rois, X_kins, y = build_joint_level_dataset(config, max_recordings=args.max_recordings)

    # Stratified 80/20 train-validation split guarantees both classes exist in validation
    X_roi_tr, X_roi_val, X_kin_tr, X_kin_val, y_tr, y_val = train_test_split(
        X_rois,
        X_kins,
        y,
        test_size=0.20,
        stratify=y,
        random_state=config["training"]["random_seed"],
    )

    print(f"Dataset summary: Train={len(y_tr)} joints, Val={len(y_val)} joints")
    model = get_compiled_model(config)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/best_arthritis_model.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        [X_roi_tr, X_kin_tr],
        y_tr,
        validation_data=([X_roi_val, X_kin_val], y_val),
        batch_size=config["dataset"]["batch_size"],
        epochs=config["training"]["epochs"],
        callbacks=[ckpt_cb],
        verbose=1,
    )

    model.save("checkpoints/final_arthritis_model.keras")
    plot_training_curves(history)


if __name__ == "__main__":
    main()
