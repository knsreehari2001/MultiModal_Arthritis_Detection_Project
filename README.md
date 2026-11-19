# Multi-Modal Arthritis Detection (Spatiotemporal Architecture)

## Overview
This project implements a zero-contact, multi-modal deep learning pipeline for the early detection of Rheumatoid Arthritis (RA). It utilizes synchronized RGB video and thermal radiometric sequences (captured via FLIR One Pro) to analyze both structural deformities (kinematics) and metabolic heat signatures (inflammation) over time.

## System Architecture
1. **Data Loader:** Parses 640x480 RGB `.mp4` and `.npy` radiometric temperature arrays, normalizing variable frame rates to a fixed temporal window `T`.
2. **Spatial Registration (OpenCV):** Applies a planar homography matrix to align the thermal data to the RGB coordinate space.
3. **Topological Tracking (MediaPipe):** Extracts 21 3D hand landmarks, computes joint deviation angles/velocities, and dynamically crops 64x64 Regions of Interest (ROIs) strictly over the MCP and PIP joints.
4. **Dual-Stream Neural Fusion (TensorFlow/Keras):**
   - **Stream A (Spatiotemporal):** A 3D CNN processes the `[T, 64, 64, 4]` tensor to evaluate localized swelling and heat gradients over time.
   - **Stream B (Kinematic):** An LSTM network evaluates the `[T, 10]` kinematic vector to measure joint stiffness and range of motion.
   - **Late Fusion:** Embeddings are concatenated to output a localized joint-level diagnostic probability.

## Project Structure
```
.
├── data/                   # Place your FLIR One dataset here
├── config.yaml             # Hyperparameters and paths
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_loader.py      # Spatiotemporal sequence parsing
│   ├── homography.py       # Cross-modal alignment
│   ├── kinematics.py       # MediaPipe joint extraction & cropping
│   ├── models.py           # 3D CNN + LSTM architecture
│   ├── train.py            # Training loop and focal loss optimization
│   └── inference.py        # Diagnostic dashboard & prediction
```

## Setup & Installation
1. Ensure you have Python 3.9+ installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the Dataset:
   Ensure your dataset matches the structure specified in the FLIR One documentation (e.g., `data/S1/left_hand/static_RGB.mp4`, etc.).

## Execution Manual

### 1. Configuration
Edit `config.yaml` to adjust the `sequence_length` (T), `roi_size`, `batch_size`, and `epochs`.

### 2. Training the Model
To initiate the training pipeline from scratch:
```bash
python src/train.py --config config.yaml
```
The script will automatically parse the video streams, extract the tensors, and train the dual-stream network. Model checkpoints will be saved to `checkpoints/`.

### 3. Running Inference
To run a diagnostic evaluation on a new patient recording:
```bash
python src/inference.py --rgb path/to/video.mp4 --thermal path/to/radiometric.npy
```
This will output the localized probabilities for the MCP and PIP joints and generate the terminal dashboard.
