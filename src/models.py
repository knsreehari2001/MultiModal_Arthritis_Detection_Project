import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    GlobalAveragePooling3D,
    BatchNormalization,
    Dense,
    Concatenate,
    LSTM,
    Dropout,
)
from tensorflow.keras.regularizers import l2


def build_dual_stream_model(seq_len=30, roi_size=64, channels=4, kin_features=4):
    """
    Constructs the stabilized Dual-Stream Spatiotemporal Network:
      - Stream A: 3D CNN with BatchNormalization + GlobalAveragePooling3D (prevents
        the 4.2M-parameter Flatten() gradient explosion seen in earlier training curves).
      - Stream B: Temporal LSTM processing normalized 3D flexion, velocity, ulnar drift,
        and task-context features.
      - Late Fusion: Outputs calibrated joint-level pathology probability P(RA).
    """
    reg = l2(1e-4)

    # Stream A: Spatiotemporal 3D CNN (Morphology + Self-Referenced ΔT)
    img_input = Input(shape=(seq_len, roi_size, roi_size, channels), name="rgb_thermal_roi")

    x = Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation="relu", kernel_regularizer=reg)(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling3D()(x)

    cnn_embedding = Dense(64, activation="relu", kernel_regularizer=reg)(x)
    cnn_embedding = Dropout(0.3)(cnn_embedding)

    # Stream B: Temporal Kinematic LSTM
    kin_input = Input(shape=(seq_len, kin_features), name="kinematic_angles")
    y = BatchNormalization()(kin_input)
    y = LSTM(32, return_sequences=False, dropout=0.2)(y)
    lstm_embedding = Dense(32, activation="relu", kernel_regularizer=reg)(y)

    # Late Fusion
    merged = Concatenate()([cnn_embedding, lstm_embedding])
    z = Dense(64, activation="relu", kernel_regularizer=reg)(merged)
    z = Dropout(0.3)(z)
    output = Dense(1, activation="sigmoid", name="arthritis_probability")(z)

    model = tf.keras.Model(inputs=[img_input, kin_input], outputs=output)
    return model