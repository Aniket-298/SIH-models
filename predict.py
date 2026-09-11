import os

import joblib
import numpy as np
import tensorflow as tf

from feature_extraction import (
    extract_features,
    extract_mel_spectrogram,
    normalize_mel
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

XGB_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_model.keras"
)


# ============================================================
# ENSEMBLE SETTINGS
# ============================================================

XGB_WEIGHT = 0.7
CNN_WEIGHT = 0.3

THRESHOLD = 0.45


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

xgb_model = joblib.load(
    XGB_MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

cnn_model = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

print("Models loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_audio(file_path):

    # ========================================================
    # XGBOOST BRANCH
    # ========================================================

    features = extract_features(
        file_path
    )

    if features is None:

        raise ValueError(
            "Could not extract XGBoost features."
        )


    # Shape:
    # (40,)
    #
    # We need:
    # (1, 40)

    features = features.reshape(
        1,
        -1
    )


    # Apply the EXACT scaler used during training
    features_scaled = scaler.transform(
        features
    )


    # Probability of class 1 = FAKE
    xgb_probability = xgb_model.predict_proba(
        features_scaled
    )[0][1]


    # ========================================================
    # CNN BRANCH
    # ========================================================

    mel = extract_mel_spectrogram(
        file_path
    )

    if mel is None:

        raise ValueError(
            "Could not extract Mel spectrogram."
        )


    # Same normalization used during CNN training
    mel = normalize_mel(
        mel
    )


    # At this point:
    #
    # (128, 256)
    #
    # Add channel:
    #
    # (128, 256, 1)

    mel = mel[..., np.newaxis]


    # Add batch dimension:
    #
    # (1, 128, 256, 1)

    mel = np.expand_dims(
        mel,
        axis=0
    )


    # CNN output is sigmoid probability
    cnn_probability = cnn_model.predict(
        mel,
        verbose=0
    )[0][0]


    # ========================================================
    # ENSEMBLE
    # ========================================================

    final_probability = (
        XGB_WEIGHT * xgb_probability
        +
        CNN_WEIGHT * cnn_probability
    )


    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    if final_probability >= THRESHOLD:

        prediction = "AI-GENERATED"

    else:

        prediction = "REAL"


    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return {

        "prediction": prediction,

        "probability": float(
            final_probability
        ),

        "xgb_probability": float(
            xgb_probability
        ),

        "cnn_probability": float(
            cnn_probability
        ),

        "threshold": THRESHOLD

    }