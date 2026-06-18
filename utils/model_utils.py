"""
Chargement du modèle ANN et des artefacts associés.
Toutes les fonctions sont cachées avec @st.cache_resource / @st.cache_data
pour éviter de recharger à chaque interaction.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

MODELS_DIR = 'models'

MODEL_PATH        = os.path.join(MODELS_DIR, 'ann_model.h5')
SCALER_PATH       = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODER_PATH      = os.path.join(MODELS_DIR, 'label_encoder_country.pkl')
FEATURES_PATH     = os.path.join(MODELS_DIR, 'feature_columns.pkl')
COMPARISON_PATH   = os.path.join(MODELS_DIR, 'model_comparison.csv')
HISTORY_PATH      = os.path.join(MODELS_DIR, 'training_history.json')


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Charge tous les artefacts entraînés en une seule fois.
    Retourne un dict pour accès clair par nom.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    # Classes personnalisées pour intercepter et supprimer quantization_config
    # qui cause des erreurs de désérialisation dans certaines versions de Keras/TensorFlow.
    class PatchedDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    class PatchedBatchNormalization(tf.keras.layers.BatchNormalization):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    class PatchedDropout(tf.keras.layers.Dropout):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    custom_objects = {
        'Dense': PatchedDense,
        'BatchNormalization': PatchedBatchNormalization,
        'Dropout': PatchedDropout,
    }

    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    except Exception:
        # Fallback de secours si la désérialisation personnalisée échoue
        model = tf.keras.models.load_model(MODEL_PATH)

    return {
        'model':           model,
        'scaler':          joblib.load(SCALER_PATH),
        'country_encoder': joblib.load(ENCODER_PATH),
        'feature_columns': joblib.load(FEATURES_PATH),
    }


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Charge le tableau comparatif des modèles produit par le notebook."""
    if not os.path.exists(COMPARISON_PATH):
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_training_history() -> dict:
    """Charge l'historique d'entraînement (loss / accuracy par epoch)."""
    if not os.path.exists(HISTORY_PATH):
        return {}
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict(model, X_scaled: np.ndarray, threshold: float = 0.5):
    """
    Prédit pour un batch déjà preprocessé.
    Retourne (predictions_binaires, probabilités).
    """
    probas = model.predict(X_scaled, verbose=0).flatten()
    preds = (probas >= threshold).astype(int)
    return preds, probas


def get_artifacts_status() -> dict:
    """Indique quels artefacts sont présents/absents (pour messages utilisateur)."""
    return {
        'model':         os.path.exists(MODEL_PATH),
        'scaler':        os.path.exists(SCALER_PATH),
        'encoder':       os.path.exists(ENCODER_PATH),
        'features':      os.path.exists(FEATURES_PATH),
        'comparison':    os.path.exists(COMPARISON_PATH),
        'history':       os.path.exists(HISTORY_PATH),
    }
