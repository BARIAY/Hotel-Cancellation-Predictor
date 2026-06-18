"""
Script d'entraînement autonome pour le modèle ANN (Réseau de Neurones Artificiels).
Ce script charge les données brutes, applique le pré-traitement modulaire, équilibre
le dataset avec SMOTE, entraîne l'ANN et sauvegarde tous les artefacts requis dans `models/`.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Importation des fonctions de preprocessing modulaires
from utils.preprocessing import (
    drop_leakage_columns,
    handle_missing,
    clean_inconsistencies,
    engineer_features,
    encode_categoricals,
    cap_outliers_iqr,
    OUTLIER_COLS
)

# Configuration globale
DATA_PATH = 'assets/hotel_bookings.csv'
MODELS_DIR = 'models'
RANDOM_STATE = 42

def build_ann(input_dim, dropout_rate=0.3):
    """Architecture ANN à 3 couches cachées avec BatchNorm + Dropout."""
    model = Sequential([
        # Couche dense 1 : 128 neurones
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Couche dense 2 : 64 neurones
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        # Couche dense 3 : 32 neurones
        Dense(32, activation='relu'),
        Dropout(0.2),

        # Couche de sortie : 1 neurone sigmoid (probabilité [0, 1])
        Dense(1, activation='sigmoid')
    ])
    return model

def main():
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE ANN — HOTEL BOOKING CANCELLATION")
    print("=" * 70)

    # 1. Vérifier la présence du dataset
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Le fichier '{DATA_PATH}' est introuvable !")
        print("Veuillez télécharger le dataset depuis Kaggle et le placer dans le dossier 'assets/'.")
        print("Lien du dataset : https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand")
        return

    # S'assurer que le dossier models existe
    Path(MODELS_DIR).mkdir(exist_ok=True)

    # Configuration de reproductibilité
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # 2. Chargement du dataset
    print(f"\n[1/7] Chargement des données brutes depuis '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Dimensions initiales : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    # 3. Pré-traitement modulaire
    print("\n[2/7] Application du pipeline de pré-traitement...")
    df_clean = drop_leakage_columns(df)
    print(f"      - Suppression leakage : {df_clean.shape}")
    
    df_clean = handle_missing(df_clean)
    print(f"      - Gestion des NaN : {df_clean.shape}")
    
    df_clean = clean_inconsistencies(df_clean)
    print(f"      - Nettoyage des incohérences : {df_clean.shape}")
    
    df_clean = engineer_features(df_clean)
    print(f"      - Feature engineering : {df_clean.shape}")
    
    df_encoded, country_encoder = encode_categoricals(df_clean, fit=True)
    print(f"      - Encodage catégoriel (OHE) : {df_encoded.shape}")
    
    df_encoded = cap_outliers_iqr(df_encoded, columns=OUTLIER_COLS, multiplier=3.0)
    print(f"      - Traitement des outliers (Gini IQR) : {df_encoded.shape}")

    # 4. Séparation X/y et sauvegarde des colonnes
    print("\n[3/7] Séparation des features/cible et alignement des colonnes...")
    y = df_encoded['is_canceled'].astype(int)
    X = df_encoded.drop(columns=['is_canceled'])
    feature_columns = list(X.columns)
    
    feature_columns_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    joblib.dump(feature_columns, feature_columns_path)
    print(f"      - Sauvegarde des colonnes ({len(feature_columns)} features) -> {feature_columns_path}")

    # 5. Split train/test stratifié
    print("\n[4/7] Séparation Train / Test stratifiée (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"      - Train : {X_train.shape[0]:,} lignes")
    print(f"      - Test  : {X_test.shape[0]:,} lignes")

    # 6. Scaling (fit sur train uniquement)
    print("\n[5/7] Normalisation (StandardScaler) fit sur train uniquement...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    country_encoder_path = os.path.join(MODELS_DIR, 'label_encoder_country.pkl')
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(country_encoder, country_encoder_path)
    print(f"      - Sauvegarde du Scaler -> {scaler_path}")
    print(f"      - Sauvegarde du Country Encoder -> {country_encoder_path}")

    # 7. Rééchantillonnage SMOTE (sur train uniquement)
    print("\n[6/7] Rééquilibrage des classes par SMOTE sur le train set...")
    print(f"      - Avant SMOTE : {pd.Series(y_train).value_counts().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"      - Après SMOTE : {pd.Series(y_train_resampled).value_counts().to_dict()}")

    # 8. Entraînement de l'ANN
    print("\n[7/7] Construction et compilation du réseau de neurones ANN...")
    ann_model = build_ann(input_dim=X_train_resampled.shape[1])
    
    ann_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nDébut de l'entraînement de l'ANN (max 100 epochs)...")
    history = ann_model.fit(
        X_train_resampled, y_train_resampled,
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Sauvegarde du modèle ANN
    model_path = os.path.join(MODELS_DIR, 'ann_model.h5')
    ann_model.save(model_path)
    print(f"\n[OK] Modèle ANN sauvegardé -> {model_path}")

    # Sauvegarde de l'historique d'entraînement
    history_to_save = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(MODELS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_to_save, f, indent=2)
    print(f"[OK] Historique d'entraînement sauvegardé -> {history_path}")

    # 9. Évaluation finale
    print("\nÉvaluation sur le test set...")
    y_pred_proba = ann_model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("-" * 50)
    print("MÉTRIQUES FINALES DE L'ANN SUR LE TEST SET :")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("-" * 50)

    # Sauvegarde du tableau de synthèse des performances (uniquement ANN)
    ann_results = pd.DataFrame([{
        'Model': 'ANN (Reseau de Neurones)',
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4),
        'Architecture': '128-64-32 + BatchNorm + Dropout',
        'Epochs': len(history.history['loss']),
    }])
    
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    ann_results.to_csv(comparison_path, index=False)
    print(f"[OK] Fichier de comparaison sauvegardé -> {comparison_path}")
    print("\nEntraînement complété avec succès ! Tous les artefacts sont prêts.")

if __name__ == '__main__':
    main()
