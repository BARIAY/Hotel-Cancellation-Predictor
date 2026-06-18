"""
Fonctions de preprocessing reproduisant à l'identique le pipeline du notebook.
Doivent être strictement cohérentes avec les transformations appliquées
avant l'entraînement, sinon le modèle reçoit des features incompatibles.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LEAK_COLS = ['reservation_status', 'reservation_status_date', 'company']

LOW_CARDINALITY_COLS = [
    'hotel', 'meal', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type', 'deposit_type',
    'customer_type', 'arrival_date_month'
]

OUTLIER_COLS = ['lead_time', 'adr', 'total_stays', 'booking_changes']


# -----------------------------------------------------------------------------
# Étapes individuelles du pipeline
# -----------------------------------------------------------------------------

def drop_leakage_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Supprime les variables qui fuient l'information cible."""
    return data.drop(columns=[c for c in LEAK_COLS if c in data.columns])


def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes selon la stratégie validée."""
    out = data.copy()
    if 'children' in out.columns:
        out['children'] = out['children'].fillna(0)
    if 'country' in out.columns:
        out['country'] = out['country'].fillna('Unknown')
    if 'agent' in out.columns:
        out['agent'] = out['agent'].fillna(0)
    return out


def clean_inconsistencies(data: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes incohérentes (ne s'applique qu'à l'entraînement)."""
    out = data.copy()
    if {'adults', 'children', 'babies'}.issubset(out.columns):
        out = out[~((out['adults'] == 0) & (out['children'] == 0) & (out['babies'] == 0))]
    if 'adr' in out.columns:
        out = out[(out['adr'] >= 0) & (out['adr'] <= 5000)]
    return out.reset_index(drop=True)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Crée les features dérivées identifiées comme utiles."""
    out = data.copy()
    out['total_stays'] = out['stays_in_week_nights'] + out['stays_in_weekend_nights']
    out['total_guests'] = out['adults'] + out['children'] + out['babies']
    out['has_special_requests'] = (out['total_of_special_requests'] > 0).astype(int)
    out['room_type_changed'] = (out['reserved_room_type'] != out['assigned_room_type']).astype(int)
    
    # Mapping robuste des mois anglais pour éviter les erreurs de locale sur le parsing de date
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    months_numeric = out['arrival_date_month'].map(month_map).fillna(1).astype(int)
    
    out['arrival_weekday'] = pd.to_datetime(
        pd.DataFrame({
            'year': out['arrival_date_year'],
            'month': months_numeric,
            'day': out['arrival_date_day_of_month']
        }), errors='coerce'
    ).dt.dayofweek.fillna(0).astype(int)
    
    return out


def encode_categoricals(data: pd.DataFrame, country_encoder=None, fit: bool = True, feature_columns=None):
    """
    Encode les variables catégorielles.
    - country : LabelEncoder (haute cardinalité)
    - autres  : One-Hot Encoding

    Retourne (DataFrame encodé, encoder country).
    En mode fit=False, gère les pays inconnus en leur attribuant -1.
    """
    out = data.copy()

    if 'country' in out.columns:
        if fit:
            country_encoder = LabelEncoder()
            out['country_encoded'] = country_encoder.fit_transform(out['country'].astype(str))
        else:
            known = set(country_encoder.classes_)
            out['country_encoded'] = out['country'].astype(str).apply(
                lambda x: country_encoder.transform([x])[0] if x in known else -1
            )
        out = out.drop(columns=['country'])

    if fit:
        cat_cols = [c for c in LOW_CARDINALITY_COLS if c in out.columns]
        out = pd.get_dummies(out, columns=cat_cols, drop_first=True)
        bool_cols = out.select_dtypes(include=['bool']).columns
        out[bool_cols] = out[bool_cols].astype(int)
    else:
        # Encodage robuste one-hot pour l'inférence (gère ligne unique / catégories manquantes)
        cat_cols = [c for c in LOW_CARDINALITY_COLS if c in out.columns]
        if feature_columns is not None:
            for col in cat_cols:
                # Récupère toutes les colonnes one-hot issues de feature_columns pour cette variable
                dummy_cols = [f for f in feature_columns if f.startswith(f"{col}_")]
                for dummy_col in dummy_cols:
                    out[dummy_col] = 0
                    category_val = dummy_col[len(col) + 1:]
                    out.loc[out[col].astype(str) == category_val, dummy_col] = 1
                out = out.drop(columns=[col])
        else:
            # Fallback si feature_columns n'est pas passé
            out = pd.get_dummies(out, columns=cat_cols, drop_first=True)
            bool_cols = out.select_dtypes(include=['bool']).columns
            out[bool_cols] = out[bool_cols].astype(int)

    return out, country_encoder


def cap_outliers_iqr(data: pd.DataFrame, columns=None, multiplier: float = 3.0) -> pd.DataFrame:
    """Cappe les valeurs au-delà de multiplier x IQR."""
    out = data.copy()
    columns = columns or OUTLIER_COLS
    for col in columns:
        if col not in out.columns:
            continue
        Q1 = out[col].quantile(0.25)
        Q3 = out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        out[col] = out[col].clip(lower=lower, upper=upper)
    return out


# -----------------------------------------------------------------------------
# Pipeline complet pour l'inférence
# -----------------------------------------------------------------------------

def preprocess_for_prediction(
    data: pd.DataFrame,
    scaler,
    country_encoder,
    feature_columns: list,
) -> np.ndarray:
    """
    Applique le pipeline complet à des données nouvelles, dans l'ordre exact
    utilisé à l'entraînement, et renvoie un array prêt à être passé au modèle.

    Étapes : drop_leakage -> handle_missing -> engineer_features ->
             encode_categoricals(fit=False) -> alignement des colonnes ->
             cap_outliers -> scaler.transform
    """
    df = data.copy()

    df = drop_leakage_columns(df)
    df = handle_missing(df)
    df = engineer_features(df)
    df, _ = encode_categoricals(df, country_encoder=country_encoder, fit=False, feature_columns=feature_columns)

    # Alignement strict sur la liste de colonnes vue à l'entraînement.
    # Les colonnes one-hot manquantes (valeur catégorielle absente du nouveau set)
    # sont ajoutées à 0. Les colonnes en trop (valeur catégorielle inconnue
    # à l'entraînement) sont ignorées.
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    df = cap_outliers_iqr(df, columns=OUTLIER_COLS)

    return scaler.transform(df.values)
