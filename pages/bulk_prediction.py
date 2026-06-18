"""Prédiction par CSV — upload, traitement en masse, téléchargement des résultats."""
import streamlit as st
import pandas as pd

from utils import visualization as viz
from utils.preprocessing import preprocess_for_prediction
from utils.model_utils import load_artifacts, predict


# Colonnes brutes attendues dans le CSV uploadé (avant feature engineering interne)
REQUIRED_RAW_COLS = [
    'hotel', 'lead_time',
    'arrival_date_year', 'arrival_date_month',
    'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'meal', 'country',
    'market_segment', 'distribution_channel',
    'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled',
    'reserved_room_type', 'assigned_room_type', 'booking_changes',
    'deposit_type', 'agent', 'days_in_waiting_list', 'customer_type',
    'adr', 'required_car_parking_spaces', 'total_of_special_requests',
]


def _require_login() -> bool:
    if not st.session_state.get('logged_in'):
        st.markdown('<div class="section-title">Prédiction par CSV</div>',
                    unsafe_allow_html=True)
        st.error('Accès réservé. Veuillez vous authentifier.')
        return False
    return True


def show():
    if not _require_login():
        return

    st.markdown('<div class="section-title">Prédiction par CSV</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#64748b;">'
        'Importez un fichier CSV contenant plusieurs réservations. '
        "L'application appliquera le pipeline complet et produira un fichier "
        'enrichi avec la prédiction et la probabilité pour chaque ligne.'
        '</p>',
        unsafe_allow_html=True,
    )

    artifacts = load_artifacts()
    if artifacts is None:
        st.warning(
            "Les artefacts du modèle ne sont pas disponibles dans `models/`. "
            "Exécutez le script d'entraînement `train.py` ou le notebook pour les générer."
        )
        return

    # Aide colonnes attendues
    with st.expander('Colonnes attendues dans le CSV'):
        st.code(', '.join(REQUIRED_RAW_COLS), language='text')
        st.caption("Les colonnes `reservation_status`, `reservation_status_date` et `company` "
                   "sont automatiquement ignorées si présentes (data leakage / colonne inutilisable).")

    uploaded = st.file_uploader('Importer un fichier CSV', type=['csv'])
    if uploaded is None:
        return

    try:
        df_upload = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f'Erreur de lecture du CSV : {e}')
        return

    missing_cols = [c for c in REQUIRED_RAW_COLS if c not in df_upload.columns]
    if missing_cols:
        st.error(f'Colonnes manquantes : {", ".join(missing_cols)}')
        return

    st.success(f'Fichier valide : {len(df_upload):,} lignes, '
               f'{len(df_upload.columns)} colonnes détectées.')

    if not st.button('Lancer la prédiction en masse', type='primary'):
        return

    with st.spinner(f'Traitement de {len(df_upload):,} réservations...'):
        try:
            X_scaled = preprocess_for_prediction(
                df_upload,
                scaler=artifacts['scaler'],
                country_encoder=artifacts['country_encoder'],
                feature_columns=artifacts['feature_columns'],
            )
            preds, probas = predict(artifacts['model'], X_scaled)
        except Exception as e:
            st.error(f'Erreur lors de la prédiction : {e}')
            return

    df_out = df_upload.copy()
    df_out['prediction'] = preds
    df_out['probability'] = probas.round(4)
    df_out['statut'] = df_out['prediction'].map({0: 'Confirmée', 1: 'Annulée'})

    # KPI
    total = len(df_out)
    cancelled = int(preds.sum())
    confirmed = total - cancelled
    ratio = cancelled / total * 100 if total else 0

    cols = st.columns(4)
    items = [
        (f'{total:,}'.replace(',', ' '),     'Total',                       '#2563eb'),
        (f'{cancelled:,}'.replace(',', ' '), 'Annulations prédites',        '#dc2626'),
        (f'{confirmed:,}'.replace(',', ' '), 'Confirmées prédites',         '#16a34a'),
        (f'{ratio:.1f}%',                    "Taux d'annulation prédit",    '#f59e0b'),
    ]
    for col, (val, label, color) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-top-color:{color};">
                    <div class="kpi-value" style="color:{color};">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<br>', unsafe_allow_html=True)

    # Distribution des probabilités
    st.plotly_chart(
        viz.proba_distribution(df_out, title='Distribution des probabilités prédites'),
        use_container_width=True
    )

    # Aperçu du résultat — colonnes pertinentes en premier
    display_cols = ['statut', 'probability', 'hotel', 'lead_time',
                    'market_segment', 'deposit_type', 'adr', 'total_of_special_requests']
    display_cols = [c for c in display_cols if c in df_out.columns]
    st.markdown('<div class="section-title">Aperçu des résultats</div>',
                unsafe_allow_html=True)
    st.dataframe(df_out[display_cols].head(50), use_container_width=True)

    # Téléchargement
    csv_result = df_out.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Télécharger les résultats complets (CSV)',
        data=csv_result,
        file_name='predictions_hotel.csv',
        mime='text/csv',
        type='primary',
    )
