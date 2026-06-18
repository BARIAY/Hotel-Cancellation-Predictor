"""Prédiction individuelle — saisie d'une réservation et résultat."""
import streamlit as st
import pandas as pd

from utils import visualization as viz
from utils.preprocessing import preprocess_for_prediction
from utils.model_utils import load_artifacts, predict


def _require_login() -> bool:
    if not st.session_state.get('logged_in'):
        st.markdown('<div class="section-title">Prédiction Individuelle</div>',
                    unsafe_allow_html=True)
        st.error('Accès réservé. Veuillez vous authentifier via la page Authentification.')
        return False
    return True


def _build_input_row(values: dict) -> pd.DataFrame:
    """
    Construit une DataFrame d'une ligne avec les colonnes BRUTES attendues par
    le pipeline de preprocessing (mêmes noms que le CSV original).
    """
    row = {
        'hotel':                          values['hotel'],
        'lead_time':                      values['lead_time'],
        'arrival_date_year':              values['arrival_date_year'],
        'arrival_date_month':             values['arrival_date_month'],
        'arrival_date_week_number':       values['arrival_date_week_number'],
        'arrival_date_day_of_month':      values['arrival_date_day_of_month'],
        'stays_in_weekend_nights':        values['stays_weekend'],
        'stays_in_week_nights':           values['stays_week'],
        'adults':                         values['adults'],
        'children':                       values['children'],
        'babies':                         values['babies'],
        'meal':                           values['meal'],
        'country':                        values['country'],
        'market_segment':                 values['market_segment'],
        'distribution_channel':           values['distribution_channel'],
        'is_repeated_guest':              1 if values['is_repeated_guest'] == 'Oui' else 0,
        'previous_cancellations':         values['previous_cancellations'],
        'previous_bookings_not_canceled': values['previous_bookings_not_canceled'],
        'reserved_room_type':             values['reserved_room_type'],
        'assigned_room_type':             values['assigned_room_type'],
        'booking_changes':                values['booking_changes'],
        'deposit_type':                   values['deposit_type'],
        'agent':                          values['agent'],
        'days_in_waiting_list':           values['days_in_waiting_list'],
        'customer_type':                  values['customer_type'],
        'adr':                            values['adr'],
        'required_car_parking_spaces':    values['parking'],
        'total_of_special_requests':      values['special_requests'],
    }
    return pd.DataFrame([row])


def _render_form():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Réservation**')
        hotel = st.selectbox('Type d\'hôtel', ['City Hotel', 'Resort Hotel'])
        lead_time = st.slider('Lead Time (jours)', 0, 500, 30)
        arrival_date_year = st.selectbox("Année d'arrivée", [2015, 2016, 2017], index=1)
        arrival_date_month = st.selectbox(
            "Mois d'arrivée",
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December']
        )
        arrival_date_week_number = st.slider("Semaine de l'année", 1, 53, 25)
        arrival_date_day_of_month = st.slider('Jour du mois', 1, 31, 15)

    with col2:
        st.markdown('**Séjour & Clients**')
        stays_week = st.number_input('Nuits de semaine', 0, 30, 2)
        stays_weekend = st.number_input('Nuits de week-end', 0, 10, 1)
        adults = st.number_input("Nombre d'adultes", 1, 10, 2)
        children = st.number_input("Nombre d'enfants", 0, 5, 0)
        babies = st.number_input('Nombre de bébés', 0, 3, 0)
        meal = st.selectbox('Type de repas', ['BB', 'HB', 'FB', 'SC', 'Undefined'])
        country = st.text_input("Pays d'origine (code ISO 3, ex. PRT)", value='PRT')

    with col3:
        st.markdown('**Commercial & Tarification**')
        market_segment = st.selectbox(
            'Segment de marché',
            ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate',
             'Groups', 'Complementary', 'Aviation']
        )
        distribution_channel = st.selectbox(
            'Canal de distribution', ['TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined']
        )
        deposit_type = st.selectbox('Type de dépôt',
                                     ['No Deposit', 'Non Refund', 'Refundable'])
        customer_type = st.selectbox('Type de client',
                                      ['Transient', 'Contract', 'Group', 'Transient-Party'])
        adr = st.number_input('ADR (tarif/nuit €)', 0.0, 2000.0, 100.0, step=5.0)
        special_requests = st.slider('Demandes spéciales', 0, 5, 0)
        is_repeated_guest = st.radio('Client fidèle', ['Non', 'Oui'], horizontal=True)

    with st.expander('Champs avancés'):
        col4, col5, col6 = st.columns(3)
        with col4:
            previous_cancellations = st.number_input('Annulations précédentes', 0, 20, 0)
            previous_bookings_not_canceled = st.number_input('Réservations précédentes non annulées',
                                                              0, 50, 0)
        with col5:
            reserved_room_type = st.selectbox('Type de chambre réservée',
                                               ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'])
            assigned_room_type = st.selectbox('Type de chambre attribuée',
                                               ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P'])
            booking_changes = st.number_input('Modifications réservation', 0, 20, 0)
        with col6:
            agent = st.number_input('ID agence (0 si direct)', 0, 600, 0)
            days_in_waiting_list = st.number_input("Jours sur liste d'attente", 0, 400, 0)
            parking = st.number_input('Places de parking', 0, 5, 0)

    return {
        'hotel': hotel, 'lead_time': lead_time,
        'arrival_date_year': arrival_date_year,
        'arrival_date_month': arrival_date_month,
        'arrival_date_week_number': arrival_date_week_number,
        'arrival_date_day_of_month': arrival_date_day_of_month,
        'stays_week': stays_week, 'stays_weekend': stays_weekend,
        'adults': adults, 'children': children, 'babies': babies,
        'meal': meal, 'country': country,
        'market_segment': market_segment, 'distribution_channel': distribution_channel,
        'is_repeated_guest': is_repeated_guest,
        'previous_cancellations': previous_cancellations,
        'previous_bookings_not_canceled': previous_bookings_not_canceled,
        'reserved_room_type': reserved_room_type,
        'assigned_room_type': assigned_room_type,
        'booking_changes': booking_changes,
        'deposit_type': deposit_type,
        'agent': agent,
        'days_in_waiting_list': days_in_waiting_list,
        'customer_type': customer_type,
        'adr': adr,
        'parking': parking,
        'special_requests': special_requests,
    }


def _render_result(values, proba, pred):
    if pred == 1:
        st.markdown(
            f"""
            <div class="result-cancel">
                <h2 style="color:#dc2626; margin:0;">Annulation Probable</h2>
                <p style="font-size:1.25rem; color:#7f1d1d; margin:0.75rem 0 0 0;">
                    Probabilité d'annulation : <strong>{proba * 100:.1f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-confirm">
                <h2 style="color:#16a34a; margin:0;">Réservation à Faible Risque</h2>
                <p style="font-size:1.25rem; color:#14532d; margin:0.75rem 0 0 0;">
                    Probabilité de confirmation : <strong>{(1 - proba) * 100:.1f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([2, 3])
    with col1:
        st.plotly_chart(viz.risk_gauge(proba * 100, pred == 1),
                        use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Recommandations</div>',
                    unsafe_allow_html=True)
        recs = []
        if pred == 1:
            if values['lead_time'] > 100:
                recs.append('Lead time élevé : envoyer une confirmation automatique '
                            "30 jours avant l'arrivée pour réengager le client.")
            if values['deposit_type'] == 'No Deposit':
                recs.append("Aucun dépôt : envisager une politique de dépôt non remboursable "
                            'pour ce segment ou exiger une caution.')
            if values['special_requests'] == 0:
                recs.append("Aucune demande spéciale : encourager le client à personnaliser "
                            "son séjour (upgrade, transfert aéroport, dîner) pour augmenter l'engagement.")
            if values['market_segment'] == 'Online TA':
                recs.append("Canal OTA : monitorer la réservation et préparer une "
                            "alternative locale si annulation tardive.")
            if not recs:
                recs.append('Surveillance recommandée. Contacter le client pour confirmer.')
        else:
            recs.append('Réservation à faible risque. Préparer un accueil standard.')
            if values['is_repeated_guest'] == 'Oui':
                recs.append('Client fidèle : accorder un statut prioritaire. '
                            'Ne jamais annuler en cas de surbooking.')

        for rec in recs:
            st.markdown(f'<div class="alert-info">{rec}</div>',
                        unsafe_allow_html=True)


def show():
    if not _require_login():
        return

    st.markdown('<div class="section-title">Prédiction Individuelle</div>',
                unsafe_allow_html=True)

    artifacts = load_artifacts()
    if artifacts is None:
        st.warning(
            "Les artefacts du modèle ne sont pas disponibles dans `models/`. "
            "Exécutez le script d'entraînement `train.py` ou le notebook pour les générer."
        )
        return

    values = _render_form()

    st.markdown('<br>', unsafe_allow_html=True)
    if st.button('Lancer la prédiction', type='primary', use_container_width=False):
        with st.spinner('Calcul en cours...'):
            input_df = _build_input_row(values)
            try:
                X_scaled = preprocess_for_prediction(
                    input_df,
                    scaler=artifacts['scaler'],
                    country_encoder=artifacts['country_encoder'],
                    feature_columns=artifacts['feature_columns'],
                )
                preds, probas = predict(artifacts['model'], X_scaled)
                proba = float(probas[0])
                pred = int(preds[0])
            except Exception as e:
                st.error(f'Erreur lors de la prédiction : {e}')
                return

        _render_result(values, proba, pred)
