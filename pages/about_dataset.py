"""Page de présentation du dataset Hotel Booking Demand."""
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import visualization as viz

DATA_PATH = 'assets/hotel_bookings.csv'

FEATURES_DESC = {
    'hotel': "Type d'hôtel (Resort Hotel ou City Hotel)",
    'is_canceled': 'Variable cible : 1 si la réservation a été annulée, 0 sinon',
    'lead_time': "Nombre de jours entre la création de la réservation et la date d'arrivée",
    'arrival_date_year': "Année d'arrivée",
    'arrival_date_month': "Mois d'arrivée",
    'arrival_date_week_number': "Numéro de semaine de l'arrivée",
    'arrival_date_day_of_month': "Jour du mois d'arrivée",
    'stays_in_weekend_nights': 'Nombre de nuits de week-end réservées',
    'stays_in_week_nights': 'Nombre de nuits de semaine réservées',
    'adults': "Nombre d'adultes",
    'children': "Nombre d'enfants",
    'babies': 'Nombre de bébés',
    'meal': 'Type de repas réservé (BB, HB, FB, SC/Undefined)',
    'country': "Pays d'origine du client",
    'market_segment': 'Segment de marché (Online TA, Offline TA, Direct, Corporate, etc.)',
    'distribution_channel': 'Canal de distribution (TA/TO, Direct, Corporate, GDS)',
    'is_repeated_guest': '1 si le client a déjà séjourné dans cet hôtel',
    'previous_cancellations': "Nombre d'annulations antérieures du client",
    'previous_bookings_not_canceled': "Nombre de réservations antérieures non annulées",
    'reserved_room_type': 'Type de chambre réservée',
    'assigned_room_type': 'Type de chambre effectivement attribuée',
    'booking_changes': 'Nombre de modifications apportées à la réservation',
    'deposit_type': 'Type de dépôt (No Deposit, Non Refund, Refundable)',
    'agent': "ID de l'agence de voyages ayant fait la réservation",
    'company': 'ID de la société ayant fait la réservation',
    'days_in_waiting_list': "Nombre de jours passés sur liste d'attente",
    'customer_type': 'Type de client (Transient, Contract, Group, Transient-Party)',
    'adr': 'Average Daily Rate (tarif journalier moyen en euros)',
    'required_car_parking_spaces': 'Nombre de places de parking demandées',
    'total_of_special_requests': 'Nombre total de demandes spéciales du client',
    'reservation_status': 'Statut final (ATTENTION : variable de fuite, à supprimer)',
    'reservation_status_date': 'Date du dernier changement de statut',
}


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    return df


def show():
    st.markdown('<div class="section-title">À propos du Dataset</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <p><strong>Source :</strong> Hotel Booking Demand — Antonio, Almeida &amp; Nunes (2019),
            Data in Brief, Elsevier.</p>
            <p><strong>Origine :</strong> Systèmes de gestion (PMS) de deux hôtels portugais —
            un Resort Hotel en Algarve et un City Hotel à Lisbonne.</p>
            <p><strong>Période :</strong> juillet 2015 à août 2017.</p>
            <p><strong>Volumétrie :</strong> 119 390 réservations &middot; 32 variables.</p>
            <p><strong>Point de mesure :</strong> jour précédant l'arrivée, pour éviter toute
            fuite d'information future (data leakage).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    if df is None:
        st.warning(
            f"Le fichier `{DATA_PATH}` n'a pas été trouvé. "
            "Téléchargez le dataset depuis Kaggle et placez-le dans le dossier `assets/`."
        )
        return

    # Aperçu
    st.markdown('<div class="section-title">Aperçu des Données</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    # Tableau des features
    st.markdown('<div class="section-title">Description des Variables</div>',
                unsafe_allow_html=True)
    feat_df = pd.DataFrame(
        [(k, v) for k, v in FEATURES_DESC.items()],
        columns=['Variable', 'Description']
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=400)

    # Visualisations
    st.markdown('<div class="section-title">Distributions Clés</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        target_counts = df['is_canceled'].value_counts().sort_index()
        fig = px.pie(
            values=target_counts.values,
            names=['Confirmée', 'Annulée'],
            color_discrete_sequence=[viz.COLOR_SUCCESS, viz.COLOR_ACCENT],
            title='Distribution de la cible',
            hole=0.4,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="alert-info">Déséquilibre modéré (~63%/37%). '
            "L'accuracy seule serait trompeuse — on privilégie F1-Score et ROC-AUC.</div>",
            unsafe_allow_html=True,
        )

    with col2:
        fig = viz.bar_cancel_by_category(df, 'hotel',
                                         title="Annulations par type d'hôtel")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="alert-info">Le City Hotel présente un taux d\'annulation '
            'plus élevé que le Resort Hotel, en cohérence avec une clientèle plus '
            'orientée affaires (plans flexibles).</div>',
            unsafe_allow_html=True,
        )

    # Lead time
    fig = px.histogram(df, x='lead_time', color=df['is_canceled'].map({0: 'Confirmée', 1: 'Annulée'}),
                       nbins=50,
                       color_discrete_map={'Confirmée': viz.COLOR_SUCCESS, 'Annulée': viz.COLOR_ACCENT},
                       title='Distribution du lead time par statut',
                       labels={'color': 'Statut'})
    fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="alert-info">Les réservations annulées ont un lead time '
        'médian plus élevé. Un long délai entre réservation et arrivée augmente '
        'le risque d\'annulation.</div>',
        unsafe_allow_html=True,
    )

    # Top pays
    top_countries = df['country'].value_counts().head(15).reset_index()
    top_countries.columns = ['country', 'count']
    fig = px.bar(top_countries, y='country', x='count', orientation='h',
                 color_discrete_sequence=[viz.COLOR_SECONDARY],
                 title="Top 15 pays d'origine")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
    st.plotly_chart(fig, use_container_width=True)

    # Valeurs manquantes
    st.markdown('<div class="section-title">Valeurs Manquantes</div>',
                unsafe_allow_html=True)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        st.success('Aucune valeur manquante après imputation de base.')
    else:
        missing_df = pd.DataFrame({
            'Colonne': missing.index,
            'Manquantes': missing.values,
            'Pourcentage': (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
