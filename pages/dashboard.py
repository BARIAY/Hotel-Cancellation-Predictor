"""Dashboard analytique — page protegee par authentification."""
import os
import streamlit as st
import pandas as pd

from utils import visualization as viz
from utils.model_utils import load_training_history

DATA_PATH = 'assets/hotel_bookings.csv'


def _require_login() -> bool:
    if not st.session_state.get('logged_in'):
        st.markdown('<div class="section-title">Dashboard Analytique</div>',
                    unsafe_allow_html=True)
        st.error('Acces reserve. Veuillez vous authentifier via la page Authentification.')
        return False
    return True


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    return df


def _render_filters(df):
    with st.sidebar:
        st.markdown('### Filtres Dashboard')
        hotel_filter = st.radio(
            "Type d'hotel",
            ['Tous'] + sorted(df['hotel'].unique().tolist()),
            key='filter_hotel'
        )
        months = sorted(df['arrival_date_month'].unique().tolist())
        month_filter = st.multiselect('Mois', months, key='filter_month')
        segments = sorted(df['market_segment'].unique().tolist())
        segment_filter = st.multiselect('Segment de marche', segments, key='filter_segment')

    df_filtered = df.copy()
    if hotel_filter != 'Tous':
        df_filtered = df_filtered[df_filtered['hotel'] == hotel_filter]
    if month_filter:
        df_filtered = df_filtered[df_filtered['arrival_date_month'].isin(month_filter)]
    if segment_filter:
        df_filtered = df_filtered[df_filtered['market_segment'].isin(segment_filter)]
    return df_filtered


def _render_kpis(df_filtered):
    total     = len(df_filtered)
    cancelled = int(df_filtered['is_canceled'].sum()) if total else 0
    confirmed = total - cancelled
    rate      = (cancelled / total * 100) if total else 0

    # Taux annulation Resort vs City
    city_rate = resort_rate = '—'
    if total:
        city_df   = df_filtered[df_filtered['hotel'] == 'City Hotel']
        resort_df = df_filtered[df_filtered['hotel'] == 'Resort Hotel']
        if len(city_df):
            city_rate   = f"{city_df['is_canceled'].mean()*100:.1f}%"
        if len(resort_df):
            resort_rate = f"{resort_df['is_canceled'].mean()*100:.1f}%"

    cols = st.columns(5)
    items = [
        (f'{total:,}'.replace(',', ' '),     'Reservations totales'),
        (f'{confirmed:,}'.replace(',', ' '), 'Confirmees'),
        (f'{cancelled:,}'.replace(',', ' '), 'Annulees'),
        (f'{rate:.1f}%',                     "Taux d'annulation"),
        (city_rate,                          'Taux City Hotel'),
    ]
    for col, (value, label) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _tab_general(df_filtered):
    st.plotly_chart(
        viz.bar_cancel_by_category(df_filtered, 'hotel',
                                   title="Annulations par type d'hotel"),
        use_container_width=True
    )
    st.markdown(
        '<div class="alert-info">Le City Hotel concentre generalement plus d\'annulations '
        'que le Resort Hotel. La clientele d\'affaires du City Hotel favorise les tarifs '
        'flexibles, augmentant le risque d\'annulation par rapport a la clientele '
        'vacanciere du Resort Hotel, plus engagee emotionnellement.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            viz.cancellation_rate_bar(df_filtered, 'market_segment',
                                      title='Taux par segment de marche',
                                      horizontal=True),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            viz.cancellation_rate_bar(df_filtered, 'distribution_channel',
                                      title='Taux par canal de distribution'),
            use_container_width=True
        )
    st.markdown(
        '<div class="alert-info">Les OTA (Online TA) et le canal TA/TO concentrent les '
        'taux d\'annulation les plus eleves. Les reservations directes sont les plus stables '
        'car le client a interagi directement avec l\'hotel, signe d\'un engagement plus fort.</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        viz.cancellation_rate_bar(df_filtered, 'deposit_type',
                                  title='Taux par type de depot'),
        use_container_width=True
    )
    st.markdown(
        '<div class="alert-warning">Paradoxe statistique : le type Non Refund presente le '
        'taux d\'annulation le plus eleve. Causalite inversee : les hotels appliquent cette '
        'politique aux reservations deja identifiees comme a risque (biais de selection). '
        'L\'ANN doit traiter cette feature avec prudence car elle peut introduire un '
        'signal trompeur dans les donnees d\'inference.</div>',
        unsafe_allow_html=True,
    )


def _tab_temporal(df_filtered):
    st.plotly_chart(
        viz.monthly_trend(df_filtered, title="Volume mensuel et taux d'annulation"),
        use_container_width=True
    )
    st.markdown(
        '<div class="alert-info">Pic d\'annulations en avril-juin, periode ou les '
        'reservations longues (lead time eleve) sont les plus frequentes. '
        'L\'ANN capte cette saisonnalite via les features d\'arrivee encodees '
        '(mois, semaine) et le lead_time, qui est la feature la plus predictive '
        'selon l\'importance des features (Random Forest).</div>',
        unsafe_allow_html=True,
    )

    import plotly.express as px
    yearly = df_filtered.groupby('arrival_date_year')['is_canceled'].agg(
        ['count', 'sum']).reset_index()
    yearly.columns = ['Annee', 'Total', 'Annulees']
    yearly['Confirmees'] = yearly['Total'] - yearly['Annulees']

    melted = yearly.melt(id_vars='Annee', value_vars=['Confirmees', 'Annulees'],
                          var_name='Statut', value_name='Nombre')
    fig = px.bar(melted, x='Annee', y='Nombre', color='Statut', barmode='group',
                 color_discrete_map={'Confirmees': viz.COLOR_SUCCESS,
                                     'Annulees': viz.COLOR_ACCENT},
                 title='Annulations par annee')
    fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="alert-info">La progression du volume entre 2015 et 2017 reflete '
        'la montee en puissance du tourisme au Portugal. L\'augmentation du taux '
        'd\'annulation suit l\'expansion des plateformes OTA sur la meme periode.</div>',
        unsafe_allow_html=True,
    )


def _tab_features(df_filtered):
    st.plotly_chart(
        viz.lead_time_box(df_filtered, title='Lead Time par statut'),
        use_container_width=True
    )
    st.markdown(
        '<div class="alert-info">Le lead time median des annulations est nettement plus '
        'eleve que celui des confirmations. C\'est la feature la plus predictive du dataset. '
        'Un lead time superieur a 100 jours multiplie le risque d\'annulation par plus de 2. '
        'L\'ANN exploite cette relation non lineaire entre lead_time et probabilite '
        'd\'annulation via ses couches de neurons profonds.</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        viz.correlation_heatmap(df_filtered, title='Matrice de correlation'),
        use_container_width=True
    )
    st.markdown(
        '<div class="alert-info">Les correlations directes avec is_canceled sont moderees '
        '(la plus forte ~0.29 pour lead_time), confirmant que les relations sont non '
        'lineaires — ce qui justifie l\'utilisation d\'un ANN capable de capturer ces '
        'interactions complexes. La multicolinearite entre certaines paires (adults/total_guests) '
        'est moins problematique pour les reseaux de neurones que pour la regression lineaire.</div>',
        unsafe_allow_html=True,
    )


def show():
    if not _require_login():
        return

    st.markdown('<div class="section-title">Dashboard Analytique</div>',
                unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.warning(
            f"Le fichier `{DATA_PATH}` n'a pas ete trouve. "
            "Telechargez le dataset depuis Kaggle et placez-le dans `assets/`."
        )
        return

    df_filtered = _render_filters(df)
    _render_kpis(df_filtered)

    st.markdown('<br>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(['Vue Generale', 'Analyse Temporelle', 'Analyse des Features'])
    with tab1:
        _tab_general(df_filtered)
    with tab2:
        _tab_temporal(df_filtered)
    with tab3:
        _tab_features(df_filtered)
