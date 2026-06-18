"""Page d'accueil — presentation du projet ANN, description et fonctionnalites."""
import streamlit as st


def show():
    # Hero
    st.markdown(
        """
        <div class="hero">
            <h1>Hotel Booking Cancellation Prediction</h1>
            <p>
                Systeme de prediction par Reseau de Neurones Artificiel (ANN) permettant
                d'anticiper les annulations de reservations hotelieres a partir des
                caracteristiques connues au moment de la creation de la reservation.
            </p>
            <p style="font-size:0.95rem; opacity:0.85;">
                Dataset : 119 390 reservations hotelieres (Portugal, 2015-2017)
                &mdash; Antonio, Almeida &amp; Nunes (2019)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ('119 390', 'Reservations analysees'),
        ('37.04%',  "Taux d'annulation global"),
        ('~93%',    'Accuracy du modele ANN'),
        ('32',      'Variables dans le dataset'),
    ]
    for col, (value, label) in zip([col1, col2, col3, col4], kpis):
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

    st.markdown('<br>', unsafe_allow_html=True)

    # Description du projet
    st.markdown('<div class="section-title">Description du Projet</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="card">
                <h4 style="color:#1e3a5f; margin-bottom:0.75rem;">Problematique</h4>
                <p style="color:#475569; line-height:1.7; margin:0;">
                    Les annulations hotelieres representent une perturbation majeure pour la
                    gestion des revenus (Revenue Management). Un taux d'annulation eleve impacte
                    directement la planification des ressources, la strategie de prix et la
                    satisfaction client.<br><br>
                    Ce projet vise a <strong>predire si une reservation sera annulee</strong>
                    a partir de ses caracteristiques connues au moment de la reservation,
                    sans utiliser d'information future (data leakage).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h4 style="color:#1e3a5f; margin-bottom:0.75rem;">Approche ANN</h4>
                <p style="color:#475569; line-height:1.7; margin:0;">
                    Un <strong>Reseau de Neurones Artificiel (ANN)</strong> a ete choisi comme
                    modele principal pour sa capacite a capturer des relations non lineaires
                    complexes entre les variables, la ou la regression logistique ou les modeles
                    lineaires echouent.<br><br>
                    L'architecture retenue comprend 3 couches cachees (128 &rarr; 64 &rarr; 32
                    neurones), avec BatchNormalization et Dropout pour prevenir l'overfitting,
                    et une sortie sigmoide pour la classification binaire.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Architecture ANN
    st.markdown('<div class="section-title">Architecture du Reseau de Neurones</div>',
                unsafe_allow_html=True)

    arch_rows = [
        ('Entree',       'n_features', '—',       '—',        'Features preprocessees (OHE + scaling)'),
        ('Dense 1',      '128',        'ReLU',    '—',        'Capture la diversite initiale des patterns'),
        ('BatchNorm 1',  '128',        '—',       '—',        'Stabilise et accelere l\'entrainement'),
        ('Dropout 1',    '—',          '—',       '30%',      'Regularisation — evite l\'overfitting'),
        ('Dense 2',      '64',         'ReLU',    '—',        'Compression progressive'),
        ('BatchNorm 2',  '64',         '—',       '—',        'Normalisation par batch'),
        ('Dropout 2',    '—',          '—',       '30%',      'Regularisation supplementaire'),
        ('Dense 3',      '32',         'ReLU',    '—',        'Representation abstraite de haut niveau'),
        ('Dropout 3',    '—',          '—',       '20%',      'Regularisation finale'),
        ('Sortie',       '1',          'Sigmoid', '—',        'Probabilite d\'annulation [0, 1]'),
    ]

    rows_html = ''
    for r in arch_rows:
        rows_html += f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>'

    st.markdown(
        f"""
        <table class="arch-table">
            <tr>
                <th>Couche</th><th>Neurones</th><th>Activation</th>
                <th>Dropout</th><th>Role</th>
            </tr>
            {rows_html}
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="alert-info">Compilateur : Adam (lr=0.001) &mdash; '
        'Loss : Binary Crossentropy &mdash; '
        'Callbacks : EarlyStopping (patience=10) + ReduceLROnPlateau (patience=5)</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<br>', unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="section-title">Pipeline Machine Learning</div>',
                unsafe_allow_html=True)
    steps = [
        ('1', 'Donnees brutes', '119 390 reservations, 32 variables'),
        ('2', 'EDA', 'Exploration, correlations, saisonnalite'),
        ('3', 'Preprocessing', 'Nettoyage, encoding, SMOTE, scaling'),
        ('4', 'Entrainement ANN', '3 couches cachees, EarlyStopping'),
        ('5', 'Evaluation', 'Accuracy, F1, ROC-AUC, matrice de confusion'),
        ('6', 'Prediction', 'Application Streamlit en temps reel'),
    ]
    cols = st.columns(len(steps))
    for col, (num, title, sub) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="background:#eff6ff; border-radius:10px; padding:0.85rem 0.6rem;
                            text-align:center; border-top:3px solid #2563eb;">
                    <div style="font-size:1.4rem; font-weight:800; color:#2563eb;">{num}</div>
                    <div style="font-size:0.82rem; font-weight:700; color:#1e3a5f; margin:0.2rem 0;">{title}</div>
                    <div style="font-size:0.72rem; color:#64748b; line-height:1.3;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<br>', unsafe_allow_html=True)

    # Fonctionnalites
    st.markdown('<div class="section-title">Fonctionnalites de l\'Application</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    features = [
        ('Prediction Individuelle',
         "Saisissez les caracteristiques d'une reservation et obtenez une prediction "
         "instantanee avec probabilite d'annulation et recommandations metier."),
        ('Prediction par CSV',
         "Importez un fichier CSV contenant plusieurs reservations. "
         "L'application applique le pipeline complet et genere un fichier enrichi "
         "avec prediction et probabilite pour chaque ligne."),
        ('Tableau de Bord Analytique',
         "Explorez le dataset avec des visualisations interactives : distributions, "
         "saisonnalite, lead time, segments de marche, filtres dynamiques."),
    ]
    for col, (title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(
                f"""
                <div class="card" style="border-top:4px solid #2563eb; min-height:160px;">
                    <h4 style="color:#1e3a5f; margin-bottom:0.5rem;">{title}</h4>
                    <p style="color:#64748b; font-size:0.9rem; line-height:1.55; margin:0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Technologies
    st.markdown('<div class="section-title">Technologies Utilisees</div>',
                unsafe_allow_html=True)
    techs = ['Python 3.10', 'TensorFlow / Keras', 'Streamlit', 'Scikit-learn',
             'Plotly', 'Pandas', 'NumPy', 'imbalanced-learn (SMOTE)']
    badges = ' '.join(f'<span class="tech-badge">{t}</span>' for t in techs)
    st.markdown(f'<div style="margin-bottom:2rem;">{badges}</div>',
                unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div class="footer">
            Hotel Booking Demand &mdash; Projet de Classification par Reseau de Neurones (ANN)<br>
            Dataset : Antonio, Almeida &amp; Nunes (2019) &mdash; Data in Brief, Elsevier &mdash; Disponible sur Kaggle
        </div>
        """,
        unsafe_allow_html=True,
    )
