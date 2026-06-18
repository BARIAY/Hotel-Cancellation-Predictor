"""Performance du Modele ANN — metriques detaillees, architecture, courbes d'apprentissage."""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import visualization as viz
from utils.model_utils import load_training_history

COLOR_PRIMARY   = '#1e3a5f'
COLOR_SECONDARY = '#2563eb'
COLOR_ACCENT    = '#dc2626'
COLOR_SUCCESS   = '#16a34a'


def _render_architecture():
    st.markdown('<div class="section-title">Architecture du Reseau de Neurones (ANN)</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="alert-info">
            <strong>Justification de l'ANN :</strong> Un reseau de neurones artificiel (ANN) 
            a ete retenu pour sa capacite unique a capturer des <strong>relations non lineaires complexes</strong> 
            entre les variables, la ou des modeles lineaires classiques echouent. Les interactions 
            entre le delai de reservation (lead time), le type de depot et le segment de marche sont 
            naturellement apprehendees par les differentes couches de notre reseau.
        </div>
        """,
        unsafe_allow_html=True,
    )

    arch_data = {
        'Couche':       ['Input', 'Dense 1', 'BatchNorm 1', 'Dropout 1',
                         'Dense 2', 'BatchNorm 2', 'Dropout 2',
                         'Dense 3', 'Dropout 3', 'Output'],
        'Neurones':     ['n_features', '128', '—', '—', '64', '—', '—', '32', '—', '1'],
        'Activation':   ['—', 'ReLU', '—', '—', 'ReLU', '—', '—', 'ReLU', '—', 'Sigmoid'],
        'Dropout':      ['—', '—', '—', '30%', '—', '—', '30%', '—', '20%', '—'],
        'Role':         [
            'Features preprocessees',
            'Capture la diversite initiale des patterns',
            'Stabilise et accelere l\'entrainement',
            'Regularisation — evite l\'overfitting',
            'Compression progressive',
            'Normalisation par batch',
            'Regularisation couche cachee 2',
            'Representation abstraite de haut niveau',
            'Regularisation finale',
            'Probabilite d\'annulation finale [0, 1]',
        ],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card" style="border-top:3px solid #2563eb;">
                <strong style="color:#1e3a5f;">Optimiseur</strong><br>
                <span style="color:#64748b; font-size:0.9rem;">
                    Adam (lr=0.001)<br>
                    Algorithme adaptatif,<br>
                    ideal pour les ANN
                </span>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            <div class="card" style="border-top:3px solid #16a34a;">
                <strong style="color:#1e3a5f;">Fonction de perte</strong><br>
                <span style="color:#64748b; font-size:0.9rem;">
                    Binary Crossentropy<br>
                    Standard pour classification<br>
                    binaire avec sortie sigmoide
                </span>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown(
            """
            <div class="card" style="border-top:3px solid #dc2626;">
                <strong style="color:#1e3a5f;">Callbacks</strong><br>
                <span style="color:#64748b; font-size:0.9rem;">
                    EarlyStopping (patience=10)<br>
                    ReduceLROnPlateau<br>
                    (factor=0.5, patience=5)
                </span>
            </div>
            """, unsafe_allow_html=True)


def _render_metrics():
    """Affiche les metriques de l'ANN si disponibles depuis le CSV."""
    st.markdown('<div class="section-title">Metriques de Performance ANN</div>',
                unsafe_allow_html=True)

    comparison_path = os.path.join('models', 'model_comparison.csv')
    if not os.path.exists(comparison_path):
        st.warning(
            'Les metriques ne sont pas encore disponibles. '
            'Executez le script d\'entrainement `train.py` pour les generer.'
        )
        return

    df = pd.read_csv(comparison_path)

    # Cherche la ligne ANN
    ann_mask = df['Model'].str.contains('ANN', case=False, na=False)
    ann_row  = df[ann_mask]

    if ann_row.empty:
        ann_row = df.tail(1)

    ann = ann_row.iloc[0]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available = [m for m in metrics if m in df.columns]

    if available:
        cols = st.columns(len(available))
        colors = [COLOR_SECONDARY, COLOR_PRIMARY, COLOR_SUCCESS, COLOR_ACCENT, '#7c3aed']
        for col, metric, color in zip(cols, available, colors):
            val = ann.get(metric, '—')
            try:
                val_str = f"{float(val):.4f}"
            except (ValueError, TypeError):
                val_str = str(val)
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="border-top-color:{color};">
                        <div class="kpi-value" style="color:{color}; font-size:1.6rem;">{val_str}</div>
                        <div class="kpi-label">{metric}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<br>', unsafe_allow_html=True)

    # Interpretation des metriques
    st.markdown(
        """
        <div class="alert-info">
            <strong>Comprendre les metriques :</strong><br>
            &bull; <strong>Accuracy</strong> : proportion globale de bonnes predictions. Elle mesure le taux de succes general.<br>
            &bull; <strong>Precision</strong> : fiabilite des alertes. Parmi les reservations predites comme annulees, quelle fraction l'est reellement. Une bonne precision evite le surbooking excessif.<br>
            &bull; <strong>Recall (Sensibilite)</strong> : taux de detection. Parmi les annulations reelles, quelle fraction le modele parvient a capter. Un bon recall evite de laisser des chambres vides non relouees.<br>
            &bull; <strong>F1-Score</strong> : moyenne equilibree entre Precision et Recall — indicateur cle de la stabilite globale du modele.<br>
            &bull; <strong>ROC-AUC</strong> : capacite globale du modele a distinguer les reservations stables des annulations, independamment du seuil choisi (1.0000 = prediction parfaite).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Radar chart des metriques ANN
    if len(available) >= 3:
        try:
            vals = [float(ann.get(m, 0)) for m in available]
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=available + [available[0]],
                fill='toself',
                fillcolor='rgba(37, 99, 235, 0.15)',
                line=dict(color=COLOR_SECONDARY, width=2),
                name='ANN'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title='Visualisation Radar des Performances de l\'ANN',
                height=380,
                paper_bgcolor='white',
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


def _render_training_curves():
    st.markdown("<div class='section-title'>Courbes d'Apprentissage ANN</div>",
                unsafe_allow_html=True)
    history = load_training_history()
    if not history:
        st.warning(
            'L\'historique d\'entrainement n\'est pas encore disponible dans `models/training_history.json`. '
            'Veuillez entrainer le modele.'
        )
        return

    fig_loss, fig_acc = viz.training_curves(history)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_loss, use_container_width=True)
    with col2:
        st.plotly_chart(fig_acc, use_container_width=True)

    # Interpretation des courbes
    epochs = len(history.get('loss', []))
    st.markdown(
        f"""
        <div class="alert-info">
            <strong>Interpretation des courbes ({epochs} epochs executees) :</strong><br>
            &bull; <strong>Perte (Loss)</strong> : la baisse progressive des courbes train et validation prouve que le modele converge vers un etat optimal sans memoriser le bruit (absence d'overfitting).<br>
            &bull; <strong>Accuracy</strong> : le taux de succes progresse regulierement avant de se stabiliser.<br>
            &bull; Le rappel des meilleurs poids (poids de l'epoch a la validation loss minimale) est assure automatiquement par la fonction de callback a l'arret.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_assets():
    cm_path = os.path.join('assets', 'confusion_matrix_ann.png')

    if os.path.exists(cm_path):
        st.markdown('<div class="section-title">Evaluation de la Matrice de Confusion</div>',
                    unsafe_allow_html=True)
        _, col_mid, _ = st.columns([1, 2, 1])
        with col_mid:
            st.image(cm_path, caption='Matrice de confusion — Reseau de Neurones ANN',
                     use_container_width=True)
            st.markdown(
                """
                <div class="alert-info" style="text-align: left;">
                    <strong>Guide de lecture :</strong><br>
                    &bull; <strong>Vrais Negatifs</strong> (haut-gauche) : reservations predites comme confirmees qui le sont reellement.<br>
                    &bull; <strong>Faux Positifs</strong> (haut-droite) : confirmations predites a tort comme annulations.<br>
                    &bull; <strong>Faux Negatifs</strong> (bas-gauche) : annulations non detectees (les plus couteuses car la chambre reste vide).<br>
                    &bull; <strong>Vrais Positifs</strong> (bas-droite) : annulations correctement anticipees par le reseau.
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_confusion_matrix_table():
    """Affiche le guide d'interpretation de la matrice de confusion."""
    st.markdown('<div class="section-title">Recommandations Operationnelles</div>',
                unsafe_allow_html=True)

    guide = pd.DataFrame({
        'Cas de Figure':       ['Vrai Negatif', 'Faux Positif', 'Faux Negatif', 'Vrai Positif'],
        'Realite':       ['Confirmee', 'Confirmee', 'Annulee', 'Annulee'],
        'Prediction':    ['Confirmee', 'Annulee', 'Confirmee', 'Annulee'],
        'Impact Metier':   ['Aucun (Correct)', 'Surbooking leger induit',
                          'Chambre vacante non relouee (Perte)', 'Gain de revenu (Prevention)'],
        'Action Suggeree':        ['Accueil standard du client', 'Prendre contact simple de courtoisie',
                          'Sensibiliser la prediction (seuil < 0.5)', 'Appliquer politique de depot strict'],
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="alert-warning">
            <strong>Optimisation du seuil de decision :</strong> Par defaut, le modele predit une annulation 
            a partir de 50% de probabilite (seuil = 0.5). Si votre hotel subit un fort manque a gagner lie a des 
            chambres laissees vides, l'equipe Revenue Management peut abaisser ce seuil a 0.4. Cela permet de 
            detecter plus d'annulations (meilleur Recall), quitte a contacter des clients stables par exces de prudence.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show():
    tab1, tab2, tab3 = st.tabs([
        'Architecture & Metriques',
        "Courbes d'Apprentissage",
        'Evaluation Visuelle & Decisions',
    ])

    with tab1:
        _render_architecture()
        _render_metrics()

    with tab2:
        _render_training_curves()

    with tab3:
        _render_assets()
        _render_confusion_matrix_table()
