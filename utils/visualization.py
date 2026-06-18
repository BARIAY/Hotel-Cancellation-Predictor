"""
Fonctions Plotly réutilisées dans plusieurs pages.
Palette cohérente avec la charte du projet et le CSS.
"""
import plotly.express as px
import plotly.graph_objects as go


COLOR_PRIMARY    = '#1e3a5f'
COLOR_SECONDARY  = '#2563eb'
COLOR_ACCENT     = '#dc2626'
COLOR_SUCCESS    = '#16a34a'
COLOR_NEUTRAL    = '#64748b'
COLOR_BG         = '#f1f5f9'

STATUS_PALETTE = {0: COLOR_SUCCESS, 1: COLOR_ACCENT}
STATUS_LABEL   = {0: 'Confirmée', 1: 'Annulée'}


def _apply_layout(fig, title=None, height=None):
    fig.update_layout(
        title=title,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Inter, system-ui, sans-serif', color='#1e293b'),
        margin=dict(l=40, r=20, t=60, b=40),
        height=height,
    )
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    return fig


# -----------------------------------------------------------------------------
# Graphiques
# -----------------------------------------------------------------------------

def bar_cancel_by_category(df, category_col, title=None):
    """Barplot groupé Confirmé / Annulé par catégorie."""
    grouped = df.groupby([category_col, 'is_canceled']).size().reset_index(name='count')
    grouped['statut'] = grouped['is_canceled'].map(STATUS_LABEL)
    fig = px.bar(
        grouped, x=category_col, y='count', color='statut',
        barmode='group',
        color_discrete_map={'Confirmée': COLOR_SUCCESS, 'Annulée': COLOR_ACCENT},
        labels={'count': 'Nombre', category_col: ''},
    )
    return _apply_layout(fig, title=title)


def cancellation_rate_bar(df, category_col, title=None, horizontal=False):
    """Barplot du taux d'annulation par catégorie."""
    rates = df.groupby(category_col)['is_canceled'].mean().mul(100).round(2).reset_index()
    rates = rates.sort_values('is_canceled', ascending=horizontal)
    if horizontal:
        fig = px.bar(rates, x='is_canceled', y=category_col, orientation='h',
                     color_discrete_sequence=[COLOR_SECONDARY])
        fig.update_layout(xaxis_title="Taux d'annulation (%)", yaxis_title='')
    else:
        fig = px.bar(rates, x=category_col, y='is_canceled',
                     color_discrete_sequence=[COLOR_SECONDARY])
        fig.update_layout(yaxis_title="Taux d'annulation (%)", xaxis_title='')
    return _apply_layout(fig, title=title)


def monthly_trend(df, title=None):
    """Volume mensuel + taux d'annulation en double axe."""
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly = df.groupby('arrival_date_month').agg(
        total=('is_canceled', 'count'),
        rate=('is_canceled', 'mean')
    ).reindex(month_order)
    monthly['rate'] = monthly['rate'] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly['total'],
        name='Volume', marker_color=COLOR_BG,
        marker_line_color=COLOR_NEUTRAL, marker_line_width=1,
        yaxis='y',
    ))
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly['rate'],
        name="Taux d'annulation (%)",
        line=dict(color=COLOR_ACCENT, width=3),
        marker=dict(size=8),
        yaxis='y2',
    ))
    fig.update_layout(
        yaxis=dict(title='Volume de réservations', side='left'),
        yaxis2=dict(title="Taux d'annulation (%)", side='right', overlaying='y', showgrid=False),
        xaxis=dict(title=''),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return _apply_layout(fig, title=title)


def lead_time_box(df, title=None):
    """Boxplot du lead_time par statut."""
    df_copy = df.copy()
    df_copy['statut'] = df_copy['is_canceled'].map(STATUS_LABEL)
    fig = px.box(df_copy, x='statut', y='lead_time', color='statut',
                 color_discrete_map={'Confirmée': COLOR_SUCCESS, 'Annulée': COLOR_ACCENT})
    fig.update_layout(xaxis_title='', yaxis_title='Lead Time (jours)', showlegend=False)
    return _apply_layout(fig, title=title)


def correlation_heatmap(df, title=None):
    """Heatmap de corrélation des variables numériques."""
    corr = df.select_dtypes(include='number').corr().round(2)
    fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    aspect='auto', text_auto=True)
    return _apply_layout(fig, title=title, height=700)


def risk_gauge(probability_pct: float, danger: bool):
    """Jauge Plotly pour la page de prédiction."""
    color = COLOR_ACCENT if danger else COLOR_SUCCESS
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=probability_pct,
        number={'suffix': '%', 'font': {'size': 36}},
        title={'text': "Risque d'annulation", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': 'white',
            'steps': [
                {'range': [0, 40],  'color': '#dcfce7'},
                {'range': [40, 70], 'color': '#fef9c3'},
                {'range': [70, 100], 'color': '#fee2e2'},
            ],
            'threshold': {
                'line': {'color': COLOR_PRIMARY, 'width': 3},
                'thickness': 0.85,
                'value': 50,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20),
                      paper_bgcolor='white', font=dict(family='Inter, sans-serif'))
    return fig


def metrics_grouped_bar(results_df, title=None):
    """Barplot groupé Accuracy / F1 / ROC-AUC pour chaque modèle."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics = [m for m in metrics if m in results_df.columns]
    melted = results_df.melt(id_vars='Model', value_vars=metrics,
                              var_name='Métrique', value_name='Score')
    fig = px.bar(melted, x='Model', y='Score', color='Métrique', barmode='group',
                 color_discrete_sequence=[
                     COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS,
                     COLOR_ACCENT, COLOR_NEUTRAL
                 ])
    fig.update_layout(xaxis_tickangle=-30, yaxis_range=[0, 1.05])
    return _apply_layout(fig, title=title)


def training_curves(history: dict):
    """Courbes Loss + Accuracy à partir de l'history d'entraînement."""
    epochs = list(range(1, len(history.get('loss', [])) + 1))

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=history.get('loss', []),
                                   name='Train', line=dict(color=COLOR_SECONDARY, width=2.5)))
    fig_loss.add_trace(go.Scatter(x=epochs, y=history.get('val_loss', []),
                                   name='Validation',
                                   line=dict(color=COLOR_ACCENT, width=2.5, dash='dash')))
    fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Binary Crossentropy')
    fig_loss = _apply_layout(fig_loss, title='Évolution de la Loss')

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=history.get('accuracy', []),
                                  name='Train', line=dict(color=COLOR_SECONDARY, width=2.5)))
    fig_acc.add_trace(go.Scatter(x=epochs, y=history.get('val_accuracy', []),
                                  name='Validation',
                                  line=dict(color=COLOR_ACCENT, width=2.5, dash='dash')))
    fig_acc.update_layout(xaxis_title='Epoch', yaxis_title='Accuracy')
    fig_acc = _apply_layout(fig_acc, title="Évolution de l'Accuracy")

    return fig_loss, fig_acc


def proba_distribution(df, title=None):
    """Histogramme des probabilités prédites colorées par statut."""
    fig = px.histogram(df, x='probability', color='statut', nbins=30,
                       color_discrete_map={'Confirmée': COLOR_SUCCESS, 'Annulée': COLOR_ACCENT})
    fig.update_layout(xaxis_title='Probabilité prédite', yaxis_title='Fréquence',
                      bargap=0.05)
    return _apply_layout(fig, title=title)
