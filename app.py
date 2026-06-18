"""
Hotel Booking Cancellation Prediction — Application Streamlit
Point d'entree. Charge le CSS, configure le menu, route vers les pages.

Navigation :
- Non connecte  : Accueil  |  Dataset  |  Authentification
- Connecte      : + Dashboard  |  Prediction  |  Prediction CSV  |  Performance
"""
import os
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title='Hotel Booking — Prediction ANN',
    page_icon=None,
    layout='wide',
    initial_sidebar_state='expanded',
)


def load_css():
    css_path = os.path.join('css', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()


# Session state
defaults = {
    'logged_in':    False,
    'username':     None,
    'active_page':  'Accueil',
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Menus
PUBLIC_PAGES    = ['Accueil', 'Dataset', 'Authentification']
PROTECTED_PAGES = ['Dashboard', 'Prediction', 'Prediction CSV', 'Performance']

SIDEBAR_STYLES = {
    'container':         {'background-color': '#1e3a5f', 'padding': '0'},
    'nav-link':          {
        'color': '#cbd5e1', 'font-size': '14px',
        'margin': '2px 0', 'padding': '8px 12px',
        'border-radius': '6px',
    },
    'nav-link-selected': {
        'background-color': '#2563eb',
        'color': 'white',
        'font-weight': '600',
    },
}


with st.sidebar:
    st.markdown('## Hotel Booking')
    st.markdown('**Prediction ANN**')
    st.divider()

    logged_in = st.session_state.get('logged_in', False)

    if not logged_in:
        selected = option_menu(
            menu_title=None,
            options=PUBLIC_PAGES,
            icons=None,
            default_index=PUBLIC_PAGES.index(
                st.session_state['active_page']
                if st.session_state['active_page'] in PUBLIC_PAGES
                else 'Accueil'
            ),
            styles=SIDEBAR_STYLES,
            key='menu_public',
        )
        st.divider()
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.8rem; padding:0 8px;">'
            'Connectez-vous pour acceder au Dashboard et aux outils de prediction.'
            '</p>',
            unsafe_allow_html=True,
        )

    else:
        # Menu unifié pour l'utilisateur connecté
        options = ['Accueil', 'Dataset', 'Dashboard', 'Prediction', 'Prediction CSV', 'Performance']
        
        # Redirection par défaut après la connexion
        if st.session_state['active_page'] not in options:
            st.session_state['active_page'] = 'Dashboard'

        selected = option_menu(
            menu_title=None,
            options=options,
            icons=None,
            default_index=options.index(st.session_state['active_page']),
            styles=SIDEBAR_STYLES,
            key='menu_logged_in',
        )

        st.divider()
        st.markdown(
            f'<p style="color:#e2e8f0; font-size:0.85rem; padding:0 8px;">'
            f'Connecté : <strong>{st.session_state["username"]}</strong></p>',
            unsafe_allow_html=True,
        )
        if st.button('Se déconnecter', use_container_width=True):
            st.session_state['logged_in']   = False
            st.session_state['username']    = None
            st.session_state['active_page'] = 'Accueil'
            st.rerun()


st.session_state['active_page'] = selected


from pages import (
    home, about_dataset, auth, dashboard,
    prediction, bulk_prediction, model_performance,
)

page_map = {
    'Accueil':          home.show,
    'Dataset':          about_dataset.show,
    'Authentification': auth.show,
    'Dashboard':        dashboard.show,
    'Prediction':       prediction.show,
    'Prediction CSV':   bulk_prediction.show,
    'Performance':      model_performance.show,
}

page_map[selected]()
