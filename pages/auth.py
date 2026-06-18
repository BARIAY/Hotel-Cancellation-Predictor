"""Page d'authentification — onglets Connexion / Inscription."""
import streamlit as st
from utils import auth_utils


def _render_login():
    email    = st.text_input('Email', key='login_email', placeholder='exemple@email.com')
    password = st.text_input('Mot de passe', type='password', key='login_password')

    if st.button('Se connecter', type='primary', use_container_width=True, key='login_btn'):
        if not email or not password:
            st.error('Veuillez remplir tous les champs.')
            return
        username = auth_utils.verify_user(email, password)
        if username:
            st.session_state['logged_in']   = True
            st.session_state['username']    = username
            st.session_state['user_email']  = email
            st.rerun()
        else:
            st.error('Email ou mot de passe incorrect.')


def _render_signup():
    username = st.text_input('Nom complet', key='signup_username')
    email    = st.text_input('Email', key='signup_email')
    password = st.text_input('Mot de passe (minimum 6 caracteres)',
                              type='password', key='signup_password')
    confirm  = st.text_input('Confirmer le mot de passe',
                              type='password', key='signup_confirm')

    if st.button("Creer mon compte", type='primary',
                 use_container_width=True, key='signup_btn'):
        if not all([username, email, password, confirm]):
            st.error('Veuillez remplir tous les champs.')
        elif not auth_utils.is_valid_email(email):
            st.error('Adresse email invalide.')
        elif not auth_utils.is_strong_enough(password):
            st.error('Le mot de passe doit contenir au moins 6 caracteres.')
        elif password != confirm:
            st.error('Les mots de passe ne correspondent pas.')
        else:
            ok, msg = auth_utils.save_user(username, email, password)
            if ok:
                st.success(msg + ' Vous pouvez maintenant vous connecter.')
            else:
                st.error(msg)


def _render_connected_panel():
    username = st.session_state.get('username', '')
    st.markdown(
        f"""
        <div class="result-confirm" style="text-align:left;">
            <h3 style="color:#16a34a; margin:0 0 0.5rem 0;">Bienvenue, {username}</h3>
            <p style="color:#166534; margin:0;">
                Vous etes connecte. Les pages protegees sont accessibles
                depuis le menu lateral dans la section <strong>Espace connecte</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    tiles = [
        ('Dashboard',       'Explorez les donnees et visualisations'),
        ('Prediction',      'Predire une reservation individuelle'),
        ('Prediction CSV',  'Analyser un fichier CSV en masse'),
    ]
    for col, (title, desc) in zip([col1, col2, col3], tiles):
        with col:
            st.markdown(
                f"""
                <div class="card" style="border-top:3px solid #2563eb; text-align:center;">
                    <div style="font-size:1.05rem; font-weight:700; color:#1e3a5f;">{title}</div>
                    <p style="color:#64748b; font-size:0.83rem; margin:0.3rem 0 0 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True
            )

    st.markdown('<br>', unsafe_allow_html=True)
    if st.button('Se deconnecter', key='auth_logout'):
        st.session_state['logged_in']   = False
        st.session_state['username']    = None
        st.session_state['active_page'] = 'Accueil'
        st.rerun()


def show():
    st.markdown('<div class="section-title">Authentification</div>',
                unsafe_allow_html=True)

    if st.session_state.get('logged_in'):
        _render_connected_panel()
        return

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(['Connexion', 'Inscription'])
        with tab1:
            _render_login()
        with tab2:
            _render_signup()
        st.markdown('</div>', unsafe_allow_html=True)
