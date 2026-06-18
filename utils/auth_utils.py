"""
Authentification basique pour l'application Streamlit.
Stocke les utilisateurs dans un fichier JSON local avec mots de passe hashés.
Pour un usage pédagogique uniquement — en production, utiliser une vraie base
de données et un système de hash plus robuste (bcrypt, argon2).
"""
import os
import json
import hashlib
import re


USERS_FILE = 'authentication/users.json'


def _ensure_dir():
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)


def hash_password(password: str) -> str:
    """Retourne le SHA-256 du mot de passe."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def load_users() -> dict:
    """Charge le dictionnaire des utilisateurs depuis le disque."""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Enregistre un nouvel utilisateur.
    Retourne (succès, message).
    """
    _ensure_dir()
    users = load_users()

    if email in users:
        return False, "Un compte existe déjà avec cet email."

    users[email] = {
        'username': username,
        'password_hash': hash_password(password),
    }
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    return True, "Compte créé avec succès."


def verify_user(email: str, password: str):
    """
    Vérifie les identifiants. Retourne le nom d'utilisateur si valide, None sinon.
    """
    users = load_users()
    record = users.get(email)
    if record is None:
        return None
    if record['password_hash'] == hash_password(password):
        return record['username']
    return None


def is_valid_email(email: str) -> bool:
    """Validation minimale d'un email."""
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email or ''))


def is_strong_enough(password: str) -> bool:
    """Critère minimal : au moins 6 caractères."""
    return isinstance(password, str) and len(password) >= 6
