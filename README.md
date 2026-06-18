# 🏨 Hotel Booking Cancellation Prediction — ANN 🚀

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Projet de classification binaire par Réseau de Neurones Artificiel (ANN) pour prédire les annulations de réservations hôtelières.**
> 
> *Binary classification project using Artificial Neural Networks (ANN) to predict hotel booking cancellations.*

---

## 🌟 Features / Fonctionnalités

- **Interactive Dashboard**: Visualisations interactives et filtres dynamiques sur les données de réservation.
- **Real-time Prediction**: Prédiction individuelle des annulations en temps réel.
- **Bulk CSV Prediction**: Traitement en masse de données de réservation à partir d'un fichier CSV.
- **Model Performance**: Évaluation des métriques du modèle ANN (courbes d'apprentissage, matrice de confusion).
- **Authentication**: Système de connexion et d'inscription sécurisé (pages protégées).

## 📁 Project Structure / Structure du Projet

```text
hotel_ann_final/
├── app.py                          # Streamlit Entry Point
├── train.py                        # Model Training Script
├── requirements.txt                # Dependencies
├── css/
│   └── style.css                   # Global styles
├── assets/
│   └── hotel_bookings.csv          # Dataset (To download)
├── authentication/
│   └── users.json                  # Users database
├── models/                         # Trained Models & Scalers
├── notebook/
│   └── hotel_booking_analysis.ipynb
├── pages/                          # Streamlit Pages
└── utils/                          # Utility functions
```

## 🚀 Installation & Usage

### 1. Download Dataset / Télécharger le dataset
Download `hotel_bookings.csv` from Kaggle and place it in the `assets/` directory.

### 2. Install Dependencies / Installer les dépendances
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run Notebook / Exécuter le notebook (Optional)
If you want to re-train the models and generate new artifacts in `models/`:
```bash
jupyter notebook notebook/hotel_booking_analysis.ipynb
```

### 4. Launch Application / Lancer l'application
Run the Streamlit application:
```bash
streamlit run app.py
```

## 📊 Dataset Reference

Antonio, N., Almeida, A., & Nunes, L. (2019).
*Hotel booking demand datasets.*
Data in Brief, 22, 41-49.
Available on Kaggle: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---
*Created with ❤️*
