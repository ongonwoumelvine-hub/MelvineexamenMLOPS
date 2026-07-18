import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="Détection de Billets", page_icon="💵", layout="wide")

# Style CSS pour un look moderne
st.markdown("""
    <style>
        .main { background-color: #f5f7f9; }
        .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("💵 Système de Détection de Billets")
st.markdown("### Analyse de conformité par Machine Learning")

# Sidebar pour l'upload
with st.sidebar:
    st.header("Paramètres")
    uploaded_file = st.file_uploader("Charger votre fichier CSV", type=["csv"])
    analyze_btn = st.button("Lancer l'analyse 🚀")

if uploaded_file and analyze_btn:
    # Appel à l'API FastAPI
    files = {"file": uploaded_file.getvalue()}
    try:
        with st.spinner("Analyse en cours..."):
            response = requests.post("https://melvineexamenmlops.onrender.com/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            # 1. Section Statistiques
            st.subheader("📊 Statistiques Globales")
            col1, col2, col3 = st.columns(3)
            col1.metric("Billets Analysés", result['statistiques']['total_analyses'])
            col2.metric("Vrais Billets", result['statistiques']['vrais_billets'])
            col3.metric("Faux Billets", result['statistiques']['faux_billets'])
            
            # 2. Visualisation des résultats
            st.divider()
            df_results = pd.DataFrame(result['table_predictions'])
            
            col_viz1, col_viz2 = st.columns([1, 1])
            with col_viz1:
                fig = px.pie(df_results, names='prediction', title="Répartition Globale", 
                             color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_viz2:
                fig_hist = px.histogram(df_results, x="score_confiance", nbins=20, title="Distribution des Scores de Confiance")
                st.plotly_chart(fig_hist, use_container_width=True)

            # 3. Tableau détaillé
            st.subheader("📋 Détails des prédictions")
            st.dataframe(df_results.style.format({"score_confiance": "{:.2f} %"}), use_container_width=True)
            
        else:
            st.error(f"Erreur API : {response.json().get('detail')}")
            
    except Exception as e:
        st.error(f"Impossible de joindre le serveur API : {e}")

elif not uploaded_file:
    st.info("Veuillez charger un fichier CSV pour commencer l'analyse.")