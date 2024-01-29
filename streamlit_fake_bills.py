import streamlit as st
import pandas as pd
from fake_bill_prediction import predict

st.title('Détection des faux billets')
st.markdown('Application de détection des faux billets basée sur un modèle de régression logistique')

st.header("Veuillez télécharger un fichier csv ';' contenant les dimensions (mm) des billets...")

uploaded_file = st.file_uploader("Sélectionnez un fichier csv", type="csv")
#uploaded_file = pd.read_csv("/home/papa/DataspellProjects/openclassrooms/OpenclassRooms/Projet 10/Data_transformed/billets_compl.csv", sep=';')
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        X = df[['length', 'margin_low', 'margin_up']].copy()
        st.write(X.sample(10))
    except:
        st.error("Vérifier le format du fichier !")
else:
    st.warning("you need to upload a csv file")

if st.button('Détection'):
    #try:
        result = predict(X)
        st.write(result.sample(10))
    #except:
        #st.error("Erreur lors de la prédiction")
