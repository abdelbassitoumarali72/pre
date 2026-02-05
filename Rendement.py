import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Ajouter un arrière-plan personnalisé avec image encodée en base64
import base64

def load_background_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    img_base64 = load_background_image("rendements-agricoles-scaled.jpg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpg;base64,{img_base64}');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("⚠️ Image d'arrière-plan non trouvée")


st.title("🏠 Rendement Agricole")

# Ajouter un style pour améliorer la lisibilité



# Chargement des données
df = pd.read_csv("Rendement_clin.csv")
# une stratégie de traitement des variables catégorielles
le_culture = LabelEncoder()
le_zone = LabelEncoder()
le_soil = LabelEncoder()
df["culture_type"] = le_culture.fit_transform(df["culture_type"])
df["zone"] = le_zone.fit_transform(df["zone"])
df["soil_type"] = le_soil.fit_transform(df["soil_type"])

X = df[["culture_type","rainfall", "fertilizer_quantity"]]
y = df["yield"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,) #random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
#y_pred = model.predict(x_test)
culture_user = st.selectbox("Type de culture", le_culture.classes_)
rainfall = st.number_input("Pluviométrie (mm)", min_value=0.0, step=10.0)
fertilizer_quantity = st.number_input("Quantité d'engrais (kg/ha)", min_value=0.0, step=10.0)
culture_num = le_culture.transform([culture_user])[0]

col1, col2, col3 = st.columns(3)
with col2:
    if st.button("🌾 PRÉDIRE LE RENDEMENT", use_container_width=True):
        input_data = np.array([[culture_num, rainfall, fertilizer_quantity]])
        prediction = model.predict(input_data)
        
if 'prediction' in locals():
    col_pred, col_conseil = st.columns(2, gap="large")
    
    if prediction[0] < 0:
        with col_pred:
            # Utilisez 'f' devant les guillemets et placez l'expression entre accolades {}
            st.success(f"{prediction[0]:.2f}")

            st.success("Rendement négatif")
        with col_conseil:    
            st.markdown("""
            ✅ Augmentez la pluviométrie
            
            ✅ Améliorez l'engrais
            
            ✅ Vérifiez la culture
            
            ✅ Consultez un agronome
            """, unsafe_allow_html=True)

    elif prediction[0] < 1000:
        with col_pred:
            st.success(f"{prediction[0]:.2f}")
        with col_conseil:
            st.markdown("""
            ✅ Augmentez la pluviométrie
            
            ✅ Améliorez l'engrais
            
            ✅ Vérifiez la culture
            
            ✅ Consultez un agronome
            """, unsafe_allow_html=True)
    
    else:
        st.success(f"{prediction[0]:.2f}")
        
     
