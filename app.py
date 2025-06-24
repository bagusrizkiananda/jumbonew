# app.py
import streamlit as st
import pickle
import pandas as pd

# Load Model
with open('naive_bayes_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.set_page_config(page_title="Aplikasi Analisis Sentimen", layout="wide")

st.title("Analisis Sentimen Review")

st.write("Masukkan teks untuk dianalisis sentimennya:")

input_text = st.text_area("Teks Review", "")

if st.button("Analisis"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        # Prediksi
        prediction = model.predict([input_text])[0]
        st.write(f"**Hasil Analisis Sentimen:** {prediction}")
        if prediction == 'Positive':
            st.success("Sentimen Positif 😊")
        else:
            st.error("Sentimen Negatif 😞")

# Footer
st.markdown("---")
st.markdown("Aplikasi ini dibuat untuk keperluan analisis sentimen menggunakan model Naive Bayes.")
