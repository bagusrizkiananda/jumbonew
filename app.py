# app.py
import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
import gzip

# Load dataset
film_df = pd.read_csv('film_jumbo.csv')
stemming_df = pd.read_csv('StemmingJumbo.csv')
preprocessed_df = pd.read_csv('data_preprocessed.csv')
classification_results = pd.read_csv('naivebayes_classification_results.csv')

# Load Naive Bayes Model
with gzip.open('naive_bayes_classifier.pkl.gz', 'rb') as file:
    nb_model = pickle.load(file)

# Buat stopword manual
stopwords = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'untuk', 'dengan', 'adalah', 'itu',
    'ini', 'karena', 'atau', 'saya', 'kamu', 'dia', 'mereka', 'kita', 'dalam', 'tidak',
    'bukan', 'akan', 'telah', 'sudah', 'belum', 'bisa', 'dapat'
])

def preprocessing(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in stopwords]
    return ' '.join(filtered)

def full_preprocess(text):
    text = preprocessing(text)
    text = remove_stopwords(text)
    return text

# Streamlit Interface
st.title('Sentiment Analysis & Text Processing Dashboard')

menu = st.sidebar.selectbox('Menu', ['Dataset', 'Preprocessing', 'TF-IDF', 'Klasifikasi'])

if menu == 'Dataset':
    st.subheader('Dataset Film Jumbo')
    st.write(film_df)

    st.subheader('Dataset Stemming Jumbo')
    st.write(stemming_df)

    st.subheader('Dataset Preprocessed')
    st.write(preprocessed_df)

    st.subheader('Hasil Klasifikasi Naive Bayes')
    st.write(classification_results)

elif menu == 'Preprocessing':
    st.subheader('Text Preprocessing')
    text_input = st.text_area('Masukkan kalimat:')
    if st.button('Proses'):
        if text_input:
            clean = preprocessing(text_input)
            stop_removed = remove_stopwords(clean)
            st.write('Hasil Cleaning:', clean)
            st.write('Hasil Stopword Removal:', stop_removed)
        else:
            st.warning('Mohon masukkan kalimat terlebih dahulu.')

elif menu == 'TF-IDF':
    st.subheader('TF-IDF Vectorization')

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_df['preprocessed_text'])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    st.write('TF-IDF Matrix:')
    st.write(tfidf_df)

elif menu == 'Klasifikasi':
    st.subheader('Klasifikasi Sentimen')

    text_input = st.text_area('Masukkan kalimat untuk diklasifikasi:')
    if st.button('Klasifikasikan'):
        if text_input:
            processed = full_preprocess(text_input)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(preprocessed_df['preprocessed_text'])
            nb_model = MultinomialNB()
            nb_model.fit(X, preprocessed_df['label'])

            X_input = vectorizer.transform([processed])
            prediction = nb_model.predict(X_input)[0]

            st.write(f'Hasil Prediksi Sentimen: **{prediction}**')
        else:
            st.warning('Mohon masukkan kalimat terlebih dahulu.')
