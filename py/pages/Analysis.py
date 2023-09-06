import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from streamlit_lottie import st_lottie
import requests
import altair as alt
from PIL import Image
import pickle

archivos = ['../data/data_eda_total_partes/data_eda_total_parte_1.csv', 
            '../data/data_eda_total_partes/data_eda_total_parte_2.csv', 
            '../data/data_eda_total_partes/data_eda_total_parte_3.csv', 
            '../data/data_eda_total_partes/data_eda_total_parte_4.csv', 
            '../data/data_eda_total_partes/data_eda_total_parte_5.csv', 
            '../data/data_eda_total_partes/data_eda_total_parte_6.csv']
dataframes = [pd.read_csv(archivo) for archivo in archivos]
data_total = pd.concat(dataframes, ignore_index=True)
data_1= data_total.iloc[:500000]

# Función para cargar mi url de lottie
def load_lottieurl(url):
  r = requests.get(url)
  if r.status_code != 200:
    return None
  return r.json()

# Cargamos las imagenes que vamos a querer subir
roc = Image.open('../data/images/Curva_roc.png')
matrix = Image.open('../data/images/Matriz_confusion.png')
lottie_coding = load_lottieurl("https://lottie.host/83c0e8d8-892d-4909-8d6e-9e84888956f9/k1JaZ8212t.json")

st.title('Data Analysis')

st.dataframe(data_1.head())

add_selectbox = st.sidebar.selectbox('Choose the analysis',('Objetive', 'Transaction types', 
                                     'Density', 'Confussion matrix', 'ROC curve'))


if add_selectbox == 'Objetive':
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
    with left_column:
        st.header("My objetive")
        st.write(
        """
            illo que pasa
        """
        )
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

if add_selectbox == 'Transaction types':
    # Diagrama de barras donde se ven los tipos de transacción
    st.markdown('<p style="font-size:30px">Transaction Types</p>', unsafe_allow_html=True)
    st.bar_chart(data_total['Type'].value_counts().sort_values(ascending=False))
    st.write("---")
    data_fraud = data_total[data_total['Fraud'] == 1]
    st.markdown('<p style="font-size:30px">Fraudulent Transaction Types</p>', unsafe_allow_html=True)
    st.bar_chart(data_fraud['Type'].value_counts().sort_values(ascending=False))#, color=['#03BD81'])
    st.write("---")
    # Balanceo de transacciones fraudulentas
    st.markdown('<p style="font-size:30px">Transaction Balance</p>', unsafe_allow_html=True)
    count_fraud = data_total['Fraud'].value_counts()
    st.bar_chart(count_fraud)
    st.markdown('''Fraud percent: :blue[**0,13%**]''')

if add_selectbox == 'Density':
    # Gráfico de densidad
    non_fraud = data_total[data_total["Fraud"] == 0]["Amount"]
    fraud = data_total[data_total["Fraud"] == 1]["Amount"]

        # Crear un gráfico de densidad para ambas categorías
    chart_non_fraud = alt.Chart(pd.DataFrame({'Amount': non_fraud})).transform_density(
        'Amount', as_=['Amount', 'Density'], extent=[0, 2e6]).mark_area(opacity=0.5, 
                                                                        color="green").encode(
                                                                            x='Amount:Q',
                                                                            y='Density:Q')

    chart_fraud = alt.Chart(pd.DataFrame({'Amount': fraud})).transform_density(
        'Amount', as_=['Amount', 'Density'], extent=[0, 2e6]).mark_area(opacity=0.5, 
                                                                        color="orange").encode(
                                                                            x='Amount:Q',
                                                                            y='Density:Q')

        # Combinar los gráficos
    combined_chart = chart_non_fraud + chart_fraud

        # Mostrar el gráfico en Streamlit
    st.markdown('<p style="font-size:30px">Density Transactions</p>', unsafe_allow_html=True)
    st.altair_chart(combined_chart, use_container_width=True)

# Análisis del modelo. Imagenes
if add_selectbox == 'Confussion matrix':
    #Matrix
    st.markdown('<p style="font-size:40px">Random Forest Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:30px">Metrics</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label='', value='Not Fraud')
        st.metric(label = '', value='Fraud')
    with col2:
        st.metric(label='Precision', value='0,99')
        st.metric(label = '', value='0,99')
    with col3:
        st.metric(label='Recall', value='0,99')
        st.metric(label = '', value='1,00')
    with col4:
        st.metric(label='F1 Score', value='0,99')
        st.metric(label = '', value='0,99')
    st.markdown('<p style="font-size:30px">Confussion Matrix</p>', unsafe_allow_html=True)
    st.image(matrix, caption='Confussion Matrix')
if add_selectbox == 'ROC curve':
    #ROC
    st.markdown('<p style="font-size:40px">Random Forest Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:30px">ROC Curve</p>', unsafe_allow_html=True)
    st.image(roc, caption='ROC curve')
