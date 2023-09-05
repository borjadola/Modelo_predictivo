import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import altair as alt
from PIL import Image
import pickle

archivos = ['../data/data_eda_total_parte_1.csv', '../data/data_eda_total_parte_2.csv', '../data/data_eda_total_parte_3.csv', '../data/data_eda_total_parte_4.csv', '../data/data_eda_total_parte_5.csv', '../data/data_eda_total_parte_6.csv']
dataframes = [pd.read_csv(archivo) for archivo in archivos]
data_total = pd.concat(dataframes, ignore_index=True)

data_1= data_total.iloc[:500000]

st.title('Data Analysis')

st.dataframe(data_1.head())

roc = Image.open('../data/images/Curva_roc.png')
matrix = Image.open('../data/images/Matriz_confusion.png')

# Diagrama de barras donde se ven los tipos de transacción
st.markdown('<p style="font-size:30px">Transaction Types</p>', unsafe_allow_html=True)
st.bar_chart(data_total['Type'].value_counts().sort_values(ascending=False))

data_fraud = data_total[data_total['Fraud'] == 1]
st.markdown('<p style="font-size:30px">Fraudulent Transaction Types</p>', unsafe_allow_html=True)
st.bar_chart(data_fraud['Type'].value_counts().sort_values(ascending=False))#, color=['#03BD81'])

# Balanceo de transacciones fraudulentas
st.markdown('<p style="font-size:30px">Transaction Balance</p>', unsafe_allow_html=True)
count_fraud = data_total['Fraud'].value_counts()
st.bar_chart(count_fraud)
st.markdown('''Fraud percent: :blue[**0,13%**]''')

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

st.markdown('<p style="font-size:40px">Random Forest Analysis</p>', unsafe_allow_html=True)

st.markdown('<p style="font-size:30px">Confussion Matrix</p>', unsafe_allow_html=True)
st.image(matrix, caption='Confussion Matrix')
st.markdown('<p style="font-size:30px">ROC Curve</p>', unsafe_allow_html=True)
st.image(roc, caption='ROC curve')
