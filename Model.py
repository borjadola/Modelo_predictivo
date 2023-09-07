import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Abrimos nuestras tablas modificadas

with open('../data/data_downsampled.pkl', 'rb') as dd:
    data_down = pickle.load(dd)


# Extraemos los archivos pickle de nuestros modelos

with open('../data/logistic_model.pkl', 'rb') as lo:
    logistic_model = pickle.load(lo)

with open('../data/tree_model.pkl', 'rb') as dt:
    tree_model = pickle.load(dt)

with open('../data/forest_model.pkl', 'rb') as rf:
    forest_model = pickle.load(rf)

def main():
    st.title('BANK FRAUD PREDICTOR')
    st.table(data_down.head())
    
    st.sidebar.header('INPUT MODEL')
   
    # funcion para clasificar fraude

    def classify(x):
        '''esta función clasifica en fraude o no fraude dependiendo si el valor
        booleano es 0 o 1'''
        if x == 1:
            return 'Fraud'
        else:
            return 'Not Fraud'
    
    # Función para introducir los parámetros
    
    def input_parameters():
        '''esta función nos generará una tabla donde introducir los valores
        que necesitamos para nuestro DF y implementar el modelo. Primero
        toma los datos y luego los convierte en un DF mediante dict'''
        time_hour = np.random.randint(1, 745)
        type = st.selectbox('Select transaction type', ('CASH_OUT', 'TRANSFER', 'DEBIT', 'CASH_IN', 'PAYMENT'), key='type')
        amount = st.number_input('Insert amount', key='amount')
        old_balance_org = st.number_input('Insert your initial account balance', key='old_balance_org')
        new_balance_org = st.number_input('Insert your final account balance', key='new_balance_org')
        old_balance_dest = st.number_input("Insert the recipient's initial balance", key='old_balance_dest')
        new_balance_dest = st.number_input("Insert the recipient's final balance", key='new_balance_dest')
        data_in = {'Time_hour': time_hour,
                    'Amount': amount,
                    'Old_balance_Org': old_balance_org,
                    'New_balance_Org': new_balance_org,
                    'Old_balance_Dest': old_balance_dest,
                    'New_balance_Dest': new_balance_dest}

        dict_type = {'CASH_OUT': [1,0,0,0],
                     'DEBIT': [0,1,0,0],
                    'PAYMENT': [0,0,1,0],
                    'TRANSFER': [0,0,0,1],
                    'CASH_IN': [0,0,0,0]}

        type_values = dict_type[type]
        column_names = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']  # Nombres de las nuevas columnas
        for col, value in zip(column_names, type_values):
            data_in[col] = value

        data_result = pd.DataFrame([data_in])

        return data_result
    
    df = input_parameters()
    
    # Para escoger un modelo

    model = st.sidebar.selectbox('Model to use:', ('Logistic Regression', 'Decision Tree', 'Random Forest'))

    st.subheader(model)
    st.write(df)

    # Creamos el boton para ejecutar cada modelo predictivo

    if st.button('Predict', type='primary', key='predict'):
        if model == 'Logistic Regression':
            st.success(classify(logistic_model.predict(df)))
        elif model == 'Decision Tree':
            st.success(classify(tree_model.predict(df)))
        else:
            st.success(classify(forest_model.predict(df)))

if __name__ == '__main__':
    main()
