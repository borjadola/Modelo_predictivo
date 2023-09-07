# Modelo predictivo para fraude bancario
Proyecto final del bootcamp de Análisis de datos en IronHack

## Objetivo

Análisis exploratorio de los datos de una entidad financiera ficticia donde se ha detectado fraude en una serie de transacciones bancarias durante un intervalo de 30 días. En concreto en más de 6 millones de transacciones. 

Se han preparado para entrenar un modelo de machine learning de aprendizaje supervisado e identificar los casos de fraude. He utilizado tres tipos de modelos de clasificación.

En este entorno de streamlit podemos realizar consultas sobre transacciones en el apartado de "Modelo" y visualizar los resultados de los datos obtenidos.

IMAGEN1

![streamlit1](https://github.com/borjadola/Modelo_predictivo/assets/132678800/323cb69e-1d89-405a-97d9-1c41ec2a5d98)

![streamlit2](https://github.com/borjadola/Modelo_predictivo/assets/132678800/190baed6-b6b9-426d-b810-56c2d4cedace)

## Requisitos de sistema

Para la ejecución del entorno en la computadora deberán realizarse los siguientes pip install en nuestra terminal:

- pip install pandas
- pip install lottie
- pip install requests
- pip install streamlit
- pip install pickle
- pip install numpy

Ejecutaremos en nuestra terminal, una vez situados en la carpeta Model.py, "streamlit run Model.py" para abrir la web como local host.

## Fuentes

[DATASET](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data)

[RANDOM FOREST DOC](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[DECISION TREE DOC](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[LOGISTIC REGRESSION](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

[PANDAS LIBRARY](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)