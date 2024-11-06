import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import pickle

# Función para preprocesar los datos
def preprocess_data(df):
    # Mapeo para Family_Income
    income_mapping = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
    df['Family_Income'] = df['Family_Income'].map(income_mapping)

    # Variables categóricas restantes
    categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level',
                           'Internet_Access', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                           'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    df = pd.get_dummies(df, columns=categorical_columns)

    # Crear la columna Low_Performance
    df['Low_Performance'] = df['Exam_Score'].apply(lambda x: 1 if x < 70 else 0)

    return df

# Función para balancear los datos
def balance_data(df):
    # Separar características (X) y objetivo (y)
    X = df.drop(columns=['Exam_Score', 'Low_Performance'])
    y = df['Low_Performance']

    # Combinar X e y para el remuestreo
    data_combined = pd.concat([X, y], axis=1)

    # Separar clases mayoritaria y minoritaria
    majority = data_combined[data_combined['Low_Performance'] == 0]
    minority = data_combined[data_combined['Low_Performance'] == 1]

    # Verificar si la clase minoritaria tiene suficientes ejemplos para el sobremuestreo
    if len(minority) > 0:
        # Sobremuestreo de la clase minoritaria (sólo en entrenamiento)
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=22)
    else:
        raise ValueError("La clase minoritaria no tiene suficientes ejemplos para el sobremuestreo.")

    # Combinar clases
    data_balanced = pd.concat([majority, minority_upsampled])

    # Actualizar X y y
    X = data_balanced.drop('Low_Performance', axis=1)
    y = data_balanced['Low_Performance']

    return X, y

# Cargar el dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Preprocesar los datos
df_clean = preprocess_data(df)

# Balancear los datos
X, y = balance_data(df_clean)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=22)
model.fit(X_train, y_train)

# Guardar el modelo entrenado y las columnas esperadas
with open("modelo_entrenado.pkl", "wb") as f:
    pickle.dump((model, X_train.columns.tolist()), f)

print("Modelo entrenado y guardado en 'modelo_entrenado.pkl'")