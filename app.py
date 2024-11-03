from flask import Flask, jsonify, request
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import scipy.stats as stats
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

# Cargar el dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Funciones del notebook

def eda(df):
    # Análisis exploratorio de datos
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    # Convertir variables categóricas a tipo 'category' si es necesario
    categorical_vars = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                        'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                        'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    for var in categorical_vars:
        df[var] = df[var].astype('category')

    # Correlación para variables numéricas
    numeric_vars = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_vars.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación entre Variables Numéricas')
    plt.show()

    # Análisis de variables categóricas con boxplots
    sns.boxplot(x='Parental_Involvement', y='Exam_Score', data=df)
    plt.title('Exam Score vs Parental Involvement')
    plt.show()
    
    g = sns.jointplot(x='Hours_Studied', y='Exam_Score', data=df, kind='reg')

    # Cambiar el color de la línea de regresión a rojo
    g.ax_joint.get_lines()[0].set_color('red')

    # Actualizar los histogramas marginales a azul
    g.ax_marg_x.hist(df['Hours_Studied'], alpha=0.6)
    g.ax_marg_y.hist(df['Exam_Score'], alpha=0.6, orientation="horizontal")

    plt.show()

    sns.boxplot(x='Attendance', y='Exam_Score', data=df)
    plt.title('Puntaje de Examen en Relación a la Asistencia')
    plt.xlabel('Asistencia')
    plt.ylabel('Puntaje de Examen')
    plt.show()

    # Prueba t para comparar medias
    low_motivation = df[df['Motivation_Level'] == 'Low']['Exam_Score']
    high_motivation = df[df['Motivation_Level'] == 'High']['Exam_Score']
    t_stat, p_value = stats.ttest_ind(low_motivation, high_motivation)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")

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

def balance_data(df):
    # Separar características (X) y objetivo (y)
    X = df.drop(columns=['Exam_Score', 'Low_Performance'])
    y = df['Low_Performance']

    # Combinar X e y para el remuestreo
    data_combined = pd.concat([X, y], axis=1)

    # Separar clases mayoritaria y minoritaria
    majority = data_combined[data_combined['Low_Performance'] == 0]
    minority = data_combined[data_combined['Low_Performance'] == 1]

    # Sobremuestreo de la clase minoritaria (sólo en entrenamiento)
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=22)

    # Combinar clases
    data_balanced = pd.concat([majority, minority_upsampled])

    # Actualizar X y y
    X = data_balanced.drop('Low_Performance', axis=1)
    y = data_balanced['Low_Performance']

    return X, y

def train_models(X_train, y_train, X_test, y_test):
    # Crear y ajustar el modelo
    hgb_model = HistGradientBoostingClassifier()
    hgb_model.fit(X_train, y_train)

    # Hacer predicciones
    hgb_predictions = hgb_model.predict(X_test)

    # Calcular la precisión
    hgb_accuracy = accuracy_score(y_test, hgb_predictions)
    print(f"\nHistGradientBoostingClassifier Precisión del modelo: {hgb_accuracy:.2f}")

    # Imprimir el reporte de clasificación
    print("\nReporte de clasificación (HistGradientBoostingClassifier):")
    print(classification_report(y_test, hgb_predictions))

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=22)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("\nRandom Forest Precisión del modelo: {:.2f}".format(accuracy_score(y_test, y_pred_rf)))
    print("\nReporte de clasificación (Random Forest):")
    print(classification_report(y_test, y_pred_rf))

    # Matriz de confusión Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Random Forest - Matriz de Confusión')
    plt.show()

    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=22))
    ])

    # Ajustar el pipeline en lugar del modelo de regresión logística directamente
    lr_pipeline.fit(X_train, y_train)
    
    # Hacer predicciones usando el pipeline
    lr_predictions = lr_pipeline.predict(X_test)
    
    # Calcular y mostrar la precisión
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"\nRegresión Logística Precisión del modelo: {lr_accuracy:.2f}")
    
    # Mostrar el reporte de clasificación
    print("\nReporte de clasificación (Regresión Logística):")
    print(classification_report(y_test, lr_predictions))

    # Matriz de confusión Regresión Logística
    cm_lr = confusion_matrix(y_test, lr_predictions)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Regresión Logística - Matriz de Confusión')
    plt.show()

    # Crear un imputador
    imputer = SimpleImputer(strategy='mean')

    # Aplicar el imputador a los datos de entrenamiento y prueba
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Ahora usa X_train_imputed y X_test_imputed para el modelo SVM
    svm_model = SVC()
    svm_model.fit(X_train_imputed, y_train)
    svm_predictions = svm_model.predict(X_test_imputed)

    # Matriz de confusión SVM
    cm_svm = confusion_matrix(y_test, svm_predictions)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('SVM - Matriz de Confusión')
    plt.show()

def feature_importance(model, X):
    # Importancia de las características (Random Forest)
    importances = model.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Visualizar las importancias de las variables
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Importancia de las Variables (Random Forest)')
    plt.show()

# Preprocesar los datos
df_clean = preprocess_data(df)

# Balancear los datos
X, y = balance_data(df_clean)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Entrenar y evaluar modelos
train_models(X_train, y_train, X_test, y_test)

# Visualizar importancia de características (para el modelo Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=22)
rf_model.fit(X_train, y_train)
feature_importance(rf_model, X)

@app.route('/')
def index():
    summary = df.describe().to_json()
    return jsonify(summary)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])
        input_data = preprocess_data(input_data)  # Asegúrate de que el preprocesamiento sea adecuado
        prediction = rf_model.predict(input_data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))