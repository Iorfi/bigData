from flask import Flask, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

# Define tus funciones aquí (eda, preprocess_data, balance_data, train_models, feature_importance)

@app.route('/')
def index():
    # Aquí puedes llamar a tus funciones y devolver resultados
    # Por ejemplo, podrías devolver un resumen de los datos
    summary = df.describe().to_json()
    return jsonify(summary)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

