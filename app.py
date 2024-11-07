from flask import Flask, jsonify, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Cargar el modelo entrenado y las columnas esperadas
with open("modelo_entrenado.pkl", "rb") as f:
    model, expected_columns = pickle.load(f)

def preprocess_data(df):
    # Mapeo para Family_Income
    income_mapping = {'Bajo': 1, 'Medio': 2, 'Alto': 3}
    if 'Family_Income' in df.columns:
        df['Family_Income'] = df['Family_Income'].map(income_mapping)
    else:
        raise KeyError("La columna 'Family_Income' no se encuentra en el dataset.")

    # Variables categóricas restantes
    categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level',
                           'Internet_Access', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                           'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    df = pd.get_dummies(df, columns=categorical_columns)

    # Asegúrate de que las columnas de entrada coincidan con las del modelo
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])
        input_data = preprocess_data(input_data)  # Asegúrate de que el preprocesamiento sea adecuado
        prediction = model.predict(input_data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5100)))
