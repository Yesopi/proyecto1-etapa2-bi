from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from joblib import load, dump
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta al modelo
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_clasificacion_texto.joblib")

# Verificar si el modelo existe al inicio
if not os.path.exists(MODEL_PATH):
    logger.warning(f"¡El modelo no existe en {MODEL_PATH}! Asegúrate de entrenar el modelo primero.")

# Cargar el modelo
def get_model():
    try:
        model = load(MODEL_PATH)
        logger.info("Modelo cargado correctamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        return None

# Rutas de la aplicación
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        try:
            # Obtener datos de la petición
            request_data = request.get_json()
            
            if not request_data or 'textos' not in request_data:
                return jsonify({'error': 'Formato de solicitud inválido'}), 400
            
            textos = request_data['textos']
            
            # Cargar el modelo
            model = get_model()
            if model is None:
                return jsonify({'error': 'Error al cargar el modelo'}), 500
            
            # Preparar los datos para la predicción
            df = pd.DataFrame([{
                'Titulo': item.get('Titulo', ''),
                'Descripcion': item.get('Descripcion', '')
            } for item in textos])
            
            # Realizar predicciones
            predicciones = model.predict(df).tolist()
            
            return jsonify({"predicciones": predicciones})
        
        except Exception as e:
            logger.error(f"Error en la predicción: {e}")
            return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'GET':
        return render_template('retrain.html')
    else:
        try:
            # Obtener datos de la petición
            request_data = request.get_json()
            
            if not request_data or 'datos' not in request_data:
                return jsonify({'error': 'Formato de solicitud inválido'}), 400
            
            datos = request_data['datos']
            
            # Cargar el modelo actual para obtener el pipeline
            current_model = get_model()
            if current_model is None:
                return jsonify({'error': 'Error al cargar el modelo'}), 500
            
            # Preparar los datos para el reentrenamiento
            df = pd.DataFrame([{
                'Titulo': item.get('Titulo', ''),
                'Descripcion': item.get('Descripcion', ''),
                'Label': item.get('Label')
            } for item in datos])
            
            # Verificar que hay suficientes datos para el entrenamiento
            if len(df) < 10:
                return jsonify({'error': 'Se necesitan al menos 10 ejemplos para reentrenar el modelo'}), 400
            
            # Verificar que hay ejemplos de ambas clases
            if len(df['Label'].unique()) < 2:
                return jsonify({'error': 'Se necesitan ejemplos de ambas clases (0 y 1)'}), 400
            
            # Reentrenar el modelo con los nuevos datos
            X = df[['Titulo', 'Descripcion']]
            y = df['Label']
            
            # Ajustar el modelo (reentrenamiento)
            current_model.fit(X, y)
            
            # Calcular métricas (usando los mismos datos por simplicidad)
            from sklearn.metrics import classification_report, accuracy_score, f1_score
            
            predicciones = current_model.predict(X)
            accuracy = accuracy_score(y, predicciones)
            f1 = f1_score(y, predicciones, average='weighted')
            report = classification_report(y, predicciones, output_dict=True)
            
            # Guardar el modelo reentrenado
            dump(current_model, MODEL_PATH)
            logger.info("Modelo reentrenado y guardado correctamente")
            
            return jsonify({
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "classification_report": report,
                "message": "Modelo reentrenado y actualizado correctamente"
            })
        
        except Exception as e:
            logger.error(f"Error en el reentrenamiento: {e}")
            return jsonify({'error': f'Error en el reentrenamiento: {str(e)}'}), 500

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)