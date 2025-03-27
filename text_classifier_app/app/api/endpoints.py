from flask import Blueprint, request, jsonify
import pandas as pd
from app.models.model_utils import load_model
import io

# Crear un Blueprint para los endpoints de la API
api_bp = Blueprint('api', __name__)

@api_bp.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    Endpoint para realizar predicciones a partir de un archivo CSV.
    
    Recibe: Archivo CSV con los datos de noticias
    Devuelve: JSON con predicciones (0: verdadera, 1: falsa) y probabilidades
    """
    try:
        # Verificar si se ha subido un archivo
        if 'file' not in request.files:
            return jsonify({"error": "No se ha subido ningún archivo"}), 400
        
        csv_file = request.files['file']
        
        # Verificar que el archivo tiene un nombre
        if csv_file.filename == '':
            return jsonify({"error": "No se ha seleccionado ningún archivo"}), 400
        
        # Verificar que es un CSV
        if not csv_file.filename.endswith('.csv'):
            return jsonify({"error": "El archivo debe ser un CSV"}), 400
        
        # Leer el CSV directamente desde el objeto de archivo
        try:
            # Intentar diferentes codificaciones y delimitadores
            df = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')), sep=";")
        except:
            try:
                # Reiniciar el puntero del archivo y probar con otra codificación/separador
                csv_file.seek(0)
                df = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')), sep=",")
            except:
                try:
                    # Última alternativa
                    csv_file.seek(0)
                    df = pd.read_csv(io.StringIO(csv_file.read().decode('latin-1')), sep=";")
                except Exception as e:
                    return jsonify({"error": f"Error al leer el CSV: {str(e)}"}), 400
        
        # Verificar que el DataFrame contiene la columna necesaria para la predicción
        if 'Descripcion' not in df.columns:
            return jsonify({"error": "El CSV debe contener la columna 'Descripcion'"}), 400
        
        # Cargar el modelo
        model = load_model()
        
        # Realizar predicción
        predictions = model.predict(df)
        
        # Obtener probabilidades
        probabilities = model.predict_proba(df)
        
        # Crear JSON de resultados, manteniendo el esquema original del CSV
        results = []
        for i, row in df.iterrows():
            result = {
                'ID': row.get('ID', f'row_{i}'),
                'Label': int(predictions[i]),
                'Titulo': row.get('Titulo', ''),
                'Descripcion': row.get('Descripcion', ''),
                'Fecha': row.get('Fecha', ''),
                'Probabilidad_Clase_0': float(probabilities[i][0]),  # Probabilidad de clase 0 (verdadera)
                'Probabilidad_Clase_1': float(probabilities[i][1])   # Probabilidad de clase 1 (falsa)
            }
            results.append(result)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Error en el procesamiento: {str(e)}"}), 500


@api_bp.route('/predict_json', methods=['POST'])
def predict_json():
    """
    Endpoint para realizar predicciones a partir de un JSON.
    
    Recibe: JSON con una o más noticias
    Devuelve: JSON con predicciones (0: verdadera, 1: falsa) y probabilidades
    
    Ejemplo de solicitud:
    [
        {
            "Titulo": "Título de la noticia 1",
            "Descripcion": "Texto de la noticia 1...",
            "Fecha": "01/01/2023"
        },
        {
            "Titulo": "Título de la noticia 2",
            "Descripcion": "Texto de la noticia 2...",
            "Fecha": "02/01/2023"
        }
    ]
    """
    try:
        # Obtener los datos JSON del request
        data = request.get_json()
        
        # Verificar que hay datos
        if not data:
            return jsonify({"error": "No se proporcionaron datos"}), 400
        
        # Si es un solo objeto (no una lista), convertirlo a lista
        if isinstance(data, dict):
            data = [data]
        
        # Convertir a DataFrame
        df = pd.DataFrame(data)
        
        # Verificar que el DataFrame contiene la columna necesaria para la predicción
        if 'Descripcion' not in df.columns:
            return jsonify({"error": "Los datos deben contener la columna 'Descripcion'"}), 400
        
        # Cargar el modelo
        model = load_model()
        
        # Realizar predicción
        predictions = model.predict(df)
        
        # Obtener probabilidades
        probabilities = model.predict_proba(df)
        
        # Crear JSON de resultados
        results = []
        for i, row in df.iterrows():
            result = {
                'ID': row.get('ID', f'row_{i}'),
                'Label': int(predictions[i]),
                'Titulo': row.get('Titulo', ''),
                'Descripcion': row.get('Descripcion', ''),
                'Fecha': row.get('Fecha', ''),
                'Probabilidad_Clase_0': float(probabilities[i][0]),  # Probabilidad de clase 0 (verdadera)
                'Probabilidad_Clase_1': float(probabilities[i][1])   # Probabilidad de clase 1 (falsa)
            }
            results.append(result)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Error en el procesamiento: {str(e)}"}), 500





@api_bp.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return jsonify({"status": "API funcionando correctamente"})