import io
from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
import requests
import json
import os

from app.models.model_utils import retrain_model_fine_tuning, retrain_model_full, retrain_model_incremental

# Crear el blueprint para la interfaz web
web_bp = Blueprint('web', __name__, template_folder='templates')

@web_bp.route('/', methods=['GET'])
def index():
    """Página principal"""
    return render_template('index.html')

@web_bp.route('/clasificate', methods=['GET', 'POST'])
def clasificate():
    """Página para Clasificar una noticia"""
    if request.method == 'POST':
        # Recoger datos del formulario
        titulo = request.form.get('titulo', '')
        descripcion = request.form.get('descripcion', '')
        fecha = request.form.get('fecha', '')
        
        if not descripcion:
            return render_template('clasificate.html', error="La descripción es obligatoria")
        
        # Crear payload para la API
        payload = {
            "Titulo": titulo,
            "Descripcion": descripcion,
            "Fecha": fecha
        }
        
        # Hacer petición a nuestra propia API
        api_url = request.host_url.rstrip('/') + url_for('api.predict_json')
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        
        if response.ok:
            result = response.json()[0]  # Tomar el primer resultado
            return render_template('result.html', result=result)
        else:
            error_msg = "Error al clasificar la noticia"
            if response.text:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg = error_data['error']
                except:
                    pass
            return render_template('clasificate.html', error=error_msg)
    
    return render_template('clasificate.html')

@web_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    """Página para cargar un CSV"""
    if request.method == 'POST':
        # Verificar si se ha subido un archivo
        if 'csv_file' not in request.files:
            return render_template('upload.html', error="No se ha subido ningún archivo")
        
        file = request.files['csv_file']
        
        if file.filename == '':
            return render_template('upload.html', error="No se ha seleccionado ningún archivo")
        
        if not file.filename.endswith('.csv'):
            return render_template('upload.html', error="El archivo debe ser un CSV")
        
        # Hacer petición a nuestra propia API
        api_url = request.host_url.rstrip('/') + url_for('api.predict_csv')
        files = {'file': (file.filename, file, 'text/csv')}
        
        response = requests.post(api_url, files=files)
        
        if response.ok:
            results = response.json()
            return render_template('results.html', results=results)
        else:
            error_msg = "Error al procesar el CSV"
            if response.text:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg = error_data['error']
                except:
                    pass
            return render_template('upload.html', error=error_msg)
    
    return render_template('upload.html')

@web_bp.route('/retrain', methods=['GET', 'POST'])
def retrain():
    """Página para reentrenar el modelo"""
    if request.method == 'POST':
        # Verificar si se ha subido un archivo
        if 'csv_file' not in request.files:
            return render_template('retrain.html', error="No se ha subido ningún archivo")
        
        csv_file = request.files['csv_file']
        retrain_type = request.form.get('retrain_type')
        
        # Verificar que el archivo tiene un nombre
        if csv_file.filename == '':
            return render_template('retrain.html', error="No se ha seleccionado ningún archivo")
        
        # Verificar que es un CSV
        if not csv_file.filename.endswith('.csv'):
            return render_template('retrain.html', error="El archivo debe ser un CSV")
        
        try:
            # Leer el CSV
            try:
                df_new = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')), sep=";")
            except:
                csv_file.seek(0)
                try:
                    df_new = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')), sep=",")
                except:
                    csv_file.seek(0)
                    df_new = pd.read_csv(io.StringIO(csv_file.read().decode('latin-1')), sep=";")
            
            # Verificar que contiene las columnas necesarias
            if 'Descripcion' not in df_new.columns or 'Label' not in df_new.columns:
                return render_template('retrain.html', 
                        error="El CSV debe contener las columnas 'Descripcion' y 'Label'")
            
            # Implementar la lógica de reentrenamiento según el tipo seleccionado
            if retrain_type == 'full':
                # Reentrenamiento completo
                result = retrain_model_full(df_new)
                
                if 'error' in result:
                    # Si hubo un error al cargar/entrenar el modelo
                    return render_template('retrain.html', 
                                        error=result['message'])
                else:
                    # Si todo salió bien
                    return render_template('retrain.html', 
                                        success=True, 
                                        message="Modelo reentrenado exitosamente con el método completo",
                                        metrics=result)
            elif retrain_type == 'incremental':
                # Reentrenamiento incremental
                result = retrain_model_incremental(df_new)
                
                if 'error' in result:
                    # Si hubo un error al cargar/entrenar el modelo
                    return render_template('retrain.html', 
                                        error=result['message'])
                else:
                    # Si todo salió bien
                    return render_template('retrain.html', 
                                        success=True, 
                                        message="Modelo reentrenado exitosamente con el método incremental",
                                        metrics=result)
            elif retrain_type == 'fine_tuning':
                # Ajuste fino
                result = retrain_model_fine_tuning(df_new)
                
                if 'error' in result:
                    # Si hubo un error al cargar/entrenar el modelo
                    return render_template('retrain.html', 
                                        error=result['message'])
                else:
                    # Si todo salió bien
                    return render_template('retrain.html', 
                                        success=True, 
                                        message="Modelo reentrenado exitosamente con el método de ajuste fino",
                                        metrics=result)
            else:
                return render_template('retrain.html', 
                                    error="Tipo de reentrenamiento no válido")
        
        
        except Exception as e:
            return render_template('retrain.html', 
                                error=f"Error en el procesamiento: {str(e)}")
    
    return render_template('retrain.html')