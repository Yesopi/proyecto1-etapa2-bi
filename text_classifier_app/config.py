import os

# Rutas de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, 'app')
DATA_DIR = os.path.join(APP_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')

# Ruta del modelo
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_clasificacion_texto.joblib')

# Configuración de la aplicación Flask
DEBUG = True
PORT = 5000