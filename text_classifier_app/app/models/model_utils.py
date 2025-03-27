from config import MODEL_PATH
from app.models.model_loader import load_model_with_classes

def load_model():
    """Carga el modelo entrenado desde el archivo joblib con las clases personalizadas"""
    return load_model_with_classes(MODEL_PATH)