from config import MODEL_PATH
from app.models.model_loader import load_model_with_classes
from config import MODEL_PATH, ORIGINAL_DATA_PATH
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
def load_model():
    """Carga el modelo entrenado desde el archivo joblib con las clases personalizadas"""
    return load_model_with_classes(MODEL_PATH)

def retrain_model_full(df_new):
    """
    Realiza un reentrenamiento completo, combinando datos originales y nuevos
    
    Args:
        df_new: DataFrame con nuevos datos de entrenamiento
        
    Returns:
        dict: Métricas de evaluación o dict con error
    """
    
    try:
        # Cargar los datos originales si existen
        df_original = None
        if os.path.exists(ORIGINAL_DATA_PATH):
            try:
                df_original = pd.read_csv(ORIGINAL_DATA_PATH, sep=";")
                print(f"Datos originales cargados: {len(df_original)} filas")
            except Exception as e:
                print(f"Error al cargar datos originales: {str(e)}")
        
        # Eliminar duplicados en los nuevos datos
        df_new = df_new.drop_duplicates(subset=['Descripcion'], keep='first')
        
        # Asegurar que Label sea de tipo entero
        df_new['Label'] = df_new['Label'].astype(int)
        
        # Combinar datos originales y nuevos si existen datos originales
        if df_original is not None:
            df_combined = pd.concat([df_original, df_new], ignore_index=True)
            # Eliminar posibles duplicados
            df_combined = df_combined.drop_duplicates(subset=['Descripcion'], keep='first')
            print(f"Conjunto de datos combinado: {len(df_combined)} filas")
        else:
            df_combined = df_new
            print(f"Usando solo datos nuevos: {len(df_combined)} filas")
            
            # Guardar estos datos como originales para futuros reentrenamientos
            os.makedirs(os.path.dirname(ORIGINAL_DATA_PATH), exist_ok=True)
            df_new.to_csv(ORIGINAL_DATA_PATH, sep=';', index=False)
        
        # Dividir en entrenamiento y prueba
        df_train, df_test = train_test_split(
            df_combined, test_size=0.2, random_state=42, stratify=df_combined['Label']
        )
        
        # Preparar X e y para entrenamiento
        X_train = df_train[['Descripcion']].copy()
        y_train = df_train['Label']
        
        # Preparar X e y para prueba
        X_test = df_test[['Descripcion']].copy()
        y_test = df_test['Label']
        
        # Cargar el pipeline existente
        pipeline = load_model()
        
        # Reentrenar el pipeline con TODOS los datos (originales + nuevos)
        pipeline.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = pipeline.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Guardar el modelo actualizado
        joblib.dump(pipeline, MODEL_PATH)
        
        # Si no había datos originales antes, guardar el conjunto combinado
        if df_original is None:
            df_combined.to_csv(ORIGINAL_DATA_PATH, sep=';', index=False)
        else:
            # Actualizar los datos originales con el conjunto combinado
            df_combined.to_csv(ORIGINAL_DATA_PATH, sep=';', index=False)
        
        # Devolver métricas
        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'samples_new': len(df_new),
            'samples_total': len(df_combined)
        }
        
        return metrics
        
    except Exception as e:
        # Si hay algún error, devolver un mensaje
        return {
            'error': True,
            'message': f"No se pudo completar el reentrenamiento: {str(e)}"
        }
def retrain_model_incremental(df_new):
    
    # Eliminar duplicados en los nuevos datos
    df_new = df_new.drop_duplicates(subset=['Descripcion'], keep='first')
    
    # Asegurar que Label sea de tipo entero
    df_new['Label'] = df_new['Label'].astype(int)
    
    # Reemplazar los datos originales con los nuevos datos
    os.makedirs(os.path.dirname(ORIGINAL_DATA_PATH), exist_ok=True)
    df_new.to_csv(ORIGINAL_DATA_PATH, sep=';', index=False)
    
    # Dividir en entrenamiento y prueba
    df_train, df_test = train_test_split(
        df_new, test_size=0.2, random_state=42, stratify=df_new['Label']
    )
    
    # Preparar X e y para entrenamiento
    X_train = df_train[['Descripcion']].copy()
    y_train = df_train['Label']
    
    # Preparar X e y para prueba
    X_test = df_test[['Descripcion']].copy()
    y_test = df_test['Label']
    
    try:
        # Cargar el pipeline existente
        pipeline = load_model()
        
        # Entrenar el pipeline con los nuevos datos únicamente
        pipeline.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = pipeline.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Guardar el modelo actualizado
        joblib.dump(pipeline, MODEL_PATH)
        
        # Devolver métricas
        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'samples': len(df_new)
        }
        
        return metrics
        
    except Exception as e:
        # Si hay algún error, devolver un mensaje
        return {
            'error': True,
            'message': f"No se pudo cargar o entrenar el modelo: {str(e)}"
        }

def retrain_model_fine_tuning(df_new):
    print(f"Datos de df new: {len(df_new)} filas")
    # Eliminar duplicados en los nuevos datos
    df_new = df_new.drop_duplicates(subset=['Descripcion'], keep='first')
    print(f"Datos de df new: {len(df_new)} filas")
    
    # Cargar los datos originales y combinarlos con los nuevos
    total_samples = len(df_new)
    if os.path.exists(ORIGINAL_DATA_PATH):
        try:
            df_original = pd.read_csv(ORIGINAL_DATA_PATH, sep=";")
            df_combined = pd.concat([df_original, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Descripcion'], keep='first')
            # Guardar datos combinados
            df_combined.to_csv(ORIGINAL_DATA_PATH, sep=';', index=False)
            total_samples = len(df_combined)
        except Exception as e:
            # Si hay error, guardar solo los nuevos datos
            df_new.to_csv(ORIGINAL_DATA_PATH, sep=';', index=False)
    else:
        # Si no existen datos originales, guardar los nuevos
        os.makedirs(os.path.dirname(ORIGINAL_DATA_PATH), exist_ok=True)
        df_new.to_csv(ORIGINAL_DATA_PATH, sep=';', index=False)
    print(f"Total de muestras: {total_samples}")
    # Dividir en entrenamiento y prueba
    df_train, df_test = train_test_split(
        df_combined, test_size=0.2, random_state=42, stratify=df_combined['Label']
    )
    print(f"Datos de df new: {len(df_new)} filas")
    # Preparar X e y para entrenamiento
    X_train = df_train[['Descripcion']].copy()
    y_train = df_train['Label']
    
    # Preparar X e y para prueba
    X_test = df_test[['Descripcion']].copy()
    y_test = df_test['Label']
    
    try:
        # Cargar el pipeline existente
        pipeline = load_model()
        
        # Reducir la tasa de aprendizaje para el ajuste fino
        if hasattr(pipeline.named_steps['classifier'], 'learning_rate'):
            print("El clasificador tiene tasa de aprendizaje configurable")
            # Guardar la tasa de aprendizaje original
            original_lr = pipeline.named_steps['classifier'].learning_rate
            # Reducir la tasa de aprendizaje a un 10% de la original para ajuste fino
            pipeline.named_steps['classifier'].learning_rate = original_lr * 0.1
            print(X_train.shape)
            # Entrenar con la tasa de aprendizaje reducida
            pipeline.fit(X_train, y_train)
            
            # Restaurar la tasa de aprendizaje original después del entrenamiento
            pipeline.named_steps['classifier'].learning_rate = original_lr
        
        # Evaluar el modelo
        y_pred = pipeline.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Guardar el modelo actualizado
        joblib.dump(pipeline, MODEL_PATH)
        
        # Devolver métricas
        metrics = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'samples_new': len(df_new),
            'samples_total': total_samples
        }
        
        return metrics
        
    except Exception as e:
        # Si hay algún error, devolver un mensaje
        return {
            'error': True,
            'message': f"No se pudo cargar o entrenar el modelo: {str(e)}"
        }