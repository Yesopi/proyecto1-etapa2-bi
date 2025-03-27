from config import MODEL_PATH
from app.models.model_loader import load_model_with_classes

def load_model():
    """Carga el modelo entrenado desde el archivo joblib con las clases personalizadas"""
    return load_model_with_classes(MODEL_PATH)

def retrain_model_full(df_new):
    from config import MODEL_PATH
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
    
    # Eliminar duplicados en los nuevos datos
    df_new = df_new.drop_duplicates(subset=['Descripcion'], keep='first')
    
    
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
        
        # Reentrenar el pipeline con los nuevos datos
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
def retrain_model_incremental(df_new):
    """
    Realiza un reentrenamiento incremental, usando solo los nuevos datos
    para actualizar el modelo existente.
    
    Args:
        df_new: DataFrame con nuevos datos de entrenamiento
        
    Returns:
        dict: Métricas de evaluación o dict con error
    """
    from config import MODEL_PATH
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
    
    # Eliminar duplicados en los nuevos datos
    df_new = df_new.drop_duplicates(subset=['Descripcion'], keep='first')
    
    # Asegurar que Label sea de tipo entero
    df_new['Label'] = df_new['Label'].astype(int)
    
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
        
        # Para el reentrenamiento incremental, solo necesitamos adaptar
        # el modelo al nuevo conjunto de datos
        if hasattr(pipeline, 'partial_fit'):
            # Si el modelo admite ajuste parcial (como SGDClassifier)
            pipeline.partial_fit(X_train, y_train)
        else:
            # Para modelos que no tienen partial_fit específicamente,
            # intentamos usar fit con warm_start si está disponible
            # Nota: Esto funciona principalmente con modelos basados en árboles como XGBoost
            if hasattr(pipeline.named_steps['classifier'], 'warm_start'):
                pipeline.named_steps['classifier'].warm_start = True
                pipeline.fit(X_train, y_train)
            else:
                # Si no podemos hacer ajuste incremental, recurrimos a un reentrenamiento normal
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