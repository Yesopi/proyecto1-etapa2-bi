import re
import unidecode
import spacy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Intentar cargar spacy
try:
    nlp = spacy.load("es_core_news_sm")
except:
    # No fallar si no está instalado, el error se manejará en las clases
    nlp = None

class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, text_col='Descripcion'):
        self.text_col = text_col
        try:
            word_tokenize("Prueba")
        except:
            import nltk
            nltk.download('punkt')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_copy = X.copy()
        
        df_copy['words'] = df_copy[self.text_col].apply(self._tokenize_text)
        
        return df_copy
    
    def _tokenize_text(self, text):
        if not isinstance(text, str):
            return []
        
        # Tokenizar el texto
        tokens = word_tokenize(text)
        return tokens

class TokenPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para preprocesar tokens ya existentes
    """
    def __init__(self):
        try:
            self.stopwords = set(stopwords.words('spanish'))
        except:
            import nltk
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('spanish'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Crear una copia del DataFrame para no modificar el original
        df_copy = X.copy()
        
        # Aplicar el preprocesamiento a la columna 'words'
        df_copy['words'] = df_copy['words'].apply(self._preprocess_tokens)
        
        return df_copy
    
    def _preprocess_tokens(self, tokens):
        """Preprocesa una lista de tokens"""
        if not isinstance(tokens, list):
            return []
        
        # Convertir a minúsculas
        tokens = self._to_lowercase(tokens)
        
        # Eliminar puntuación
        tokens = self._remove_punctuation(tokens)
        
        # Eliminar caracteres no ASCII
        tokens = self._remove_non_ascii(tokens)
        
        # Eliminar stopwords
        tokens = self._remove_stopwords(tokens)
        
        return tokens
    
    def _to_lowercase(self, words):
        """Convertir a minúsculas"""
        return [word.lower() for word in words if word is not None]
    
    def _remove_punctuation(self, words):
        """Eliminar puntuación"""
        new_words = []
        for word in words:
            if word is not None:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
        return new_words
    
    def _remove_non_ascii(self, words):
        """Eliminar caracteres no ASCII"""
        new_words = []
        for word in words:
            if word is not None:
                new_word = unidecode.unidecode(word)
                new_words.append(new_word)
        return new_words
    
    def _remove_stopwords(self, words):
        """Eliminar stopwords"""
        new_words = []
        for word in words:
            if word not in self.stopwords:
                new_words.append(word)
        return new_words

class DuplicateHandler(BaseEstimator, TransformerMixin):
    """
    Transformador que maneja duplicados para entrenamiento y es seguro para predicción
    """
    def __init__(self, words_col='words', label_col='Label'):
        self.words_col = words_col
        self.label_col = label_col
        self.keep_indices_ = None
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return X

class Lemmatizer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para realizar la lematización de verbos
    usando spaCy para una columna de tokens
    """
    def __init__(self):
        # Asegurarnos de que el modelo de spaCy está cargado
        global nlp
        if nlp is None:
            try:
                nlp = spacy.load("es_core_news_sm")
            except:
                import sys
                import os
                os.system("python -m spacy download es_core_news_sm")
                nlp = spacy.load("es_core_news_sm")
        self.nlp = nlp
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Crear copia del DataFrame
        df_copy = X.copy()
        
        # Aplicar lematización a la columna 'words'
        df_copy['words'] = df_copy['words'].apply(self._lemmatize_tokens)
        
        return df_copy
    
    def _lemmatize_tokens(self, tokens):
        """Lematiza una lista de tokens"""
        if not isinstance(tokens, list):
            return []
        
        # Unir los tokens en una cadena para procesarlos con spaCy
        text = " ".join(tokens)
        
        # Procesar el texto con spaCy
        doc = self.nlp(text)
        
        # Lematizar, enfocándose en los verbos
        lemmatized_tokens = [token.lemma_ if token.pos_ == "VERB" else token.text 
                            for token in doc]
        
        return lemmatized_tokens

class JoinTokens(BaseEstimator, TransformerMixin):
    """
    Transformador para unir tokens en una cadena de texto
    para preparar los datos para la vectorización
    """
    def __init__(self, words_col='words'):
        self.words_col = words_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Obtenemos solo la columna de tokens
        if isinstance(X, pd.DataFrame):
            # Si es un DataFrame, extraemos solo la columna de tokens
            tokens_lists = X[self.words_col]
        else:
            # Si ya es una serie o lista
            tokens_lists = X
            
        # Unimos los tokens en cadenas de texto
        return [' '.join(tokens) if isinstance(tokens, list) else '' for tokens in tokens_lists]