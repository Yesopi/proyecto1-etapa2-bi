import os
import joblib
import sys
import pandas as pd
import re
import unidecode
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Intentar cargar spacy
try:
    nlp = spacy.load("es_core_news_sm")
except:
    # Descargar el modelo si no está disponible
    import os
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# Definir todas las clases personalizadas que usa el modelo
class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, text_col='Descripcion'):
        self.text_col = text_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_copy = X.copy()
        df_copy['words'] = df_copy[self.text_col].apply(self._tokenize_text)
        return df_copy
    
    def _tokenize_text(self, text):
        if not isinstance(text, str):
            return []
        tokens = word_tokenize(text)
        return tokens

class TokenPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set(stopwords.words('spanish'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_copy = X.copy()
        df_copy['words'] = df_copy['words'].apply(self._preprocess_tokens)
        return df_copy
    
    def _preprocess_tokens(self, tokens):
        if not isinstance(tokens, list):
            return []
        tokens = self._to_lowercase(tokens)
        tokens = self._remove_punctuation(tokens)
        tokens = self._remove_non_ascii(tokens)
        tokens = self._remove_stopwords(tokens)
        return tokens
    
    def _to_lowercase(self, words):
        return [word.lower() for word in words if word is not None]
    
    def _remove_punctuation(self, words):
        new_words = []
        for word in words:
            if word is not None:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
        return new_words
    
    def _remove_non_ascii(self, words):
        new_words = []
        for word in words:
            if word is not None:
                new_word = unidecode.unidecode(word)
                new_words.append(new_word)
        return new_words
    
    def _remove_stopwords(self, words):
        new_words = []
        for word in words:
            if word not in self.stopwords:
                new_words.append(word)
        return new_words

class DuplicateHandler(BaseEstimator, TransformerMixin):
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
    def __init__(self):
        self.nlp = nlp
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_copy = X.copy()
        df_copy['words'] = df_copy['words'].apply(self._lemmatize_tokens)
        return df_copy
    
    def _lemmatize_tokens(self, tokens):
        if not isinstance(tokens, list):
            return []
        text = " ".join(tokens)
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_ if token.pos_ == "VERB" else token.text 
                            for token in doc]
        return lemmatized_tokens

class JoinTokens(BaseEstimator, TransformerMixin):
    def __init__(self, words_col='words'):
        self.words_col = words_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            tokens_lists = X[self.words_col]
        else:
            tokens_lists = X
        return [' '.join(tokens) if isinstance(tokens, list) else '' for tokens in tokens_lists]

# Función para cargar el modelo con el contexto correcto
def load_model_with_classes(model_path):
    """
    Carga el modelo asegurando que las clases personalizadas 
    estén en el espacio de nombres correcto
    """
    try:
        # Preparar el espacio de nombres para joblib
        import __main__
        __main__.TextTokenizer = TextTokenizer
        __main__.TokenPreprocessor = TokenPreprocessor
        __main__.DuplicateHandler = DuplicateHandler
        __main__.Lemmatizer = Lemmatizer
        __main__.JoinTokens = JoinTokens
        
        # Cargar el modelo
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise Exception(f"El modelo no existe. Asegúrate de que el archivo joblib esté en la ruta correcta: {model_path}")
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {str(e)}")