import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Asegurar recursos NLTK minimos (rapido si ya estan)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stemmer = SnowballStemmer("spanish")
stop_words = set(stopwords.words("spanish"))

# IMPORTANTE:
# Esta funcion debe llamarse EXACTO como en tu pipeline joblib (preprocessor=limpiar_texto_stem)
def limpiar_texto_stem(texto: str) -> str:
    if texto is None:
        return ""
    texto = str(texto)

    # 1) Normalizar y quitar acentos
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))

    # 2) Minusculas
    texto = texto.lower()

    # 3) Eliminar numeros
    texto = re.sub(r"\d+", " ", texto)

    # 4) Eliminar puntuacion (dejamos letras/espacios)
    texto = re.sub(r"[^a-zA-ZñÑáéíóúü\s]", " ", texto)

    # 5) Tokenizar
    tokens = word_tokenize(texto)

    # 6) Stopwords + tokens cortos
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # 7) Stemming
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)