import streamlit as st
import joblib

ODS_INFO = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educacion de calidad",
    5: "Igualdad de genero",
    6: "Agua limpia y saneamiento",
    7: "Energia asequible y no contaminante",
    8: "Trabajo decente y crecimiento economico",
    9: "Industria, innovacion e infraestructura",
    10: "Reduccion de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Produccion y consumo responsables",
    13: "Accion por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones solidas",
    17: "Alianzas para lograr los objetivos",
}

def ods_label(ods_value) -> str:
    try:
        n = int(str(ods_value).strip())
        name = ODS_INFO.get(n, "")
        return f"ODS {n} - {name}".strip()
    except Exception:
        return str(ods_value)

def inject_light_theme_css():
    """
    Fondo blanco + texto oscuro + cards + uploader button blanco.
    """
    st.markdown(
        """
        <style>
          /* Fondo general */
          .stApp, section.main { background: #FFFFFF !important; }

          /* Texto base */
          .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div {
            color: #111827 !important;
          }

          /* Titulos */
          .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #0F172A !important;
          }

          /* Contenedor centrado */
          section.main > div { max-width: 1100px; margin: 0 auto; }

          /* Cards */
          div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 6px 20px rgba(0,0,0,0.06);
          }

          /* Inputs */
          textarea, input, .stTextInput input, .stSelectbox div, .stMultiSelect div {
            background: #FFFFFF !important;
            color: #111827 !important;
            border: 1px solid #E5E7EB !important;
          }

          /* Boton principal */
          .stButton > button[kind="primary"] {
            background: #E63946 !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 650 !important;
            height: 44px !important;
          }
          .stButton > button[kind="primary"]:hover { background: #C92A36 !important; }

          /* Boton secundario (blanco) */
          .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 10px !important;
            font-weight: 650 !important;
            height: 44px !important;
          }
          .stButton > button[kind="secondary"]:hover { background: #F3F4F6 !important; }

          /* Tabs */
          button[data-baseweb="tab"] { font-weight: 650; }

          /* File uploader: boton Browse files en blanco */
          div[data-testid="stFileUploader"] button {
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 10px !important;
            font-weight: 650 !important;
            height: 40px !important;
          }
          div[data-testid="stFileUploader"] button:hover {
            background: #F3F4F6 !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model(path="pipeline_calibrado.pkl"):
    return joblib.load(path)