import numpy as np
import pandas as pd
from helpers import ods_label

def predict_one(model, classes, texto: str):
    probs = model.predict_proba([texto])[0]
    top_idx = np.argsort(probs)[::-1]
    pred = classes[top_idx[0]]
    return pred, probs, top_idx

def make_topk_df(classes, probs, top_idx, k=5):
    idx = top_idx[:k]
    return pd.DataFrame({
        "ODS": [ods_label(classes[i]) for i in idx],
        "probabilidad": probs[idx]
    })

def probs_df(classes, probs):
    df = pd.DataFrame({
        "ODS": [ods_label(c) for c in classes],
        "probabilidad": probs
    })
    return df.sort_values("probabilidad", ascending=False)

def margin_confidence(probs):
    sp = np.sort(probs)[::-1]
    return float(sp[0] - sp[1]) if len(sp) > 1 else 0.0

def _ods_num_from_label(label: str) -> str:
    """
    'ODS 6 - Agua limpia...' -> '6'
    """
    s = str(label)
    if "ODS" in s:
        s = s.replace("ODS", "").split("-")[0].strip()
    return s

def plot_probs(st, df_plot: pd.DataFrame):
    """
    Grafica horizontal limpia:
    - Fondo blanco (plotly_white)
    - Eje Y muestra SOLO numero ODS
    """
    try:
        import plotly.express as px
        import plotly.io as pio

        pio.templates.default = "plotly_white"

        dfp = df_plot.copy()
        dfp["ODS_num"] = dfp["ODS"].apply(_ods_num_from_label)
        dfp = dfp.sort_values("probabilidad", ascending=True)

        fig = px.bar(
            dfp,
            x="probabilidad",
            y="ODS_num",
            orientation="h",
            text="probabilidad",
            height=420
        )

        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title="Probabilidad",
            yaxis_title="ODS",
            font=dict(color="#0F172A"),
        )

        fig.update_xaxes(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB")
        fig.update_yaxes(gridcolor="#FFFFFF")

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        # fallback sin plotly
        dfp = df_plot.copy()
        dfp["ODS"] = dfp["ODS"].apply(_ods_num_from_label)
        st.bar_chart(dfp.set_index("ODS"))