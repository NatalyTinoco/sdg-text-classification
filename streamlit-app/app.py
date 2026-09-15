import streamlit as st
import numpy as np
import pandas as pd

# IMPORTANTE para joblib: si el pipeline usa preprocessor=limpiar_texto_stem
from text_preprocess import limpiar_texto_stem  # noqa: F401

from helpers import inject_light_theme_css, load_model, ods_label
from viz import predict_one, make_topk_df, probs_df, margin_confidence, plot_probs

st.set_page_config(page_title="Clasificador ODS", layout="wide")
inject_light_theme_css()

# Header
st.title("Clasificador de texto a ODS")
st.write("Pega un texto o sube un CSV/Excel y te digo el ODS mas probable + topK con probabilidades.")

# Cargar modelo
try:
    model = load_model("pipeline_calibrado.pkl")
except Exception as e:
    st.error(
        "No pude cargar `pipeline_calibrado.pkl`.\n\n"
        "Revisa que este en la carpeta `dash/`.\n"
        "Si el pipeline usa `preprocessor=limpiar_texto_stem`, esta funcion debe existir (por eso importamos text_preprocess).\n\n"
        f"Error:\n{e}"
    )
    st.stop()

classes = model.classes_

tab_texto, tab_csv = st.tabs(["Texto", "CSV / Excel"])

# =========================
# TAB TEXTO (mockup)
# =========================
with tab_texto:
    c_inst, c_text, c_obj = st.columns([1.0, 2.2, 1.0], gap="large")

    with c_inst:
        with st.container(border=True):
            st.markdown("### Instrucciones de uso")
            st.markdown(
                "1) Texto individual: pega un comentario y clasifica.\n\n"
                "2) CSV/Excel: sube un archivo con comentarios."
            )

    with c_text:
        with st.container(border=True):
            st.markdown("### Aqui la persona puede escribir texto libre")
            texto = st.text_area(
                "Texto",
                height=190,
                label_visibility="collapsed",
                placeholder="Escribe o pega aqui..."
            )

            a1, a2, a3 = st.columns([1, 1, 1.2])
            with a1:
                top_k = st.slider("Top K", 3, 10, 5)
            with a2:
                mostrar_todas = st.checkbox("Mostrar todas", value=False)
            with a3:
                btn = st.button("Clasificar", type="primary", use_container_width=True)

    with c_obj:
        with st.container(border=True):
            st.markdown("### Objetivo")
            obj_ph = st.empty()
            prob_ph = st.empty()
            margin_ph = st.empty()

    st.write("")

    with st.container(border=True):
        st.markdown("### Distribucion de probabilidades")
        chart_ph = st.container()
        topk_ph = st.container()
        info_ph = st.empty()

    if btn and texto.strip():
        pred, probs, top_idx = predict_one(model, classes, texto)
        margin = margin_confidence(probs)

        obj_ph.markdown(
            f"""
            <div style="background:#F1F5FF;border:1px solid #DCE4FF;border-radius:12px;padding:12px;">
              <div style="font-weight:800;color:#0F172A;font-size:16px;">{ods_label(pred)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        prob_ph.metric("Probabilidad top1", f"{float(np.max(probs)):.4f}")
        margin_ph.caption(f"Confianza (top1-top2): {margin:.4f}")

        df_top = make_topk_df(classes, probs, top_idx, k=top_k)
        df_plot = probs_df(classes, probs) if mostrar_todas else df_top.copy()

        with chart_ph:
            plot_probs(st, df_plot)

        with topk_ph:
            st.markdown(f"### Top {top_k} ODS")
            st.dataframe(df_top, use_container_width=True)

        if margin < 0.10:
            info_ph.info("Texto ambiguo: top1 y top2 estan muy cerca.")
        else:
            info_ph.empty()
    else:
        with chart_ph:
            st.caption("Pega un texto y dale 'Clasificar'.")


# =========================
# TAB CSV / EXCEL
# =========================
with tab_csv:
    with st.container(border=True):
        st.markdown("### Sube tu archivo")
        st.caption("Acepta CSV (.csv) o Excel (.xlsx/.xls).")

        up = st.file_uploader("Archivo", type=["csv", "xlsx", "xls"])

    if up is None:
        st.caption("Sube un archivo, elige la columna con comentarios, y corre el clasificador.")
    else:
        name = up.name.lower()

        if name.endswith(".csv"):
            df = pd.read_csv(up)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(up)
        else:
            st.error("Formato no soportado. Sube .csv, .xlsx o .xls")
            st.stop()

        st.write("Preview:")
        st.dataframe(df.head(20), use_container_width=True)

        col_text = st.selectbox("Elige la columna a analizar", list(df.columns), index=0)
        k_csv = st.slider("Top K para guardar en salida", 3, 10, 5, key="k_csv")

        run = st.button("Clasificar archivo", type="secondary", use_container_width=True)

        if run:
            textos = df[col_text].fillna("").astype(str).tolist()

            probs_all = model.predict_proba(textos)
            pred_idx = np.argmax(probs_all, axis=1)
            pred_class = classes[pred_idx]
            prob_top1 = probs_all[np.arange(len(textos)), pred_idx]

            topk_idx = np.argsort(probs_all, axis=1)[:, ::-1][:, :k_csv]
            topk_json = []
            for i in range(len(textos)):
                items = []
                for j in topk_idx[i]:
                    items.append({
                        "ods": str(classes[j]),
                        "label": ods_label(classes[j]),
                        "p": float(probs_all[i, j])
                    })
                topk_json.append(items)

            df_out = df.copy()
            df_out["ods_pred"] = [ods_label(x) for x in pred_class]
            df_out["prob_top1"] = prob_top1
            df_out["topk_json"] = topk_json

            st.success("Listo. Clasificacion terminada.")

            st.subheader("Cantidad de comentarios por ODS")
            counts = df_out["ods_pred"].value_counts().reset_index()
            counts.columns = ["ODS", "conteo"]
            st.dataframe(counts, use_container_width=True)
            st.bar_chart(counts.set_index("ODS"))

            st.subheader("Descarga")
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CSV con predicciones",
                data=csv_bytes,
                file_name="archivo_con_ods.csv",
                mime="text/csv",
                use_container_width=True
            )