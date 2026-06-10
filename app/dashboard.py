from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import tempfile

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from app.config import MODELS_DIR, OUTPUTS_DIR
from predict import predict_image


st.set_page_config(
    page_title="Circular Vision",
    layout="wide",
)

st.title("Circular Vision")
st.caption("Triagem visual com IA para manufatura sustentável")

st.markdown(
    """
    O Circular Vision é uma prova de conceito que aplica visão computacional e aprendizado de máquina
    para apoiar decisões de economia circular em ambientes de manufatura. O sistema classifica fluxos
    de materiais e resíduos, recomenda ações sustentáveis e gera indicadores operacionais relacionados
    à reciclagem, redução de envio para aterro e necessidade de tratamento especial.
    """
)

model_path = MODELS_DIR / "circular_vision_model.joblib"
metrics_path = OUTPUTS_DIR / "metrics.json"
report_path = OUTPUTS_DIR / "sustainability_report.csv"

if not model_path.exists():
    st.warning("Modelo não encontrado. Treine-o primeiro com: python train.py --dataset-dir <raiz_do_dataset>")
    st.stop()

payload = joblib.load(model_path)
classes = [str(item) for item in payload.get("classes", [])]

st.sidebar.header("Modelo")
st.sidebar.write("Classes disponíveis")
st.sidebar.write(", ".join(classes))

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = metrics.get("best_model", "não disponível")

    st.sidebar.metric("Melhor modelo", best)

    if "results" in metrics and best in metrics["results"]:
        st.sidebar.metric("F1 Macro", f'{metrics["results"][best]["macro_f1"]:.3f}')
        st.sidebar.metric("Acurácia", f'{metrics["results"][best]["accuracy"]:.3f}')

st.header("Visão executiva")

if report_path.exists():
    report = pd.read_csv(report_path)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Itens inspecionados", len(report))
    c2.metric("Pontuação média de circularidade", f'{report["circularity_score"].mean():.1f}')
    c3.metric(
        "Desvio estimado de aterro",
        f'{report["estimated_landfill_avoidance_kg"].sum():.2f} kg',
    )
    c4.metric("Alertas de tratamento especial", int(report["special_handling"].sum()))

    st.divider()

    st.header("Insights operacionais")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição das classes previstas")
        if "predicted_class" in report.columns:
            class_counts = report["predicted_class"].value_counts().sort_values(ascending=False)
            st.bar_chart(class_counts)

    with col2:
        st.subheader("Distribuição de prioridade")
        if "priority" in report.columns:
            priority_counts = report["priority"].value_counts().sort_values(ascending=False)
            st.bar_chart(priority_counts)

    st.divider()

    st.header("Relatório de economia circular")

    available_priorities = sorted(report["priority"].dropna().unique()) if "priority" in report.columns else []
    available_classes = sorted(report["predicted_class"].dropna().unique()) if "predicted_class" in report.columns else []

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_priorities = st.multiselect(
            "Filtrar por prioridade",
            options=available_priorities,
            default=available_priorities,
        )

    with filter_col2:
        selected_classes = st.multiselect(
            "Filtrar por classe prevista",
            options=available_classes,
            default=available_classes,
        )

    filtered_report = report.copy()

    if selected_priorities:
        filtered_report = filtered_report[filtered_report["priority"].isin(selected_priorities)]

    if selected_classes:
        filtered_report = filtered_report[filtered_report["predicted_class"].isin(selected_classes)]

    st.dataframe(filtered_report, use_container_width=True)

    csv_data = filtered_report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Baixar relatório filtrado em CSV",
        data=csv_data,
        file_name="circular_vision_relatorio_filtrado.csv",
        mime="text/csv",
    )

    st.divider()

    st.header("Galeria de amostras inspecionadas")

    gallery_size = min(6, len(filtered_report))

    if gallery_size > 0:
        gallery = filtered_report.head(gallery_size)
        gallery_columns = st.columns(3)

        for index, (_, row) in enumerate(gallery.iterrows()):
            image_path = Path(str(row["image_path"]))

            with gallery_columns[index % 3]:
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)

                st.write(f'**Classe prevista:** {row["predicted_class"]}')
                st.write(f'**Classe de origem:** {row.get("source_label", "não disponível")}')
                st.write(f'**Prioridade:** {row["priority"]}')
                st.write(f'**Ação recomendada:** {row["recommended_action"]}')
                st.write(f'**Pontuação de circularidade:** {row["circularity_score"]}')

else:
    st.info(
        """
        Nenhum relatório operacional foi encontrado. Gere-o com:
        
        python batch_inference.py --dataset-dir "<raiz_do_dataset>" --max-per-class 10
        """
    )

st.divider()

st.header("Inspeção manual")

st.markdown(
    """
    Use esta seção para inspecionar manualmente uma única imagem. Em um cenário real de manufatura,
    esta entrada poderia ser substituída por uma câmera industrial, dispositivo de borda ou estação de inspeção.
    """
)

uploaded = st.file_uploader(
    "Envie uma imagem de material ou fluxo de resíduo",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Imagem de entrada", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    result = predict_image(temp_path, model_path)
    decision = result["circular_decision"]

    with col2:
        st.subheader("Predição")
        st.metric("Classe prevista", result["predicted_class"])

        if result["confidence"] is not None:
            st.metric("Confiança", f'{result["confidence"]:.2%}')

        st.subheader("Decisão de economia circular")
        st.write(f'**Fluxo de material:** {decision["material_stream"]}')
        st.write(f'**Ação recomendada:** {decision["recommended_action"]}')
        st.write(f'**Prioridade:** {decision["priority"]}')
        st.metric("Pontuação de circularidade", decision["circularity_score"])
        st.metric(
            "Desvio estimado de aterro por item",
            f'{decision["estimated_landfill_avoidance_kg"]:.2f} kg',
        )

    if result["probabilities"]:
        probabilities = pd.DataFrame(
            [
                {
                    "classe": key,
                    "probabilidade": value,
                }
                for key, value in result["probabilities"].items()
            ]
        ).sort_values("probabilidade", ascending=False)

        st.subheader("Probabilidades por classe")
        st.bar_chart(probabilities.set_index("classe"))