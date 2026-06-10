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
    initial_sidebar_state="expanded",
)


CLASS_TRANSLATIONS = {
    "battery": "bateria",
    "biological": "orgânico",
    "brown-glass": "vidro marrom",
    "cardboard": "papelão",
    "clothes": "tecido",
    "green-glass": "vidro verde",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plástico",
    "shoes": "calçado",
    "trash": "rejeito",
    "white-glass": "vidro transparente",
}

PRIORITY_TRANSLATIONS = {
    "high": "alta",
    "medium": "média",
    "low": "baixa",
}

MATERIAL_STREAM_TRANSLATIONS = {
    "special hazardous disposal": "destinação especial ou resíduo perigoso",
    "organic waste stream": "fluxo de resíduos orgânicos",
    "glass recycling stream": "fluxo de reciclagem de vidro",
    "paper recycling stream": "fluxo de reciclagem de papel e papelão",
    "recyclable paper stream": "fluxo de papel reciclável",
    "recyclable packaging stream": "fluxo de embalagens recicláveis",
    "textile reuse or recycling stream": "fluxo de reúso ou reciclagem têxtil",
    "textile recovery stream": "fluxo de recuperação têxtil",
    "metal recycling stream": "fluxo de reciclagem de metais",
    "plastic recycling stream": "fluxo de reciclagem de plásticos",
    "mixed material recovery stream": "fluxo de recuperação de materiais mistos",
    "non-recyclable waste stream": "fluxo de rejeitos não recicláveis",
    "unknown material stream": "fluxo de material não identificado",
}

ACTION_TRANSLATIONS = {
    "send to certified reverse logistics or hazardous waste handling": (
        "encaminhar para logística reversa certificada ou tratamento de resíduo perigoso"
    ),
    "send to composting, biodigestion or controlled organic waste treatment": (
        "encaminhar para compostagem, biodigestão ou tratamento controlado de resíduos orgânicos"
    ),
    "send to glass recycling stream": (
        "encaminhar para reciclagem de vidro"
    ),
    "send to glass recycling stream after contamination check": (
        "encaminhar para reciclagem de vidro após verificação de contaminação"
    ),
    "send to paper or cardboard recycling stream": (
        "encaminhar para reciclagem de papel ou papelão"
    ),
    "send to cardboard and paper recycling stream": (
        "encaminhar para reciclagem de papelão e papel"
    ),
    "send to paper recycling stream": (
        "encaminhar para reciclagem de papel"
    ),
    "send to textile reuse, donation or recycling stream": (
        "encaminhar para reúso, doação ou reciclagem têxtil"
    ),
    "send to textile reuse, recovery or recycling flow": (
        "encaminhar para reúso, recuperação ou reciclagem têxtil"
    ),
    "send to metal recycling stream": (
        "encaminhar para reciclagem de metais"
    ),
    "send to plastic recycling stream with resin identification check": (
        "encaminhar para reciclagem de plásticos com verificação do tipo de resina"
    ),
    "send to plastic recycling stream after resin and contamination check": (
        "encaminhar para reciclagem de plásticos após verificação de resina e contaminação"
    ),
    "send to reuse, material recovery or specialized recycling flow": (
        "encaminhar para reúso, recuperação de materiais ou reciclagem especializada"
    ),
    "send to controlled disposal and review source reduction opportunities": (
        "encaminhar para descarte controlado e revisar oportunidades de redução na origem"
    ),
    "send to manual inspection before disposal decision": (
        "encaminhar para inspeção manual antes da decisão de destinação"
    ),
}

COLUMN_TRANSLATIONS = {
    "source_label": "Classe original",
    "predicted_class": "Classe prevista",
    "confidence": "Confiança",
    "priority": "Prioridade",
    "material_stream": "Fluxo do material",
    "recommended_action": "Ação recomendada",
    "circularity_score": "Índice de circularidade",
    "estimated_landfill_avoidance_kg": "Potencial de redução de aterro (kg)",
    "special_handling": "Tratamento especial",
    "image_path": "Caminho técnico da imagem",
}


def inject_custom_css():
    st.markdown(
        """
        <style>
            :root {
                --bg: #f7f4ee;
                --surface: #ffffff;
                --surface-soft: #f1f5f9;
                --text: #1f2937;
                --text-soft: #4b5563;
                --primary: #00539f;
                --primary-dark: #003b73;
                --accent: #6aa84f;
                --border: #d9e2ec;
                --shadow: 0 12px 30px rgba(0, 43, 92, 0.08);
                --radius: 20px;
            }

            .stApp {
                background-color: var(--bg);
                color: var(--text);
            }

            .main .block-container {
                max-width: 1280px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                background: #f3f6fa;
                border-right: 1px solid var(--border);
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(0, 83, 159, 0.08), rgba(106, 168, 79, 0.10));
                border: 1px solid var(--border);
                border-radius: 26px;
                padding: 34px 36px;
                box-shadow: var(--shadow);
                margin-bottom: 28px;
            }

            .eyebrow {
                display: inline-block;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(0, 83, 159, 0.10);
                color: var(--primary);
                font-size: 0.88rem;
                font-weight: 700;
                margin-bottom: 16px;
            }

            .hero-title {
                margin: 0;
                color: var(--primary-dark);
                font-size: 3rem;
                font-weight: 800;
                line-height: 1.08;
            }

            .hero-subtitle {
                margin-top: 12px;
                margin-bottom: 18px;
                color: var(--text-soft);
                font-size: 1.08rem;
                line-height: 1.7;
                max-width: 980px;
            }

            .chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 18px;
            }

            .chip {
                background: #ffffff;
                border: 1px solid var(--border);
                color: var(--primary-dark);
                border-radius: 999px;
                padding: 8px 12px;
                font-size: 0.86rem;
                font-weight: 600;
            }

            .small-kicker {
                color: var(--primary);
                font-size: 0.84rem;
                font-weight: 800;
                letter-spacing: 0.03em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .section-title {
                color: var(--primary-dark);
                font-size: 2rem;
                font-weight: 800;
                margin-top: 0;
                margin-bottom: 12px;
            }

            .section-text {
                color: var(--text-soft);
                font-size: 1rem;
                line-height: 1.7;
                margin-bottom: 16px;
            }

            .metric-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 22px;
                min-height: 132px;
            }

            .metric-label {
                color: var(--text-soft);
                font-size: 0.94rem;
                margin-bottom: 12px;
            }

            .metric-value {
                color: var(--primary-dark);
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
            }

            .manual-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 22px;
                box-shadow: var(--shadow);
                padding: 24px;
                margin-top: 8px;
                margin-bottom: 18px;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid var(--border);
                border-radius: 16px;
                overflow: hidden;
                background: #ffffff;
            }

            div[data-baseweb="select"] > div {
                border-radius: 14px;
                border-color: var(--border);
                background: #ffffff;
            }

            button[kind="secondary"] {
                border-radius: 12px;
            }

            hr {
                border-color: var(--border);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def translate_class(value):
    return CLASS_TRANSLATIONS.get(str(value), str(value))


def translate_priority(value):
    return PRIORITY_TRANSLATIONS.get(str(value), str(value))


def translate_material_stream(value):
    return MATERIAL_STREAM_TRANSLATIONS.get(str(value), str(value))


def translate_action(value):
    return ACTION_TRANSLATIONS.get(str(value), str(value))


def translate_boolean(value):
    if value in [True, "True", "true", 1, "1"]:
        return "Sim"
    if value in [False, "False", "false", 0, "0"]:
        return "Não"
    return value


def translate_decision(decision):
    translated_decision = decision.copy()
    translated_decision["material_stream"] = translate_material_stream(
        translated_decision.get("material_stream", "")
    )
    translated_decision["recommended_action"] = translate_action(
        translated_decision.get("recommended_action", "")
    )
    translated_decision["priority"] = translate_priority(
        translated_decision.get("priority", "")
    )
    return translated_decision


def prepare_counts_for_display(series, translation_function):
    translated = series.astype(str).apply(translation_function)
    return translated.value_counts().sort_values(ascending=False)


def prepare_report_for_display(dataframe):
    display_report = dataframe.copy()

    for column in ["source_label", "predicted_class"]:
        if column in display_report.columns:
            display_report[column] = display_report[column].apply(translate_class)

    if "priority" in display_report.columns:
        display_report["priority"] = display_report["priority"].apply(translate_priority)

    if "material_stream" in display_report.columns:
        display_report["material_stream"] = display_report["material_stream"].apply(translate_material_stream)

    if "recommended_action" in display_report.columns:
        display_report["recommended_action"] = display_report["recommended_action"].apply(translate_action)

    if "special_handling" in display_report.columns:
        display_report["special_handling"] = display_report["special_handling"].apply(translate_boolean)

    if "confidence" in display_report.columns:
        display_report["confidence"] = pd.to_numeric(
            display_report["confidence"],
            errors="coerce",
        ).round(3)

    if "estimated_landfill_avoidance_kg" in display_report.columns:
        display_report["estimated_landfill_avoidance_kg"] = pd.to_numeric(
            display_report["estimated_landfill_avoidance_kg"],
            errors="coerce",
        ).round(2)

    business_columns = [
        "source_label",
        "predicted_class",
        "confidence",
        "priority",
        "material_stream",
        "recommended_action",
        "circularity_score",
        "estimated_landfill_avoidance_kg",
        "special_handling",
    ]

    technical_columns = [
        "image_path",
    ]

    visible_columns = [
        column for column in business_columns
        if column in display_report.columns
    ]

    complete_columns = visible_columns + [
        column for column in technical_columns
        if column in display_report.columns
    ]

    business_report = display_report[visible_columns].rename(columns=COLUMN_TRANSLATIONS)
    technical_report = display_report[complete_columns].rename(columns=COLUMN_TRANSLATIONS)

    return business_report, technical_report


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def crop_center_region(image, crop_ratio=0.55):
    width, height = image.size

    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)

    left = int((width - crop_width) / 2)
    top = int((height - crop_height) / 2)
    right = left + crop_width
    bottom = top + crop_height

    return image.crop((left, top, right, bottom))


def run_single_image_inspection(input_image, caption, model_path, apply_center_crop=False):
    col1, col2 = st.columns([1, 1])

    processed_image = crop_center_region(input_image) if apply_center_crop else input_image

    with col1:
        st.image(
            input_image,
            caption=caption,
            use_container_width=True,
        )

        if apply_center_crop:
            st.image(
                processed_image,
                caption="Região analisada pelo modelo",
                use_container_width=True,
            )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        processed_image.save(tmp.name)
        temp_path = tmp.name

    result = predict_image(temp_path, model_path)
    decision = translate_decision(result["circular_decision"])

    with col2:
        st.subheader("Resultado da inspeção")
        st.metric("Classe prevista", translate_class(result["predicted_class"]))

        if result["confidence"] is not None:
            st.metric("Confiança", f'{result["confidence"]:.2%}')

        st.write(f'**Fluxo do material:** {decision["material_stream"]}')
        st.write(f'**Ação recomendada:** {decision["recommended_action"]}')
        st.write(f'**Prioridade:** {decision["priority"]}')
        st.metric("Índice de circularidade", decision["circularity_score"])
        st.metric(
            "Potencial de redução de aterro por item",
            f'{decision["estimated_landfill_avoidance_kg"]:.2f} kg',
        )

    if result["probabilities"]:
        probabilities = pd.DataFrame(
            [
                {
                    "Classe": translate_class(key),
                    "Probabilidade": value,
                }
                for key, value in result["probabilities"].items()
            ]
        ).sort_values("Probabilidade", ascending=False)

        st.subheader("Probabilidades por classe")
        st.bar_chart(probabilities.set_index("Classe"), use_container_width=True)


inject_custom_css()

model_path = MODELS_DIR / "circular_vision_model.joblib"
metrics_path = OUTPUTS_DIR / "metrics.json"
report_path = OUTPUTS_DIR / "sustainability_report.csv"

if not model_path.exists():
    st.warning("Modelo não encontrado. Treine-o primeiro com: python train.py --dataset-dir <raiz_do_dataset>")
    st.stop()

payload = joblib.load(model_path)
classes = [str(item) for item in payload.get("classes", [])]
translated_classes = [translate_class(item) for item in classes]

st.sidebar.markdown("## Modelo")
st.sidebar.write("Classes disponíveis")
st.sidebar.write(", ".join(translated_classes))

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = metrics.get("best_model", "não disponível")

    st.sidebar.metric("Melhor modelo", best)

    if "results" in metrics and best in metrics["results"]:
        st.sidebar.metric("F1 Macro", f'{metrics["results"][best]["macro_f1"]:.3f}')
        st.sidebar.metric("Acurácia", f'{metrics["results"][best]["accuracy"]:.3f}')

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">PoC de IA aplicada à sustentabilidade industrial</div>
        <h1 class="hero-title">Circular Vision</h1>
        <p class="hero-subtitle">
            Plataforma de inspeção visual inteligente para apoiar decisões de economia circular em operações
            de manufatura. A solução transforma imagens de materiais e resíduos em recomendações operacionais,
            indicadores executivos e apoio à destinação sustentável.
        </p>
        <div class="chip-row">
            <div class="chip">Python</div>
            <div class="chip">Visão Computacional</div>
            <div class="chip">Machine Learning</div>
            <div class="chip">Data Analytics</div>
            <div class="chip">Indústria 4.0</div>
            <div class="chip">Economia Circular</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="small-kicker">Visão executiva</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Indicadores para sustentabilidade operacional</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-text">
        O dashboard consolida a análise visual em indicadores de circularidade, potencial de redução de envio a aterro
        e alertas críticos de destinação, com foco em apoio à manufatura, sustentabilidade e melhoria contínua.
    </div>
    """,
    unsafe_allow_html=True,
)

if report_path.exists():
    report = pd.read_csv(report_path)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card("Itens analisados", len(report))

    with c2:
        render_metric_card(
            "Índice médio de circularidade",
            f'{report["circularity_score"].mean():.1f}',
        )

    with c3:
        render_metric_card(
            "Potencial de redução de aterro",
            f'{report["estimated_landfill_avoidance_kg"].sum():.2f} kg',
        )

    with c4:
        render_metric_card(
            "Alertas críticos",
            int(report["special_handling"].sum()),
        )

    st.divider()

    st.markdown('<div class="small-kicker">Insights operacionais</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Distribuições e leitura analítica do lote</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Classes previstas")
        if "predicted_class" in report.columns:
            class_counts = prepare_counts_for_display(
                report["predicted_class"],
                translate_class,
            )
            st.bar_chart(class_counts, use_container_width=True)

    with chart_col2:
        st.subheader("Níveis de prioridade")
        if "priority" in report.columns:
            priority_counts = prepare_counts_for_display(
                report["priority"],
                translate_priority,
            )
            st.bar_chart(priority_counts, use_container_width=True)

    st.divider()

    st.markdown('<div class="small-kicker">Relatório operacional</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Relatório de economia circular</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="section-text">
            Filtre o lote processado por prioridade e classe prevista. A tabela prioriza a leitura executiva
            e mantém informações técnicas disponíveis em uma área separada.
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_priorities = (
        sorted(report["priority"].dropna().unique())
        if "priority" in report.columns
        else []
    )

    available_classes = (
        sorted(report["predicted_class"].dropna().unique())
        if "predicted_class" in report.columns
        else []
    )

    priority_options = {
        translate_priority(priority): priority
        for priority in available_priorities
    }

    class_options = {
        translate_class(class_name): class_name
        for class_name in available_classes
    }

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_priorities_display = st.multiselect(
            "Filtrar por prioridade",
            options=list(priority_options.keys()),
            default=list(priority_options.keys()),
        )

    with filter_col2:
        selected_classes_display = st.multiselect(
            "Filtrar por classe prevista",
            options=list(class_options.keys()),
            default=list(class_options.keys()),
        )

    selected_priorities = [
        priority_options[item]
        for item in selected_priorities_display
    ]

    selected_classes = [
        class_options[item]
        for item in selected_classes_display
    ]

    filtered_report = report.copy()

    if selected_priorities:
        filtered_report = filtered_report[
            filtered_report["priority"].isin(selected_priorities)
        ]

    if selected_classes:
        filtered_report = filtered_report[
            filtered_report["predicted_class"].isin(selected_classes)
        ]

    business_report, technical_report = prepare_report_for_display(filtered_report)

    st.dataframe(
        business_report,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = technical_report.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Baixar relatório filtrado em CSV",
        data=csv_data,
        file_name="circular_vision_relatorio_filtrado.csv",
        mime="text/csv",
    )

    with st.expander("Exibir informações técnicas"):
        st.dataframe(
            technical_report,
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        """
        Nenhum relatório operacional foi encontrado.

        Gere-o com:

        python batch_inference.py --dataset-dir "<raiz_do_dataset>" --max-per-class 10
        """
    )

st.divider()

st.markdown('<div class="small-kicker">Inspeção individual</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Inspeção manual e captura por câmera</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="manual-card">
        <div class="section-text">
            Use esta seção para inspecionar uma imagem de material ou resíduo por envio manual
            ou captura pela câmera. Em um cenário industrial real, essa entrada pode ser conectada
            a uma câmera de inspeção, estação de triagem, dispositivo de borda ou ponto de controle
            próximo a uma linha de produção.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

camera_tab, upload_tab = st.tabs(
    [
        "Captura por câmera",
        "Envio manual",
    ]
)

with camera_tab:
    st.subheader("Captura por câmera")
    st.write(
        "Posicione o material no centro da câmera, com fundo simples e boa iluminação. "
        "A PoC analisa automaticamente a região central da imagem capturada."
    )

    camera_file = st.camera_input("Abrir câmera")

    if camera_file is not None:
        camera_image = Image.open(camera_file).convert("RGB")
        run_single_image_inspection(
            camera_image,
            "Imagem capturada pela câmera",
            model_path,
            apply_center_crop=True,
        )

with upload_tab:
    st.subheader("Envio manual")
    st.write(
        "Envie manualmente uma imagem de material ou fluxo de resíduo."
    )

    uploaded = st.file_uploader(
        "Enviar imagem",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is not None:
        uploaded_image = Image.open(uploaded).convert("RGB")
        run_single_image_inspection(
            uploaded_image,
            "Imagem enviada manualmente",
            model_path,
            apply_center_crop=False,
        )