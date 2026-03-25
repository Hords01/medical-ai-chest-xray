"""
XAI-Enhanced Streamlit Interface Pattern
==========================================
Illustrative structure of the Tab-based XAI display
added to the Streamlit clinical interface in Phase 2.
"""

import streamlit as st
import numpy as np
from PIL import Image


def render_xai_tab():
    """
    Tab 1: XAI — Visual explanations of model decisions.

    Displays CAM heatmaps and LIME grid overlays side by side
    so physicians can see WHERE the model is looking.
    """
    st.markdown("### Açıklanabilirlik Haritaları (XAI)")

    st.info(
        "Bu sekmede, modelin kararını hangi görüntü bölgelerine dayanarak "
        "verdiğini gösteren açıklanabilirlik haritaları bulunmaktadır. "
        "Kırmızı/sıcak bölgeler yüksek katkı, mavi/soğuk bölgeler düşük katkı gösterir."
    )

    # CAM heatmaps — displayed as 2×3 grid
    cam_techniques = [
        "EigenCAM", "EigenGradCAM", "ScoreCAM",
        "AblationCAM", "KPCA-CAM", "Ensemble Average CAM"
    ]

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for idx, technique in enumerate(cam_techniques):
        with columns[idx % 3]:
            st.caption(technique)
            # In actual system:
            # heatmap = cam_results[technique]
            # overlay = show_cam_on_image(original_rgb, heatmap, use_rgb=True)
            # st.image(overlay, use_container_width=True)
            st.text(f"[{technique} heatmap overlay]")

    # LIME Grid
    st.markdown("---")
    st.markdown("### LIME Grid Overlay (3×4)")
    st.caption(
        "Görüntü 12 bölgeye ayrılmış, her bölgenin karara katkısı ölçülmüştür. "
        "Süper piksel bazlı yerel açıklama."
    )
    # In actual system:
    # lime_grid = cam_results["LIME_Grid"]
    # Display as colored grid overlay on image


def render_advanced_analysis_tab(patoloji_confidence, deger_confidence):
    """
    Tab 2: İleri Analiz — Quantitative confidence metrics.

    Displays confidence scores, uncertainty, and entropy for
    both models with clinician-friendly interpretations.
    """
    st.markdown("### Güven & Belirsizlik Analizi")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Değerlendirme Modeli")
        st.metric("Sınıf 0 Olasılığı",
                  f"%{deger_confidence['class_0_prob']*100:.2f}")
        st.metric("Sınıf 1 Olasılığı",
                  f"%{deger_confidence['class_1_prob']*100:.2f}")
        st.metric("Belirsizlik",
                  f"%{deger_confidence['uncertainty']*100:.2f}")
        st.metric("Entropi",
                  f"{deger_confidence['entropy']:.4f}")

        kategori, aciklama = get_confidence_interpretation(
            deger_confidence['confidence']
        )
        st.info(f"**{kategori}:** {aciklama}")

    with col2:
        st.markdown("#### Patoloji Modeli")
        st.metric("Sınıf 0 Olasılığı",
                  f"%{patoloji_confidence['class_0_prob']*100:.2f}")
        st.metric("Sınıf 1 Olasılığı",
                  f"%{patoloji_confidence['class_1_prob']*100:.2f}")
        st.metric("Belirsizlik",
                  f"%{patoloji_confidence['uncertainty']*100:.2f}")
        st.metric("Entropi",
                  f"{patoloji_confidence['entropy']:.4f}")

        kategori, aciklama = get_confidence_interpretation(
            patoloji_confidence['confidence']
        )
        st.info(f"**{kategori}:** {aciklama}")


def render_report_tab(eval_result, path_result, confidence_data):
    """
    Tab 3: Rapor — Summary report of all findings.

    Consolidates classification results, XAI highlights, and
    confidence metrics into a single exportable summary.
    """
    st.markdown("### Özet Rapor")

    st.markdown(f"""
    | Analiz | Sonuç |
    |--------|-------|
    | **Değerlendirme** | {eval_result} |
    | **Patoloji** | {path_result} |
    | **Güven (Değerlendirme)** | {confidence_data['eval_confidence']:.1%} |
    | **Güven (Patoloji)** | {confidence_data['path_confidence']:.1%} |
    """)

    st.caption("Bu rapor otomatik olarak oluşturulmuştur ve klinik karar "
               "verme sürecinde yardımcı bir araç olarak kullanılmalıdır.")


def get_confidence_interpretation(confidence):
    """Translate confidence to clinical categories."""
    if confidence >= 0.95:
        return "Çok Yüksek Güven", "Model bu tahminden çok emin"
    elif confidence >= 0.85:
        return "Yüksek Güven", "Model bu tahminden oldukça emin"
    elif confidence >= 0.70:
        return "Orta Güven", "Model bu tahminden orta derecede emin"
    elif confidence >= 0.55:
        return "Düşük Güven", "Model kararsız, ikinci görüş önerilir"
    else:
        return "Çok Düşük Güven", "Model karar veremiyor, uzman incelemesi gerekli"
