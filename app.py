"""
Handwritten Digit Recognition System Using Convolutional Neural Networks (CNN)
Interactive Neural Network Visualization & AI Lab Platform.
"Draw a Digit. Watch the Neural Network Think."
"""

import os
import json
import time
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

from src.device import configure_device, get_device_summary
from src.preprocessing import (
    preprocess_drawn_image,
    get_sample_digits,
    get_mnist_subset,
    load_mnist_raw
)
from src.activations import (
    extract_intermediate_activations,
    get_layer_metadata,
    get_top_active_filters,
    build_activation_extractor
)
from src.model import load_trained_model

# Page configuration
st.set_page_config(
    page_title="Neural Lab | Handwritten Digit Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Research-Lab Dark Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main app typography & dark theme base */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1322 0%, #080b12 90%);
        color: #E2E8F0;
    }
    
    /* Headings */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 20%, #38BDF8 70%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* Metric & Summary Cards */
    .metric-glass-card {
        background: rgba(19, 27, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        text-align: left;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .metric-tag {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    
    .metric-num {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-subtext {
        font-size: 0.8rem;
        color: #38BDF8;
        font-weight: 500;
        margin-top: 2px;
    }
    
    /* Status Badges */
    .pill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .pill-purple {
        background: rgba(139, 92, 246, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Prediction Banner */
    .prediction-display-box {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(6, 182, 212, 0.12);
        margin-bottom: 20px;
    }
    
    .pred-digit-giant {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .pred-confidence-lead {
        font-size: 1.35rem;
        font-weight: 700;
        color: #34D399;
    }
    
    /* Section dividers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 1.5rem;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    
    /* Stage progress badge */
    .stage-indicator {
        background: #1E293B;
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: 600;
        color: #38BDF8;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    /* Educational callouts */
    .edu-callout {
        background: rgba(30, 58, 138, 0.25);
        border-left: 4px solid #38BDF8;
        padding: 12px 18px;
        border-radius: 6px;
        margin: 14px 0;
        color: #E2E8F0;
        font-size: 0.93rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/mnist_cnn.keras"
METRICS_PATH = "results/evaluation_metrics.json"
HISTORY_PATH = "results/training_history.json"
MISCLASSIFICATIONS_PATH = "results/misclassifications.json"


# ---------------- CACHED RESOURCES ----------------
@st.cache_resource
def get_hardware_info():
    """Detects available hardware devices (GPU/CPU) with memory growth."""
    return configure_device(verbose=False)


@st.cache_resource
def get_model():
    """Loads and caches the trained Keras CNN model."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load_trained_model(MODEL_PATH)
    except Exception:
        return None


@st.cache_resource
def get_activation_extractor(_model):
    """Builds and caches the multi-output intermediate activation extractor."""
    if _model is None:
        return None
    try:
        return build_activation_extractor(_model)
    except Exception:
        return None


@st.cache_data
def load_evaluation_data():
    """Loads saved metrics and training history."""
    metrics = None
    history = None
    misclassifications = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
        except Exception:
            pass

    if os.path.exists(MISCLASSIFICATIONS_PATH):
        try:
            with open(MISCLASSIFICATIONS_PATH, "r") as f:
                misclassifications = json.load(f)
        except Exception:
            pass

    return metrics, history, misclassifications


@st.cache_data
def get_cached_sample_digits():
    """Caches 5 real MNIST samples per digit for fast preset loading."""
    try:
        return get_sample_digits(num_per_digit=5)
    except Exception:
        return {}


# ---------------- SESSION STATE INITIALIZATION ----------------
if "active_tensor" not in st.session_state:
    st.session_state["active_tensor"] = None
if "raw_display_image" not in st.session_state:
    st.session_state["raw_display_image"] = None
if "activations_data" not in st.session_state:
    st.session_state["activations_data"] = None
if "live_stage" not in st.session_state:
    st.session_state["live_stage"] = 0
if "learning_mode" not in st.session_state:
    st.session_state["learning_mode"] = True
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "🏠 Home Dashboard & AI Lab"


# ---------------- LOAD SYSTEM ASSETS ----------------
hardware_info = get_hardware_info()
model = get_model()
extractor = get_activation_extractor(model)
metrics_data, history_data, misclass_data = load_evaluation_data()
sample_digits_map = get_cached_sample_digits()


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 8px;">
        <div style="font-size: 1.35rem; font-weight: 800; color: #38BDF8; display: flex; align-items: center; gap: 8px;">
            🔬 NEURAL LAB
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">
            Interactive CNN Digit Recognition Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status Badges
    badge_col1, badge_col2 = st.columns(2)
    with badge_col1:
        if model is not None:
            st.markdown('<span class="pill-badge pill-green">● Model Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-badge pill-amber">● Model Missing</span>', unsafe_allow_html=True)
    with badge_col2:
        if hardware_info["gpu_available"]:
            st.markdown('<span class="pill-badge pill-cyan">⚡ GPU Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-badge pill-amber">💻 CPU Mode</span>', unsafe_allow_html=True)

    st.markdown("---")

    nav_options = [
        "🏠 Home Dashboard & AI Lab",
        "⚡ Live Inference (\"Watch Network Think\")",
        "🧠 Neural Network Explorer",
        "🏋️ Model Training Dashboard",
        "📊 Model Evaluation Dashboard",
        "🔍 MNIST Dataset Explorer",
        "📖 About & Learning Mode"
    ]

    selected_page = st.radio(
        "Navigation Menu",
        nav_options,
        index=nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0,
        key="nav_radio"
    )
    st.session_state["nav_page"] = selected_page

    st.markdown("---")
    st.markdown("### 🎓 Interactive Controls")
    st.session_state["learning_mode"] = st.toggle("Enable Learning Mode", value=st.session_state["learning_mode"], help="Shows educational annotations and quizzes.")

    st.markdown("---")
    st.markdown("### 📊 Quick Performance")
    if metrics_data:
        st.metric("Test Accuracy", f"{metrics_data.get('test_accuracy', 0.0) * 100:.2f}%", help="Evaluated on 10,000 untouched test samples")
        st.metric("Test Loss", f"{metrics_data.get('test_loss', 0.0):.4f}")
    else:
        st.caption("Metrics not found. Run `python evaluate.py`.")

    st.caption("B.Tech AIML Mini Project • TensorFlow 2.x • MNIST")


# Helper to run inference on active tensor
def process_and_run_inference(raw_image_input):
    """Preprocesses input image, extracts all activations, and saves to session state."""
    tensor, processed_28 = preprocess_drawn_image(raw_image_input)
    st.session_state["active_tensor"] = tensor
    st.session_state["raw_display_image"] = processed_28

    if model is not None:
        act_res = extract_intermediate_activations(model, tensor, extractor=extractor)
        st.session_state["activations_data"] = act_res
        return act_res
    return None


# Helper to create Plotly horizontal probabilities bar chart
def plot_probabilities_chart(probabilities, winning_digit):
    digits = [str(i) for i in range(10)]
    percentages = [p * 100 for p in probabilities]
    colors = ['#06B6D4' if i == winning_digit else 'rgba(148, 163, 184, 0.3)' for i in range(10)]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=[f"Digit {d}" for d in digits],
        orientation='h',
        marker=dict(color=colors, line=dict(width=1, color='#0891B2')),
        text=[f"{p:.1f}%" for p in percentages],
        textposition='outside',
        textfont=dict(family="JetBrains Mono", size=12, color="#F8FAFC")
    ))

    fig.update_layout(
        title=dict(text="Class Probability Distribution (Softmax Output)", font=dict(family="Inter", size=13, color="#94A3B8")),
        xaxis=dict(title="Confidence (%)", range=[0, 115], showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#F8FAFC")),
        margin=dict(l=60, r=40, t=35, b=35),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# ==============================================================================
# SECTION 1: HOME DASHBOARD & AI LAB
# ==============================================================================
if st.session_state["nav_page"] == "🏠 Home Dashboard & AI Lab":
    st.markdown('<div class="hero-title">Handwritten Digit Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Explore how convolutional neural networks transform raw pixels into intelligent predictions.</div>', unsafe_allow_html=True)

    # 6 Top Summary Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-tag">Model Architecture</div>
            <div class="metric-num">CNN</div>
            <div class="metric-subtext">2 Conv + 2 MaxPool</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-tag">Dataset Benchmark</div>
            <div class="metric-num">70,000</div>
            <div class="metric-subtext">MNIST 28×28 Grayscale</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-tag">Classes Target</div>
            <div class="metric-num">10</div>
            <div class="metric-subtext">Numerals (0 to 9)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        acc_val = f"{metrics_data.get('test_accuracy', 0.0) * 100:.2f}%" if metrics_data else "Untrained"
        st.markdown(f"""
        <div class="metric-glass-card">
            <div class="metric-tag">Test Accuracy</div>
            <div class="metric-num">{acc_val}</div>
            <div class="metric-subtext">Held-out 10k Test Set</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        dev_title = "GPU Active" if hardware_info["gpu_available"] else "CPU Fallback"
        dev_sub = hardware_info["gpu_devices"][0].name if hardware_info["gpu_available"] else "WSL2 for RTX 3050"
        st.markdown(f"""
        <div class="metric-glass-card">
            <div class="metric-tag">Hardware Target</div>
            <div class="metric-num">{hardware_info['device_name']}</div>
            <div class="metric-subtext">{dev_sub[:18]}</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown("""
        <div class="metric-glass-card">
            <div class="metric-tag">Keras Artifact</div>
            <div class="metric-num">Ready</div>
            <div class="metric-subtext">models/mnist_cnn.keras</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Pipeline Flowchart
    st.markdown('<div class="section-title">⚡ Interactive Inference Flowchart</div>', unsafe_allow_html=True)
    st.caption("Click any stage to understand the mathematical transformation:")

    flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)
    with flow_col1:
        with st.expander("1. Input Tensor"):
            st.write("28×28 normalized grayscale tensor. Centered using center-of-mass with aspect ratio preserved.")
    with flow_col2:
        with st.expander("2. Conv2D Layers"):
            st.write("Learnable 3×3 kernels slide over pixels to detect edges, curves, and contours using ReLU.")
    with flow_col3:
        with st.expander("3. Max Pooling"):
            st.write("2×2 downsampling retains maximum local activations, reducing spatial footprint while providing translation invariance.")
    with flow_col4:
        with st.expander("4. Dense Layer"):
            st.write("Flattens spatial maps into 3,136 features and maps them to 128 high-level semantic hidden units.")
    with flow_col5:
        with st.expander("5. Softmax Output"):
            st.write("Calculates relative probability distribution over 10 mutually exclusive classes (0 through 9).")

    # Main Drawing & Prediction Workspace
    st.markdown('<div class="section-title">🎨 AI Lab — Interactive Drawing Workspace</div>', unsafe_allow_html=True)
    st.caption("Draw a digit (0–9) or click one of the quick test presets below:")

    # Quick Preset Buttons
    st.markdown("**Quick Example Digit Presets (from real MNIST test set):**")
    preset_cols = st.columns(10)
    preset_triggered = None
    for d in range(10):
        with preset_cols[d]:
            if st.button(f"Digit {d}", key=f"preset_btn_{d}", use_container_width=True):
                if sample_digits_map and d in sample_digits_map and len(sample_digits_map[d]) > 0:
                    preset_triggered = sample_digits_map[d][0]

    col_canvas, col_results = st.columns([1.1, 1.2], gap="large")

    with col_canvas:
        st.markdown("#### 🖌️ Virtual Drawing Canvas")
        brush_width = st.slider("Stroke Thickness", min_value=12, max_value=32, value=22, step=2)

        # Drawing Canvas
        canvas_data = st_canvas(
            fill_color="#000000",
            stroke_width=brush_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="main_digit_canvas",
            display_toolbar=True
        )

        st.caption("💡 **Tip:** Draw boldly in the center. Use the trash icon on the canvas toolbar to clear.")

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            btn_predict = st.button("⚡ Instant Predict", type="primary", use_container_width=True)
        with action_col2:
            btn_explain = st.button("🧠 Explain Prediction", use_container_width=True)

        with st.expander("📁 Upload an image file instead"):
            uploaded_file = st.file_uploader("Upload PNG/JPG of handwritten digit", type=["png", "jpg", "jpeg"])

    with col_results:
        st.markdown("#### 🎯 Prediction & Neural Activity")

        active_img = None
        if preset_triggered is not None:
            active_img = preset_triggered
        elif uploaded_file is not None:
            active_img = Image.open(uploaded_file)
        elif btn_predict or btn_explain:
            if canvas_data is not None and canvas_data.image_data is not None:
                active_img = canvas_data.image_data

        if active_img is not None:
            try:
                with st.spinner("Executing neural network forward pass..."):
                    res = process_and_run_inference(active_img)

                pred_digit = res["predicted_digit"]
                conf = res["confidence"]
                time_ms = res["inference_time_ms"]
                top_3 = res["top_3"]

                # Prediction Banner
                st.markdown(f"""
                <div class="prediction-display-box">
                    <div class="metric-tag" style="color: #38BDF8;">Recognized Handwritten Digit</div>
                    <div class="pred-digit-giant">{pred_digit}</div>
                    <div class="pred-confidence-lead">Confidence: {conf * 100:.2f}%</div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
                        ⚡ Latency: {time_ms:.2f} ms | Device: {hardware_info['device_string']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top-3 Predictions Row
                top_c1, top_c2, top_c3 = st.columns(3)
                with top_c1:
                    st.markdown(f"**#1 Prediction:** Digit `{top_3[0][0]}` ({top_3[0][1]*100:.1f}%)")
                with top_c2:
                    st.markdown(f"**#2 Candidate:** Digit `{top_3[1][0]}` ({top_3[1][1]*100:.1f}%)")
                with top_c3:
                    st.markdown(f"**#3 Candidate:** Digit `{top_3[2][0]}` ({top_3[2][1]*100:.1f}%)")

                # Probability Distribution Chart
                fig_probs = plot_probabilities_chart(res["probabilities"], pred_digit)
                st.plotly_chart(fig_probs, use_container_width=True)

                # Preprocessing Transparency Inspection
                st.markdown("##### 🔬 Preprocessing Verification (Input Tensor)")
                debug_col1, debug_col2 = st.columns([1, 2])
                with debug_col1:
                    st.image(st.session_state["raw_display_image"], width=110, caption="28×28 Centered Input", clamp=True)
                with debug_col2:
                    st.markdown(f"""
                    - **Tensor Shape**: `(1, 28, 28, 1)`
                    - **Centering**: Center-of-mass alignment
                    - **Pixel Range**: `[{st.session_state['raw_display_image'].min():.2f}, {st.session_state['raw_display_image'].max():.2f}]`
                    """)

                if btn_explain or st.button("🚀 Watch Neural Network Think (Step-by-Step)", type="primary"):
                    st.session_state["nav_page"] = "⚡ Live Inference (\"Watch Network Think\")"
                    st.session_state["live_stage"] = 0
                    st.rerun()

            except ValueError as ve:
                st.warning(f"⚠️ {ve}")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
        else:
            st.info("✍️ Draw any digit on the canvas or click a **'Digit Preset'** button above to test the model.")


# ==============================================================================
# SECTION 2: LIVE INFERENCE — "WATCH THE NEURAL NETWORK THINK"
# ==============================================================================
elif st.session_state["nav_page"] == "⚡ Live Inference (\"Watch Network Think\")":
    st.markdown('<div class="hero-title">Watch the Neural Network Think</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Step-by-step layer inspection extracting genuine intermediate feature maps and activations.</div>', unsafe_allow_html=True)

    # Ensure an active input tensor exists; if not, default to Digit 7
    if st.session_state["activations_data"] is None:
        if sample_digits_map and 7 in sample_digits_map:
            process_and_run_inference(sample_digits_map[7][0])

    act_data = st.session_state["activations_data"]
    if act_data is None:
        st.warning("⚠️ No input digit is currently loaded. Return to AI Lab to draw a digit.")
        st.stop()

    stages = [
        "1. Input Image & Preprocessing",
        "2. Conv2D Layer 1 (Edge & Contour Filters)",
        "3. Max Pooling 1 (Spatial Reduction)",
        "4. Conv2D Layer 2 (Hierarchical Feature Maps)",
        "5. Max Pooling 2 (Feature Compression)",
        "6. Flatten & Dense Features (Hidden Neurons)",
        "7. Softmax Output Layer (Probability Distribution)",
        "8. Complete Inference Journey Summary"
    ]

    # Step Controls Bar
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 1])
    with ctrl_col1:
        if st.button("⏮️ Previous Stage", disabled=(st.session_state["live_stage"] == 0), use_container_width=True):
            st.session_state["live_stage"] -= 1
            st.rerun()
    with ctrl_col2:
        selected_stage_idx = st.selectbox(
            "Inference Stage",
            range(len(stages)),
            format_func=lambda idx: stages[idx],
            index=st.session_state["live_stage"],
            label_visibility="collapsed"
        )
        if selected_stage_idx != st.session_state["live_stage"]:
            st.session_state["live_stage"] = selected_stage_idx
            st.rerun()
    with ctrl_col3:
        if st.button("Next Stage ⏭️", disabled=(st.session_state["live_stage"] == len(stages) - 1), use_container_width=True):
            st.session_state["live_stage"] += 1
            st.rerun()

    stage_idx = st.session_state["live_stage"]
    st.markdown(f"""
    <div class="stage-indicator">
        <span>Stage {stage_idx + 1} of 8: {stages[stage_idx]}</span>
        <span>Predicted: <strong>Digit {act_data['predicted_digit']}</strong> ({act_data['confidence']*100:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- STAGE 1: INPUT PREPROCESSING ----------------
    if stage_idx == 0:
        st.markdown('<div class="section-title">Stage 1: Input Tensor & Normalization</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(st.session_state["raw_display_image"], caption="Normalized Input Tensor (28×28)", width=240, clamp=True)
        with col2:
            st.markdown("""
            ### Mathematical Transformations:
            1. **Grayscale Conversion**: Multi-channel canvas color data is mapped to a single intensity matrix.
            2. **Polarity Check**: Ensures dark background ($0.0$) and bright stroke ($1.0$), matching MNIST statistical training conditions.
            3. **Bounding Box Isolation**: Discards extraneous border margins.
            4. **Center-of-Mass Alignment**: Calculates centroid $(c_x, c_y)$ of the drawn stroke mass and shifts it to frame center $(14, 14)$.
            """)
            st.markdown(f"""
            - **Input Shape**: `(1, 28, 28, 1)`
            - **Min Value**: `{st.session_state['raw_display_image'].min():.2f}`
            - **Max Value**: `{st.session_state['raw_display_image'].max():.2f}`
            - **Active Pixels**: `{np.sum(st.session_state['raw_display_image'] > 0.1)}`
            """)

        if st.session_state["learning_mode"]:
            st.markdown("""
            <div class="edu-callout">
                <strong>🎓 Learning Note: Why Polarity & Centering Matter</strong><br/>
                Deep neural networks are highly sensitive to statistical domain shifts. If a digit was inverted (black on white) or off-center, the early convolution kernels would activate in the wrong spatial coordinates, severely degrading accuracy.
            </div>
            """, unsafe_allow_html=True)

    # ---------------- STAGE 2: CONV2D LAYER 1 ----------------
    elif stage_idx == 1:
        st.markdown('<div class="section-title">Stage 2: First Convolution Layer (`conv2d_1`)</div>', unsafe_allow_html=True)
        conv1_act = act_data["activations"]["conv2d_1"]  # (1, 28, 28, 32)
        meta1 = act_data["layers_meta"]["conv2d_1"]

        st.write(f"The first layer applies **32 filters** of size **3×3** over the 28×28 image, producing a `(1, 28, 28, 32)` feature volume.")

        filter_idx = st.slider("Select Filter to Inspect (0 - 31):", min_value=0, max_value=31, value=0)
        fmap = conv1_act[0, :, :, filter_idx]

        fc1, fc2 = st.columns([1.2, 1.8])
        with fc1:
            fig_fmap = px.imshow(fmap, color_continuous_scale="Viridis", title=f"Activation Heatmap: Filter #{filter_idx}")
            fig_fmap.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fmap, use_container_width=True)
        with fc2:
            st.markdown(f"""
            #### Layer Parameters & Specs:
            - **Layer Name**: `{meta1['name']}`
            - **Filter Count**: `{meta1['filters']}`
            - **Kernel Dimensions**: `{meta1['kernel_size']}`
            - **Activation**: `{meta1['activation']}` (Rectified Linear Unit)
            - **Parameters**: `{meta1['params']}` $(3 \\times 3 \\times 1 \\times 32 + 32 \\text{ biases})$
            - **Filter Activation Mean**: `{np.mean(fmap):.4f}`
            - **Filter Activation Max**: `{np.max(fmap):.4f}`
            """)

        # Grid of top-8 most active feature maps
        st.markdown("#### Top 8 Most Energetic Feature Maps in Layer 1:")
        top_filters = get_top_active_filters(conv1_act, top_k=8)
        grid_cols = st.columns(8)
        for i, f_id in enumerate(top_filters):
            with grid_cols[i]:
                st.image(conv1_act[0, :, :, f_id], caption=f"Filter #{f_id}", width=85, clamp=True)

        if st.session_state["learning_mode"]:
            st.markdown("""
            <div class="edu-callout">
                <strong>🎓 Learning Note: What Do Early Convolution Filters Learn?</strong><br/>
                Early layers act as feature detectors sensitive to primitive visual cues: vertical strokes, horizontal bars, corners, and local intensity gradients. The ReLU activation ensures that negative responses are clipped to zero, inducing sparse representations.
            </div>
            """, unsafe_allow_html=True)

    # ---------------- STAGE 3: MAX POOLING 1 ----------------
    elif stage_idx == 2:
        st.markdown('<div class="section-title">Stage 3: First Max Pooling Layer (`maxpool_1`)</div>', unsafe_allow_html=True)
        pool1_act = act_data["activations"]["maxpool_1"]  # (1, 14, 14, 32)
        conv1_act = act_data["activations"]["conv2d_1"]

        st.write("2×2 Max Pooling reduces the spatial footprint from **28×28** to **14×14** by selecting the maximum activation in each 2×2 window.")

        filter_choice = st.slider("Select Filter to Compare (0 - 31):", 0, 31, 0, key="pool1_slider")
        comp_c1, comp_c2 = st.columns(2)
        with comp_c1:
            st.markdown("##### Before Pooling: `conv2d_1` (28×28)")
            st.image(conv1_act[0, :, :, filter_choice], width=200, clamp=True)
        with comp_c2:
            st.markdown("##### After Pooling: `maxpool_1` (14×14)")
            st.image(pool1_act[0, :, :, filter_choice], width=200, clamp=True)

        if st.session_state["learning_mode"]:
            st.markdown("""
            <div class="edu-callout">
                <strong>🎓 Learning Note: Why Max Pooling?</strong><br/>
                Pooling provides two crucial advantages:
                1. <strong>Spatial Invariance</strong>: Small shifts or wiggles in the digit strokes do not alter the pooled feature representation.
                2. <strong>Dimensionality Reduction</strong>: Reduces computation by 75% per channel, allowing subsequent layers to cover larger receptive fields.
            </div>
            """, unsafe_allow_html=True)

    # ---------------- STAGE 4: CONV2D LAYER 2 ----------------
    elif stage_idx == 3:
        st.markdown('<div class="section-title">Stage 4: Second Convolution Layer (`conv2d_2`)</div>', unsafe_allow_html=True)
        conv2_act = act_data["activations"]["conv2d_2"]  # (1, 14, 14, 64)
        meta2 = act_data["layers_meta"]["conv2d_2"]

        st.write("The second convolution layer operates over the pooled feature maps using **64 filters**, producing a `(1, 14, 14, 64)` representation.")

        f2_choice = st.slider("Select Filter to Inspect (0 - 63):", 0, 63, 0)
        fmap2 = conv2_act[0, :, :, f2_choice]

        c2_col1, c2_col2 = st.columns([1.2, 1.8])
        with c2_col1:
            fig2 = px.imshow(fmap2, color_continuous_scale="Inferno", title=f"Activation Heatmap: Filter #{f2_choice}")
            fig2.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        with c2_col2:
            st.markdown(f"""
            #### Layer Parameters & Specs:
            - **Layer Name**: `{meta2['name']}`
            - **Filter Count**: `{meta2['filters']}`
            - **Kernel Size**: `{meta2['kernel_size']}`
            - **Parameters**: `{meta2['params']:,}` $(3 \\times 3 \\times 32 \\times 64 + 64 \\text{ biases})$
            - **Output Volume**: `14 × 14 × 64` = **12,544 total activations**
            """)

        st.markdown("#### Top 8 Most Active Feature Maps in Layer 2:")
        top2_filters = get_top_active_filters(conv2_act, top_k=8)
        g2_cols = st.columns(8)
        for i, f_id in enumerate(top2_filters):
            with g2_cols[i]:
                st.image(conv2_act[0, :, :, f_id], caption=f"Filter #{f_id}", width=85, clamp=True)

        if st.session_state["learning_mode"]:
            st.markdown("""
            <div class="edu-callout">
                <strong>🎓 Learning Note: Hierarchical Feature Composition</strong><br/>
                Unlike the first layer which looked at raw pixel strokes, Layer 2 combines multiple primitive edge activations. Its filters recognize higher-level structural components: loops (as in 8, 0, 6), sharp joints (as in 4, 7), and curves (as in 3, 5).
            </div>
            """, unsafe_allow_html=True)

    # ---------------- STAGE 5: MAX POOLING 2 ----------------
    elif stage_idx == 4:
        st.markdown('<div class="section-title">Stage 5: Second Max Pooling Layer (`maxpool_2`)</div>', unsafe_allow_html=True)
        pool2_act = act_data["activations"]["maxpool_2"]  # (1, 7, 7, 64)

        st.write("Second 2×2 Max Pooling downsamples the feature volume from **14×14** to **7×7**, finalizing the spatial feature extraction stage.")

        p2_c1, p2_c2 = st.columns([1, 1.5])
        with p2_c1:
            st.markdown("##### 7×7 Downsampled Feature Map")
            st.image(pool2_act[0, :, :, 0], width=180, caption="Filter #0 (7×7 resolution)", clamp=True)
        with p2_c2:
            st.markdown("""
            #### Transition to Dense Classification:
            - **Final Spatial Tensor Shape**: `(1, 7, 7, 64)`
            - **Total Spatial Features**: $7 \\times 7 \\times 64 = 3,136$ values.
            - Next step: The 3D tensor is unrolled into a flat 1D vector and fed to fully connected classification layers.
            """)

    # ---------------- STAGE 6: FLATTEN & DENSE FEATURES ----------------
    elif stage_idx == 5:
        st.markdown('<div class="section-title">Stage 6: Flatten & Dense Hidden Layer (`dense_features`)</div>', unsafe_allow_html=True)
        dense_act = act_data["activations"]["dense_features"][0]  # (128,)
        active_neurons_count = int(np.sum(dense_act > 0))

        st.markdown(f"""
        The 3,136 spatial feature values are flattened and multiplied by weight matrix $W$ with bias $b$ to activate **128 hidden neurons**:
        $$\\text{{Output}} = \\text{{ReLU}}(X \\cdot W + b)$$
        - **Active Neurons ($> 0$)**: `{active_neurons_count} / 128` ({active_neurons_count/128*100:.1f}%)
        - **Dormant Neurons ($= 0$)**: `{128 - active_neurons_count}` (due to ReLU sparsity)
        """)

        # Neuron Activation Bar Chart
        bar_colors = ['#38BDF8' if val > 0 else '#334155' for val in dense_act]
        fig_dense = go.Figure(go.Bar(
            x=list(range(128)),
            y=dense_act,
            marker=dict(color=bar_colors),
            text=[f"{v:.2f}" for v in dense_act],
            hoverinfo="x+y"
        ))
        fig_dense.update_layout(
            title="Activation Magnitude Across All 128 Hidden Neurons",
            xaxis=dict(title="Neuron Index (0 - 127)", showgrid=False),
            yaxis=dict(title="ReLU Activation Value", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            height=320,
            margin=dict(l=40, r=20, t=35, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dense, use_container_width=True)

        # Top 5 most active neurons
        top_neurons = np.argsort(dense_act)[::-1][:5]
        st.markdown(f"**Top 5 Most Strongly Fired Neurons**: " + ", ".join([f"Neuron #{idx} (`{dense_act[idx]:.2f}`)" for idx in top_neurons]))

    # ---------------- STAGE 7: SOFTMAX OUTPUT ----------------
    elif stage_idx == 6:
        st.markdown('<div class="section-title">Stage 7: Softmax Output Layer (`output_probabilities`)</div>', unsafe_allow_html=True)
        probs = act_data["probabilities"]
        pred_digit = act_data["predicted_digit"]

        st.markdown(f"""
        The final dense layer produces 10 unnormalized logits $z_0, \\dots, z_9$. The **Softmax activation** normalizes them into a probability distribution:
        $$P(y = i \\mid X) = \\frac{{e^{{z_i}}}}{{\\sum_{{j=0}}^{{9}} e^{{z_j}}}}$$
        """)

        fig_soft = plot_probabilities_chart(probs, pred_digit)
        st.plotly_chart(fig_soft, use_container_width=True)

        st.markdown(f"""
        <div class="prediction-display-box" style="padding: 16px;">
            <div class="metric-tag">Winning Classification</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #34D399;">Digit {pred_digit} ({probs[pred_digit]*100:.2f}% Confidence)</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- STAGE 8: COMPLETE JOURNEY SUMMARY ----------------
    elif stage_idx == 7:
        st.markdown('<div class="section-title">Stage 8: Complete Inference Journey Summary</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="prediction-display-box">
            <div class="metric-tag" style="color: #38BDF8;">End-to-End Prediction Summary</div>
            <div class="pred-digit-giant">{act_data['predicted_digit']}</div>
            <div class="pred-confidence-lead">Confidence: {act_data['confidence'] * 100:.2f}%</div>
            <div style="margin-top: 10px; font-size: 0.9rem; color: #94A3B8;">
                ✓ Total Layers Inspected: <strong>{len(act_data['layer_names'])} layers</strong><br/>
                ✓ Actual Measured Inference Duration: <strong>{act_data['inference_time_ms']:.2f} ms</strong><br/>
                ✓ Execution Hardware: <strong>{hardware_info['device_string']} ({hardware_info['device_name']})</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Complete Pipeline Recap:")
        st.markdown("""
| Stage | Component | Input Shape | Output Shape | Parameters | Functional Role |
|---|---|---|---|---|---|
| **1** | Input Preprocessor | Canvas | `(28, 28, 1)` | 0 | Aspect-ratio bounding box, center-of-mass alignment |
| **2** | `conv2d_1` (Conv2D) | `(28, 28, 1)` | `(28, 28, 32)` | 320 | 32 local edge and stroke filters |
| **3** | `maxpool_1` (MaxPooling2D) | `(28, 28, 32)` | `(14, 14, 32)` | 0 | 2×2 max downsampling, spatial invariance |
| **4** | `conv2d_2` (Conv2D) | `(14, 14, 32)` | `(14, 14, 64)` | 18,496 | 64 hierarchical structural filters (curves, loops) |
| **5** | `maxpool_2` (MaxPooling2D) | `(14, 14, 64)` | `(7, 7, 64)` | 0 | 2×2 feature compression |
| **6** | `flatten` & `dense_features` | `(7, 7, 64)` | `(128,)` | 401,536 | Unrolling to 3,136 features -> 128 hidden units |
| **7** | `dropout_regularization` | `(128,)` | `(128,)` | 0 | Overfitting regularizer (inference mode: pass-through) |
| **8** | `output_probabilities` | `(128,)` | `(10,)` | 1,290 | Softmax probability distribution over classes 0–9 |
""")

        if st.button("🎨 Test Another Digit in AI Lab", type="primary"):
            st.session_state["nav_page"] = "🏠 Home Dashboard & AI Lab"
            st.rerun()


# ==============================================================================
# SECTION 3: NEURAL NETWORK EXPLORER
# ==============================================================================
elif st.session_state["nav_page"] == "🧠 Neural Network Explorer":
    st.markdown('<div class="hero-title">Neural Network Architecture Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Interactive structural blueprint of the Convolutional Neural Network.</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("Model not found. Train first via `python train.py`.")
        st.stop()

    # Architecture Overview Metrics
    total_params = model.count_params()
    trainable_params = sum(np.prod(w.shape) for w in model.trainable_weights)
    layers_count = len(model.layers)

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Total Layers", layers_count)
    with a2:
        st.metric("Total Parameters", f"{total_params:,}")
    with a3:
        st.metric("Trainable Parameters", f"{trainable_params:,}")
    with a4:
        st.metric("Model Footprint", f"{total_params * 4 / (1024*1024):.2f} MB (FP32)")

    st.markdown('<div class="section-title">🔍 Interactive Layer Inspector</div>', unsafe_allow_html=True)
    st.caption("Select a layer from the architectural pipeline to inspect its weights, dimensions, and mathematical function:")

    layer_names = [l.name for l in model.layers]
    selected_layer_name = st.selectbox("Select Layer to Inspect:", layer_names, index=0)
    selected_layer = model.get_layer(selected_layer_name)
    meta = get_layer_metadata(selected_layer)

    col_meta1, col_meta2 = st.columns([1, 1])
    with col_meta1:
        st.markdown(f"""
        ### Specifications: `{meta['name']}`
        - **Type**: `{meta['type']}`
        - **Output Shape**: `{meta['output_shape']}`
        - **Parameter Count**: `{meta['params']:,}`
        - **Trainable Parameters**: `{meta['trainable_params']:,}`
        - **Activation Function**: `{meta['activation']}`
        """)
        if meta["type"] == "Conv2D":
            st.markdown(f"- **Filters**: `{meta['filters']}` | **Kernel**: `{meta['kernel_size']}` | **Padding**: `{meta['padding']}`")
        elif meta["type"] == "MaxPooling2D":
            st.markdown(f"- **Pool Size**: `{meta['pool_size']}` | **Strides**: `{meta['strides']}`")
        elif meta["type"] == "Dense":
            st.markdown(f"- **Hidden Units**: `{meta['units']}`")
        elif meta["type"] == "Dropout":
            st.markdown(f"- **Dropout Rate**: `{meta['rate']}` (30% dropped during training)")

    with col_meta2:
        st.markdown("### Functional Role in Digit Recognition:")
        role_map = {
            "conv2d_1": "Primary feature extractor. Convolves 32 separate 3×3 kernels directly across input pixel gradients to detect strokes, vertical bars, and corners.",
            "maxpool_1": "Spatial downsampler. Compresses 28×28 grids to 14×14 by taking the maximum activation in 2×2 blocks. Imparts local translation invariance.",
            "conv2d_2": "Higher-order synthesizer. Combines primitive stroke features into complex patterns such as loops, curves, and junctions.",
            "maxpool_2": "Final downsampler. Compresses 14×14 representations to 7×7 to minimize dimensionality before dense feature mapping.",
            "flatten": "Reshaper. Unrolls the 7×7×64 3D feature representation into a 1D vector of 3,136 distinct attributes.",
            "dense_features": "Fully connected feature learner. Blends 3,136 spatial attributes through 401,536 learnable weights into 128 high-level semantic hidden units.",
            "dropout_regularization": "Regularizer. Randomly zeroes 30% of activations during training to prevent co-adaptation. Operates in pass-through mode during inference.",
            "output_probabilities": "Classifier. Computes 10 logits and normalizes them via Softmax to yield relative probability distribution over digits 0–9."
        }
        st.info(role_map.get(meta["name"], "Processes intermediate neural representations."))

    # Parameter Distribution Chart
    st.markdown('<div class="section-title">📊 Parameter Distribution Across Layers</div>', unsafe_allow_html=True)
    layer_param_data = [(l.name, l.count_params()) for l in model.layers if l.count_params() > 0]
    p_names = [x[0] for x in layer_param_data]
    p_counts = [x[1] for x in layer_param_data]

    fig_params = go.Figure(go.Pie(
        labels=p_names,
        values=p_counts,
        hole=0.45,
        marker=dict(colors=['#38BDF8', '#818CF8', '#34D399', '#FBBF24']),
        textinfo='label+percent',
        insidetextorientation='radial'
    ))
    fig_params.update_layout(
        title="Distribution of Learnable Parameters (Dominated by Dense Layer)",
        height=320,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_params, use_container_width=True)


# ==============================================================================
# SECTION 4: MODEL TRAINING DASHBOARD
# ==============================================================================
elif st.session_state["nav_page"] == "🏋️ Model Training Dashboard":
    st.markdown('<div class="hero-title">Model Training Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Inspect training trajectory, convergence curves, and optimization dynamics.</div>', unsafe_allow_html=True)

    if history_data is None:
        st.warning("⚠️ Training history not found. Train the model by running `python train.py`.")
        st.stop()

    acc = history_data.get("accuracy", [])
    val_acc = history_data.get("val_accuracy", [])
    loss = history_data.get("loss", [])
    val_loss = history_data.get("val_loss", [])
    epochs = list(range(1, len(acc) + 1))

    best_val_acc = max(val_acc) if val_acc else 0.0
    best_val_epoch = val_acc.index(best_val_acc) + 1 if val_acc else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Epochs", len(epochs))
    with m2:
        st.metric("Best Val Accuracy", f"{best_val_acc*100:.2f}%", f"Epoch {best_val_epoch}")
    with m3:
        st.metric("Final Training Loss", f"{loss[-1]:.4f}" if loss else "N/A")
    with m4:
        st.metric("Early Stopping", "Triggered (Best Restored)")

    st.markdown('<div class="section-title">📈 Learning Curves (Accuracy & Loss)</div>', unsafe_allow_html=True)

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=[a*100 for a in acc], mode='lines+markers', name='Train Accuracy', line=dict(color='#38BDF8', width=2.5)))
        fig_acc.add_trace(go.Scatter(x=epochs, y=[va*100 for va in val_acc], mode='lines+markers', name='Val Accuracy', line=dict(color='#34D399', width=2.5, dash='dash')))
        fig_acc.update_layout(
            title="Accuracy per Epoch (%)",
            xaxis=dict(title="Epoch", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Accuracy (%)", range=[88, 100], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            height=340,
            margin=dict(l=40, r=20, t=35, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with chart_c2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Train Loss', line=dict(color='#F87171', width=2.5)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Val Loss', line=dict(color='#FBBF24', width=2.5, dash='dash')))
        fig_loss.update_layout(
            title="Cross-Entropy Loss per Epoch",
            xaxis=dict(title="Epoch", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Loss", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            height=340,
            margin=dict(l=40, r=20, t=35, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Deep learning theory cards
    st.markdown('<div class="section-title">🧠 Optimization Concepts</div>', unsafe_allow_html=True)
    th1, th2, th3 = st.columns(3)
    with th1:
        st.markdown("""
        **1. Forward & Backward Pass:**
        - **Forward Pass**: Images propagate through Conv, Pool, and Dense layers to compute predictions.
        - **Loss Calculation**: Sparse Categorical Cross-Entropy quantifies the divergence between predictions and true labels.
        """)
    with th2:
        st.markdown("""
        **2. Gradient Descent & Adam:**
        - **Backpropagation**: Gradients of the loss with respect to all 421,642 weights are computed via chain rule.
        - **Adam Optimizer**: Combines momentum and adaptive learning rates for fast, stable convergence.
        """)
    with th3:
        st.markdown("""
        **3. Regularization:**
        - **Dropout (30%)**: Prevents neurons from co-adapting to spurious training patterns.
        - **Early Stopping**: Halts training when validation loss stops improving, restoring optimal weights.
        """)


# ==============================================================================
# SECTION 5: MODEL EVALUATION DASHBOARD
# ==============================================================================
elif st.session_state["nav_page"] == "📊 Model Evaluation Dashboard":
    st.markdown('<div class="hero-title">Model Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Comprehensive evaluation metrics and confusion analysis on the 10,000 untouched MNIST test set.</div>', unsafe_allow_html=True)

    if metrics_data is None:
        st.warning("⚠️ Metrics not found. Run `python evaluate.py` to evaluate the model.")
        st.stop()

    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        st.metric("Test Accuracy", f"{metrics_data.get('test_accuracy', 0.0) * 100:.2f}%")
    with e2:
        st.metric("Test Loss", f"{metrics_data.get('test_loss', 0.0):.4f}")
    with e3:
        st.metric("Macro Precision", f"{metrics_data.get('precision_macro', 0.0) * 100:.2f}%")
    with e4:
        st.metric("Macro Recall", f"{metrics_data.get('recall_macro', 0.0) * 100:.2f}%")
    with e5:
        st.metric("Macro F1-Score", f"{metrics_data.get('f1_score_macro', 0.0) * 100:.2f}%")

    st.markdown(f"""
    <div class="edu-callout">
        <strong>📌 Academic Note: Why the Test Set Must Remain Isolated</strong><br/>
        {metrics_data.get('rationale_test_split', '')}
    </div>
    """, unsafe_allow_html=True)

    tab_cm, tab_mis, tab_report = st.tabs(["📉 Confusion Matrix", "🔍 Misclassified Examples Inspector", "📋 Full Classification Report"])

    with tab_cm:
        cm_path = "results/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on MNIST 10,000 Test Images", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    with tab_mis:
        st.markdown("#### Interactive Misclassification Browser")
        st.caption("Inspect genuine test samples that confused the neural network:")

        if misclass_data is not None:
            filter_digit = st.selectbox("Filter by Actual True Label:", list(range(10)), index=4)
            filtered_samples = [item for item in misclass_data if item["true_label"] == filter_digit]
            st.write(f"Total misclassifications for **Digit {filter_digit}**: **{len(filtered_samples)} out of ~1,000 test samples**")

            if filtered_samples:
                sample_pick = st.slider("Select Sample Case:", 0, len(filtered_samples) - 1, 0)
                selected_mis = filtered_samples[sample_pick]

                # Load genuine sample from test set
                (_, _), (X_test_all, _) = load_mnist_raw()
                raw_mis_img = X_test_all[selected_mis["index"]]

                mc_col1, mc_col2 = st.columns([1, 1.5])
                with mc_col1:
                    st.image(raw_mis_img, caption=f"Test Sample #{selected_mis['index']} (True: {selected_mis['true_label']})", width=160)
                with mc_col2:
                    st.markdown(f"""
                    - **Actual True Label**: `Digit {selected_mis['true_label']}`
                    - **Model Predicted Label**: `Digit {selected_mis['pred_label']}` ❌
                    - **Model Confidence**: `{selected_mis['confidence']*100:.2f}%`
                    """)
                    fig_mis_prob = plot_probabilities_chart(selected_mis["probabilities"], selected_mis["pred_label"])
                    st.plotly_chart(fig_mis_prob, use_container_width=True)
            else:
                st.success(f"🎉 Perfect classification! Zero misclassifications found for Digit {filter_digit}.")
        else:
            st.info("Misclassifications file not found.")

    with tab_report:
        report_path = "results/classification_report.txt"
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                rep_text = f.read()
            st.code(rep_text, language="text")


# ==============================================================================
# SECTION 6: MNIST DATASET EXPLORER
# ==============================================================================
elif st.session_state["nav_page"] == "🔍 MNIST Dataset Explorer":
    st.markdown('<div class="hero-title">MNIST Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Interactive browser for the 70,000 standard benchmark grayscale digit images.</div>', unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1, 2])
    with d_col1:
        digit_filter_choice = st.selectbox("Filter by Digit Class:", ["All Digits"] + [f"Digit {i}" for i in range(10)], index=0)
    with d_col2:
        num_display = st.slider("Number of Samples to Display:", 8, 32, 16, step=8)

    filter_val = None if digit_filter_choice == "All Digits" else int(digit_filter_choice.split()[-1])
    subset_data = get_mnist_subset(count=num_display, digit_filter=filter_val)

    st.markdown("#### Sample Grid:")
    grid_cols = st.columns(8)
    for i, (img, lbl) in enumerate(subset_data):
        col_idx = i % 8
        with grid_cols[col_idx]:
            st.image(img, caption=f"Label: {lbl}", width=80)

    # Pixel intensity distribution
    st.markdown('<div class="section-title">📊 Pixel Intensity Distribution</div>', unsafe_allow_html=True)
    if subset_data:
        all_pixels = np.concatenate([img.flatten() for img, _ in subset_data[:10]])
        fig_hist = px.histogram(
            x=all_pixels,
            nbins=30,
            title="Pixel Intensity Distribution across Sample Digits (0 = Black Background, 255 = White Stroke)",
            labels={'x': 'Pixel Value (0-255)'}
        )
        fig_hist.update_layout(
            height=300,
            margin=dict(l=40, r=20, t=35, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ==============================================================================
# SECTION 7: ABOUT & LEARNING MODE
# ==============================================================================
elif st.session_state["nav_page"] == "📖 About & Learning Mode":
    st.markdown('<div class="hero-title">About the Project & Learning Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Theoretical foundations of Convolutional Neural Networks and interactive knowledge checks.</div>', unsafe_allow_html=True)

    tab_about, tab_quiz, tab_glossary = st.tabs(["📘 Theoretical Foundations", "🎯 Interactive Knowledge Quiz", "📖 Machine Learning Glossary"])

    with tab_about:
        st.markdown("""
        ### Problem Statement
        Handwritten digit recognition is a benchmark computer vision challenge. Traditional rule-based algorithms fail due to the virtually infinite variation in human writing styles, slants, strokes, and pressures. **Convolutional Neural Networks (CNNs)** overcome this by learning hierarchical representations directly from pixel matrices through spatial convolutions.

        ### Why CNNs Outperform Multilayer Perceptrons (MLPs)
        1. **Parameter Sharing**: Instead of learning a separate weight for every single pixel-to-neuron connection, convolution kernels share the exact same weights across the entire image.
        2. **Translation Equivariance**: A loop or stroke is recognized regardless of whether it appears in the top-left or bottom-right corner of the canvas.
        3. **Receptive Field Hierarchy**: Early layers capture simple edges; deeper layers capture full loops, curves, and digit geometry.

        ### NVIDIA RTX 3050 & WSL2 Hardware Acceleration
        - **Auto-Detection**: Probes `tf.config.list_physical_devices('GPU')`.
        - **Dynamic Memory Growth**: Allocates VRAM on demand to prevent out-of-memory errors on 4 GB VRAM GPUs.
        - **WSL2 Deployment**: For GPU training, execute in WSL2 Ubuntu using `tensorflow[and-cuda]`.
        """)

    with tab_quiz:
        st.markdown("### Test Your Neural Network Knowledge:")
        
        # Quiz 1
        q1 = st.radio(
            "1. What is the primary function of a Convolution filter (kernel)?",
            [
                "To randomly invert pixel colors",
                "To detect local spatial patterns (edges, lines, curves) via element-wise multiplication and summation",
                "To convert grayscale images into RGB",
                "To reduce the number of output classes"
            ],
            index=None
        )
        if q1 is not None:
            if "detect local spatial patterns" in q1:
                st.success("✅ Correct! Convolution kernels perform spatial filtering to highlight edges, contours, and strokes.")
            else:
                st.error("❌ Incorrect. Convolution kernels slide over the image to detect local features like edges and curves.")

        st.markdown("---")

        # Quiz 2
        q2 = st.radio(
            "2. Why is Max Pooling used immediately following a Convolution layer?",
            [
                "To increase image resolution",
                "To downsample spatial dimensions, reduce computation, and provide translation invariance",
                "To increase the number of trainable weights",
                "To convert negative values into positive values"
            ],
            index=None
        )
        if q2 is not None:
            if "downsample spatial dimensions" in q2:
                st.success("✅ Correct! Max Pooling selects the strongest local activation, reducing dimensions while preserving key features.")
            else:
                st.error("❌ Incorrect. Max Pooling downsamples feature maps to reduce computation and improve translation invariance.")

        st.markdown("---")

        # Quiz 3
        q3 = st.radio(
            "3. What is the mathematical purpose of the Softmax activation function in the final layer?",
            [
                "To eliminate all zero values",
                "To compute an exponential moving average",
                "To convert raw network logits into a valid probability distribution summing to 1.0",
                "To calculate gradient descent steps"
            ],
            index=None
        )
        if q3 is not None:
            if "convert raw network logits" in q3:
                st.success("✅ Correct! Softmax exponents each logit and normalizes the sum to 1.0, enabling probabilistic interpretation.")
            else:
                st.error("❌ Incorrect. Softmax maps unconstrained logits into relative probabilities summing to 1.0.")

    with tab_glossary:
        st.markdown("""
| Term | Formal Definition |
|---|---|
| **Convolution** | Mathematical operation applying a sliding filter over an input matrix to produce a feature map. |
| **Kernel / Filter** | A small 2D matrix of learnable weights (e.g. 3×3) that detects specific visual patterns. |
| **Stride** | The step size by which the filter slides across the input image. |
| **Padding ('same')** | Padding borders with zeros so the output feature map maintains the exact same spatial dimensions. |
| **Receptive Field** | The specific patch of the input image that directly influences the activation of a given neuron. |
| **Max Pooling** | Downsampling operation selecting the maximum value across a 2×2 spatial window. |
| **ReLU** | Rectified Linear Unit activation: $f(x) = \\max(0, x)$, which introduces non-linearity and promotes sparsity. |
| **Logits** | Unnormalized real-valued outputs produced by the final dense layer before Softmax. |
| **Softmax** | Normalization function $\\sigma(z)_i = \\frac{e^{z_i}}{\\sum e^{z_j}}$ creating a probability distribution. |
| **Cross-Entropy Loss** | Loss function measuring the logarithmic distance between true probability and predicted probability. |
""")
