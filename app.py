import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Agricultural Crop Disease Detection with Treatment Recommendation",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS - Agricultural Theme
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --soil:       #3d2b1f;
    --bark:       #5c3d2e;
    --wheat:      #c8a96e;
    --wheat-lt:   #e8d5a3;
    --sage:       #4a7c59;
    --sage-lt:    #6aab7a;
    --sage-dk:    #2d5a3d;
    --meadow:     #8fbc5a;
    --sky:        #c8dfc8;
    --parchment:  #faf6ee;
    --cream:      #f5f0e8;
    --rust:       #c0392b;
    --amber:      #e67e22;
    --card-bg:    #fffdf7;
    --border:     #d4c5a0;
    --shadow:     rgba(61, 43, 31, 0.12);
    --text-dark:  #2c1810;
    --text-mid:   #5a4a3a;
    --text-light: #8b7355;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-dark);
}

.stApp {
    background-color: var(--parchment);
    background-image:
        radial-gradient(circle at 15% 20%, rgba(74,124,89,0.06) 0%, transparent 50%),
        radial-gradient(circle at 85% 80%, rgba(200,169,110,0.08) 0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%234a7c59' fill-opacity='0.025'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1300px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sage-dk) 0%, var(--soil) 100%) !important;
    border-right: 3px solid var(--wheat) !important;
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] .stSelectbox label {
    color: var(--wheat-lt) !important;
    font-weight: 500; font-size: 0.8rem;
    letter-spacing: 0.08em; text-transform: uppercase;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(200,169,110,0.4) !important;
    color: var(--cream) !important; border-radius: 8px;
}
[data-testid="stSidebar"] hr { border-color: rgba(200,169,110,0.3) !important; }

/* Headings */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 800 !important; color: var(--sage-dk) !important;
    letter-spacing: -0.02em !important; line-height: 1.1 !important;
}
h2, h3 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important; color: var(--bark) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    border-radius: 50px !important; padding: 0.6rem 1.8rem !important;
    transition: all 0.25s ease !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important; font-size: 0.8rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--sage) 0%, var(--sage-dk) 100%) !important;
    border: none !important; color: white !important;
    box-shadow: 0 4px 15px rgba(45,90,61,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(45,90,61,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 2px solid var(--sage) !important; color: var(--sage-dk) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--sage) !important; color: white !important;
}

/* Download button */
.stDownloadButton > button {
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    border-radius: 50px !important; padding: 0.6rem 1.8rem !important;
    background: linear-gradient(135deg, var(--bark) 0%, var(--soil) 100%) !important;
    border: none !important; color: white !important;
    box-shadow: 0 4px 15px rgba(61,43,31,0.3) !important;
    transition: all 0.25s ease !important;
    font-size: 0.8rem !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.stDownloadButton > button * {
    color: white !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(61,43,31,0.4) !important;
}
.stDownloadButton > button:hover * {
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--card-bg) !important;
    border: 2px dashed var(--wheat) !important;
    border-radius: 16px !important; padding: 1.5rem !important;
    transition: border-color 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--sage) !important; }

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--sage) 0%, var(--meadow) 100%) !important;
    border-radius: 50px !important;
}
.stProgress > div > div > div {
    background: var(--sky) !important; border-radius: 50px !important; height: 12px !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    color: var(--sage-dk) !important; padding: 0.75rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { background: var(--sky) !important; }

/* Divider */
hr {
    border: none !important; border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--sage) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--wheat); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--sage); }

/* ── Sidebar inner padding wrapper ── */
.css-1544g2n.e1fqkh3o4 {
    padding-top: 2rem !important;
    padding-bottom: 0.5rem !important;
}
/* Fallback for other Streamlit versions */
[data-testid="stSidebar"] > div:first-child > div:first-child > div {
    padding-top: 2rem !important;
}

/* ── Sidebar Radio Nav ── */
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.35rem !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    padding: 0.28rem 0.7rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: rgba(232, 213, 163, 0.75) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(200, 169, 110, 0.15) !important;
    color: #e8d5a3 !important;
    border-color: rgba(200, 169, 110, 0.25) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] {
    background: rgba(200, 169, 110, 0.18) !important;
    color: #f5f0e8 !important;
    border-color: rgba(200, 169, 110, 0.5) !important;
    font-weight: 600 !important;
}
/* Hide the native radio circle */
[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}
/* Active/selected item - target the checked input's parent label */
[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div {
    color: #f5f0e8 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"]:checked ~ div {
    color: #e8d5a3 !important;
}
/* Selected label highlight via has() */
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(200,169,110,0.28) 0%, rgba(143,188,90,0.18) 100%) !important;
    color: #faf6ee !important;
    border-color: rgba(200, 169, 110, 0.55) !important;
    font-weight: 600 !important;
    box-shadow: inset 0 0 0 1px rgba(200,169,110,0.2) !important;
}
/* Nav section label above radio */
[data-testid="stSidebar"] .nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(200,169,110,0.6);
    padding: 0 0.3rem 0.4rem 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_treatments():
    path = os.path.join(os.path.dirname(__file__), "treatments.json")
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5")


CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]

CROP_ICONS = {
    "Apple": "🍎", "Blueberry": "🫐", "Cherry": "🍒", "Corn": "🌽", "Grape": "🍇",
    "Orange": "🍊", "Peach": "🍑", "Pepper": "🫑", "Potato": "🥔", "Raspberry": "🫐",
    "Soybean": "🌱", "Squash": "🥦", "Strawberry": "🍓", "Tomato": "🍅",
}

SEVERITY_CONFIG = {
    "None":           {"color": "#27ae60", "bg": "#eafaf1", "border": "#a9dfbf"},
    "Low":            {"color": "#f39c12", "bg": "#fef9e7", "border": "#f9e79f"},
    "Moderate":       {"color": "#e67e22", "bg": "#fef5e7", "border": "#fad7a0"},
    "High":           {"color": "#e74c3c", "bg": "#fdedec", "border": "#f5b7b1"},
    "Extremely High": {"color": "#922b21", "bg": "#f9ebea", "border": "#e74c3c"},
}


def get_severity_cfg(sev_str):
    for key, cfg in SEVERITY_CONFIG.items():
        if key.lower() in sev_str.lower():
            return cfg
    return SEVERITY_CONFIG["Moderate"]


def get_crop_icon(class_key):
    for crop, icon in CROP_ICONS.items():
        if class_key.startswith(crop):
            return icon
    return "🌿"


def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = np.array([tf.keras.preprocessing.image.img_to_array(image)])
    predictions = model.predict(input_arr)[0]
    top3 = np.argsort(predictions)[::-1][:3]
    return [{"class": CLASS_NAMES[i], "confidence": float(predictions[i])} for i in top3]


# ════════════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════════

def page_header(title, subtitle="", icon="🌾"):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#2d5a3d 0%,#4a7c59 60%,#3d2b1f 100%);
        border-radius:20px;padding:2.5rem 3rem;margin-bottom:2.5rem;position:relative;
        overflow:hidden;box-shadow:0 8px 32px rgba(45,90,61,0.3);">
      <div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;
          border-radius:50%;background:rgba(200,169,110,0.15);"></div>
      <div style="position:absolute;bottom:-30px;right:80px;width:120px;height:120px;
          border-radius:50%;background:rgba(143,188,90,0.12);"></div>
      <div style="position:absolute;top:20px;right:30px;font-size:4rem;opacity:0.18;">{icon}</div>
      <div style="position:absolute;bottom:12px;left:0;right:0;text-align:center;
          opacity:0.15;font-size:0.6rem;letter-spacing:8px;color:#c8a96e;">
        ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦</div>
      <div style="position:relative;z-index:1;">
        <div style="font-family:'DM Sans',sans-serif;font-size:0.7rem;font-weight:600;
            letter-spacing:0.15em;text-transform:uppercase;color:#c8a96e;margin-bottom:0.5rem;">
          Agricultural Crop Disease Detection with Treatment Recommendation</div>
        <h1 style="font-family:'Playfair Display',serif!important;font-size:2.4rem!important;
            font-weight:800!important;color:#faf6ee!important;margin:0 0 0.5rem 0!important;
            line-height:1.15!important;text-shadow:0 2px 8px rgba(0,0,0,0.2);">{title}</h1>
        {f'<p style="color:rgba(232,213,163,0.85);font-size:1rem;margin:0;font-weight:300;">{subtitle}</p>' if subtitle else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)


def section_title(text, icon=""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;margin:2rem 0 1rem 0;
        padding-bottom:0.6rem;border-bottom:2px solid #c8a96e;">
      <span style="font-size:1.2rem;">{icon}</span>
      <span style="font-family:'Playfair Display',serif;font-size:1.25rem;
          font-weight:700;color:#2d5a3d;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def stat_card(icon, value, label, color="#4a7c59"):
    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:1.2rem 1rem;text-align:center;
        border:1px solid #d4c5a0;box-shadow:0 2px 12px rgba(61,43,31,0.08);">
      <div style="font-size:1.8rem;margin-bottom:0.3rem;">{icon}</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;
          color:{color};line-height:1.1;">{value}</div>
      <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;
          color:#8b7355;margin-top:0.2rem;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def confidence_bar(label, confidence, color, rank=0, crop_icon="🌿"):
    medals = ["🥇", "🥈", "🥉"]
    medal = medals[rank] if rank < 3 else ""
    pct = confidence * 100
    w = max(int(pct), 2)
    hi = rank == 0
    st.markdown(f"""
    <div style="background:white;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.7rem;
        border:1px solid {'#a9dfbf' if hi else '#d4c5a0'};
        box-shadow:{'0 3px 14px rgba(45,90,61,0.15)' if hi else '0 1px 6px rgba(61,43,31,0.06)'};">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <span style="font-size:1rem;">{medal}</span>
          <span style="font-size:1rem;">{crop_icon}</span>
          <span style="font-family:'DM Sans',sans-serif;font-weight:600;font-size:0.9rem;color:#2c1810;">{label}</span>
        </div>
        <span style="font-family:'DM Mono',monospace;font-size:0.95rem;font-weight:500;color:{color};
            background:{'#eafaf1' if hi else '#f5f0e8'};padding:0.1rem 0.6rem;border-radius:20px;">{pct:.1f}%</span>
      </div>
      <div style="background:#f0ece4;border-radius:50px;height:10px;overflow:hidden;">
        <div style="width:{w}%;height:100%;background:linear-gradient(90deg,{color},{'#8fbc5a' if hi else '#c8a96e'});
            border-radius:50px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def treatment_item(text, bullet, color):
    return f"""<div style="display:flex;gap:0.6rem;margin-bottom:0.6rem;align-items:flex-start;">
      <span style="color:{color};font-size:1rem;line-height:1.5;flex-shrink:0;margin-top:0.05rem;">{bullet}</span>
      <span style="font-family:'DM Sans',sans-serif;font-size:0.88rem;line-height:1.6;color:#3d2b1f;">{text}</span>
    </div>"""


def treatment_col(title, icon, items, bg, border, color, bullet):
    body = "".join([treatment_item(i, bullet, color) for i in items]) if items else \
        '<div style="color:#8b7355;font-style:italic;font-size:0.85rem;">No specific treatments required.</div>'
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-top:4px solid {color};
        border-radius:14px;padding:1.3rem 1.2rem;height:100%;box-shadow:0 2px 10px rgba(61,43,31,0.07);">
      <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
        <span style="font-size:1.3rem;">{icon}</span>
        <span style="font-family:'DM Sans',sans-serif;font-size:0.8rem;font-weight:700;
            letter-spacing:0.08em;text-transform:uppercase;color:{color};">{title}</span>
      </div>
      {body}
    </div>
    """, unsafe_allow_html=True)


def symptom_badge(s):
    return f"""<div style="display:flex;gap:0.6rem;align-items:flex-start;background:white;
        border:1px solid #d4c5a0;border-left:4px solid #e67e22;border-radius:8px;
        padding:0.6rem 0.9rem;margin-bottom:0.5rem;">
      <span style="color:#e67e22;font-size:0.85rem;margin-top:0.1rem;">⚠</span>
      <span style="font-size:0.86rem;color:#3d2b1f;line-height:1.5;">{s}</span>
    </div>"""


def display_treatment_card(class_key, treatments_db):
    info = treatments_db.get(class_key)
    if not info:
        st.warning("Treatment data not found. Check treatments.json.")
        return

    is_healthy = info["pathogen"] == "None"
    sev = get_severity_cfg(info["severity"])
    crop_icon = get_crop_icon(class_key)

    # Banner
    if is_healthy:
        bg = "linear-gradient(135deg,#1a7a4a 0%,#27ae60 100%)"
        title = "✅  Plant Appears Healthy"
        sub = "No disease detected. See care tips below."
    else:
        bg = f"linear-gradient(135deg,{sev['color']}cc 0%,{sev['color']} 100%)"
        title = f"🦠  {info['disease_name']}"
        sub = f"{info['crop']}  ·  Severity: {info['severity']}  ·  {info['pathogen']}"

    st.markdown(f"""
    <div style="background:{bg};border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem;
        box-shadow:0 4px 20px {sev['color']}40;display:flex;align-items:center;gap:1rem;">
      <div style="font-size:3rem;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.2));">{crop_icon}</div>
      <div>
        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;
            color:white;text-shadow:0 1px 4px rgba(0,0,0,0.25);">{title}</div>
        <div style="color:rgba(255,255,255,0.82);font-size:0.85rem;margin-top:0.2rem;">{sub}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Description
    st.markdown(f"""
    <div style="background:{sev['bg']};border:1px solid {sev['border']};border-radius:12px;
        padding:1rem 1.3rem;margin-bottom:1.5rem;font-size:0.9rem;line-height:1.65;color:#3d2b1f;">
      {info['description']}
    </div>
    """, unsafe_allow_html=True)

    # Symptoms
    if info["symptoms"]:
        section_title("Symptoms to Watch For", "🔍")
        st.markdown("".join([symptom_badge(s) for s in info["symptoms"]]), unsafe_allow_html=True)

    # Treatments
    section_title("Treatment Recommendations", "💊")
    c1, c2, c3 = st.columns(3)
    t = info["treatments"]
    with c1:
        treatment_col("Chemical Treatments", "🧪", t["chemical"],
                      "#fef9f9", "#f5c6c6", "#c0392b", "💊")
    with c2:
        treatment_col("Cultivation Practices", "🌾", t["cultural"],
                      "#f5faf5", "#a9dfbf", "#2d7a4a", "🌿")
    with c3:
        treatment_col("Organic Options", "🍃", t["organic"],
                      "#fafff5", "#c2e5a0", "#5d9e2f", "🌱")

    # Prevention tip
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#fef9e7 0%,#fef5e7 100%);border:1px solid #f9e79f;
        border-left:5px solid #f1c40f;border-radius:12px;padding:1.1rem 1.4rem;margin-top:1.5rem;
        display:flex;gap:0.8rem;align-items:flex-start;">
      <span style="font-size:1.3rem;flex-shrink:0;">💡</span>
      <div>
        <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
            color:#b7770d;margin-bottom:0.3rem;">Prevention Tip</div>
        <div style="font-size:0.88rem;line-height:1.6;color:#3d2b1f;">{info['prevention']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.6rem 0 0.8rem 0;">
      <div style="font-size:1.8rem;margin-bottom:0.2rem;">🌿</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:800;color:#e8d5a3;line-height:1.3;">Agricultural Crop Disease Detection with Treatment Recommendation</div>
      <div style="font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:rgba(200,169,110,0.7);margin-top:0.2rem;">Powered by DenseNet121</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigate</div>', unsafe_allow_html=True)
    app_mode = st.radio(
        "NAVIGATE",
        ["🏡  Home", "🔬  Disease Recognition", "📊  Model Insights", "📖  About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
        text-transform:uppercase;color:rgba(200,169,110,0.7);margin-bottom:0.8rem;padding:0 0.2rem;">
        System Info</div>""", unsafe_allow_html=True)

    for icon, val, lbl in [("🌱","38","Disease Classes"),("🌍","14","Crop Species"),("📷","54K+","Training Images")]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.7rem;padding:0.5rem 0.6rem;
            margin-bottom:0.4rem;background:rgba(255,255,255,0.07);border-radius:8px;">
          <span style="font-size:1.1rem;">{icon}</span>
          <div>
            <div style="font-size:1rem;font-weight:700;color:#e8d5a3;line-height:1.1;">{val}</div>
            <div style="font-size:0.65rem;color:rgba(200,169,110,0.65);text-transform:uppercase;letter-spacing:0.06em;">{lbl}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""<div style="text-align:center;font-size:0.72rem;color:rgba(200,169,110,0.5);padding-bottom:0.5rem;">
        CS 719 · University of Regina<br><span style="opacity:0.7;">{datetime.now().strftime('%B %Y')}</span>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════

treatments_db = load_treatments()

# ════════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ════════════════════════════════════════════════════════════════════════════════

if "Home" in app_mode:
    page_header("Crop Disease Detection",
                "DenseNet-powered plant leaf analysis with instant diagnosis and evidence-based treatment guidance.", "🌿")

    image_path = "home_page.png"
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True,
                 caption="🤖 Image generated using AI - for illustrative purposes only.")

    section_title("Platform Capabilities", "📊")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("🌱", "38", "Disease Classes", "#4a7c59")
    with c2: stat_card("🌍", "14", "Crop Species", "#5c3d2e")
    with c3: stat_card("📷", "54K+", "Training Images", "#4a7c59")
    with c4: stat_card("⚡", "< 2s", "Avg. Inference", "#e67e22")

    section_title("How It Works", "🔄")
    steps = [
        ("📁", "Upload", "Go to Disease Recognition and upload a clear photo of a plant leaf."),
        ("🤖", "Analyze", "Our DenseNet121 model processes the image through 38 disease classifications."),
        ("📊", "Diagnose", "View top-3 predictions with confidence scores and severity rating."),
        ("💊", "Treat", "Receive chemical, cultural & organic treatment recommendations."),
    ]
    for col, (icon, title, desc) in zip(st.columns(4), steps):
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:14px;padding:1.4rem 1rem;text-align:center;
                border:1px solid #d4c5a0;box-shadow:0 2px 10px rgba(61,43,31,0.07);height:100%;">
              <div style="font-size:2rem;margin-bottom:0.7rem;">{icon}</div>
              <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;
                  color:#2d5a3d;margin-bottom:0.4rem;">{title}</div>
              <div style="font-size:0.8rem;color:#8b7355;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section_title("Crops Covered", "🌿")
    crops = [
        ("🍎","Apple"),("🫐","Blueberry"),("🍒","Cherry"),("🌽","Corn"),("🍇","Grape"),
        ("🍊","Orange"),("🍑","Peach"),("🫑","Bell Pepper"),("🥔","Potato"),("🫐","Raspberry"),
        ("🌱","Soybean"),("🥦","Squash"),("🍓","Strawberry"),("🍅","Tomato"),
    ]
    cols = st.columns(7)
    for idx, (icon, name) in enumerate(crops):
        with cols[idx % 7]:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:0.8rem 0.5rem;text-align:center;
                border:1px solid #d4c5a0;margin-bottom:0.5rem;box-shadow:0 1px 6px rgba(61,43,31,0.06);">
              <div style="font-size:1.5rem;">{icon}</div>
              <div style="font-size:0.7rem;font-weight:600;color:#5c3d2e;margin-top:0.3rem;
                  text-transform:uppercase;letter-spacing:0.04em;">{name}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  DISEASE RECOGNITION PAGE
# ════════════════════════════════════════════════════════════════════════════════

elif "Recognition" in app_mode:
    page_header("Disease Recognition",
                "DenseNet121 analyses your leaf image and returns top-3 diagnoses with confidence scores.", "🔬")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#f5faf5 0%,#eafaf1 100%);border:1px solid #a9dfbf;
        border-radius:12px;padding:0.9rem 1.3rem;margin-bottom:1.5rem;display:flex;gap:2rem;flex-wrap:wrap;">
      <div style="font-size:0.8rem;color:#2d5a3d;"><strong>📸 Photo Tips:</strong></div>
      <div style="font-size:0.8rem;color:#4a7c59;">✓ Single leaf, clear background</div>
      <div style="font-size:0.8rem;color:#4a7c59;">✓ Good natural lighting</div>
      <div style="font-size:0.8rem;color:#4a7c59;">✓ Symptoms visible &amp; in focus</div>
      <div style="font-size:0.8rem;color:#4a7c59;">✓ JPG / PNG format</div>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        section_title("Upload Leaf Image", "📁")
        test_image = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if test_image:
            st.image(test_image, caption="Uploaded leaf - ready for analysis", use_column_width=True)

    with right_col:
        section_title("Run Analysis", "⚡")
        if test_image is None:
            st.markdown("""
            <div style="background:white;border:2px dashed #d4c5a0;border-radius:14px;
                padding:3rem 1.5rem;text-align:center;color:#8b7355;">
              <div style="font-size:3rem;margin-bottom:0.8rem;opacity:0.4;">🌿</div>
              <div style="font-size:0.9rem;font-weight:500;">No image uploaded yet.</div>
              <div style="font-size:0.8rem;margin-top:0.4rem;opacity:0.7;">Upload a leaf image to begin.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#eafaf1;border:1px solid #a9dfbf;border-radius:12px;
                padding:1rem 1.3rem;margin-bottom:1rem;">
              <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#2d7a4a;margin-bottom:0.3rem;">Ready</div>
              <div style="font-size:0.85rem;color:#3d2b1f;">
                <strong>{test_image.name}</strong><br>
                <span style="color:#8b7355;">{round(test_image.size/1024,1)} KB · {test_image.type}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            predict_btn = st.button("🔬  Analyze & Diagnose", type="primary", use_container_width=True)
            st.markdown("""<div style="font-size:0.75rem;color:#8b7355;text-align:center;
                margin-top:0.5rem;font-style:italic;">Returns top-3 diagnoses with confidence scores</div>""",
                unsafe_allow_html=True)

            if predict_btn:
                with st.spinner("🌿  DenseNet processing leaf features..."):
                    top3 = model_prediction(test_image)
                st.session_state["top3"] = top3
                st.session_state["analyzed_image"] = test_image.name

    # Results
    if "top3" in st.session_state and test_image is not None:
        top3 = st.session_state["top3"]
        best = top3[0]
        best_class = best["class"]
        best_conf = best["confidence"] * 100

        st.markdown("<hr>", unsafe_allow_html=True)
        section_title("Prediction Results", "📊")

        colors = ["#27ae60", "#e67e22", "#95a5a6"]
        for i, pred in enumerate(top3):
            label = pred["class"].replace("___", "  ›  ").replace("_", " ")
            confidence_bar(label, pred["confidence"], colors[i], rank=i,
                           crop_icon=get_crop_icon(pred["class"]))

        if best_conf < 60:
            st.markdown(f"""
            <div style="background:#fef9e7;border:1px solid #f9e79f;border-left:5px solid #f39c12;
                border-radius:10px;padding:0.9rem 1.2rem;margin-top:0.8rem;font-size:0.85rem;color:#6e4c0f;">
              <strong>⚠️ Low Confidence ({best_conf:.1f}%)</strong><br>
              The model is uncertain about this diagnosis. Try uploading a clearer, better-lit image
              or consult a local agricultural extension officer.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        section_title("Treatment Recommendations", "💊")
        display_treatment_card(best_class, treatments_db)

        # Export
        st.markdown("<hr>", unsafe_allow_html=True)
        section_title("Export Diagnosis Report", "📄")

        info = treatments_db.get(best_class, {})
        lines = [
            "Agricultural Crop Disease Detection with Treatment Recommendation - Diagnosis Report",
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 52,
            f"Image     : {st.session_state.get('analyzed_image','N/A')}",
            f"Prediction: {best_class.replace('___',' - ').replace('_',' ')}",
            f"Confidence: {best_conf:.1f}%", "",
            "All Top-3 Predictions:",
        ]
        for i, p in enumerate(top3):
            lines.append(f"  {i+1}. {p['class'].replace('___',' - ').replace('_',' ')}: {p['confidence']*100:.1f}%")
        if info:
            lines += ["", f"Disease   : {info.get('disease_name','')}",
                      f"Crop      : {info.get('crop','')}",
                      f"Pathogen  : {info.get('pathogen','')}",
                      f"Severity  : {info.get('severity','')}",
                      "", "Description:", f"  {info.get('description','')}",
                      "", "Prevention:", f"  {info.get('prevention','')}",
                      "", "Chemical Treatments:"]
            for t in info.get("treatments",{}).get("chemical",[]): lines.append(f"  • {t}")
            lines += ["", "Cultural Practices:"]
            for t in info.get("treatments",{}).get("cultural",[]): lines.append(f"  • {t}")
            lines += ["", "Organic Options:"]
            for t in info.get("treatments",{}).get("organic",[]): lines.append(f"  • {t}")

        st.download_button(
            label="📄  Download Diagnosis Report (.txt)",
            data="\n".join(lines),
            file_name=f"phytosense_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )


# ════════════════════════════════════════════════════════════════════════════════
#  MODEL INSIGHTS PAGE
# ════════════════════════════════════════════════════════════════════════════════

elif "Insights" in app_mode:
    page_header("Model Insights", "DenseNet121 architecture, training performance & comparison against baseline models.", "📊")

    # ── DenseNet badge ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a3a2a 0%,#2d5a3d 60%,#3d2b1f 100%);
        border-radius:14px;padding:1.2rem 1.8rem;margin-bottom:1.5rem;
        display:flex;align-items:center;gap:1.2rem;box-shadow:0 4px 16px rgba(45,90,61,0.3);">
      <div style="font-size:2.5rem;">🏆</div>
      <div>
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:#e8d5a3;">
          Best Model: DenseNet121 (Transfer Learning)</div>
        <div style="font-size:0.8rem;color:rgba(232,213,163,0.75);margin-top:0.2rem;">
          Outperformed all baseline and fine-tuned models across accuracy, precision, recall &amp; F1-score on the PlantVillage test set.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section_title("DenseNet121 - Training Performance", "📈")
    m1, m2, m3, m4 = st.columns(4)
    with m1: stat_card("🎯", "98.73%", "Train Accuracy", "#4a7c59")
    with m2: stat_card("✅", "94.82%", "Best Val Accuracy", "#5c3d2e")
    with m3: stat_card("📉", "0.0510", "Final Train Loss", "#e67e22")
    with m4: stat_card("🏆", "94.27%", "Test Accuracy", "#4a7c59")

    st.markdown("<br>", unsafe_allow_html=True)
    m5, m6, m7, m8 = st.columns(4)
    with m5: stat_card("🧮", "91.62%", "Macro F1-Score", "#4a7c59")
    with m6: stat_card("⏱️", "10", "Epochs Run", "#5c3d2e")
    with m7: stat_card("🕐", "32.3 min", "Training Time", "#e67e22")
    with m8: stat_card("⚡", "< 2s", "Avg. Inference", "#4a7c59")

    # ── Model comparison table ───────────────────────────────────────────────────
    section_title("Model Comparison", "🔬")
    st.markdown("""
    <div style="background:white;border:1px solid #d4c5a0;border-radius:14px;overflow:hidden;
        box-shadow:0 2px 10px rgba(61,43,31,0.07);">
      <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
        <thead>
          <tr style="background:linear-gradient(135deg,#2d5a3d,#4a7c59);">
            <th style="padding:0.85rem 1.1rem;text-align:left;color:#e8d5a3;font-weight:700;letter-spacing:0.05em;">Model</th>
            <th style="padding:0.85rem 0.8rem;text-align:center;color:#e8d5a3;font-weight:700;">Type</th>
            <th style="padding:0.85rem 0.8rem;text-align:center;color:#e8d5a3;font-weight:700;">Test Acc.</th>
            <th style="padding:0.85rem 0.8rem;text-align:center;color:#e8d5a3;font-weight:700;">Macro F1</th>
            <th style="padding:0.85rem 0.8rem;text-align:center;color:#e8d5a3;font-weight:700;">Epochs</th>
            <th style="padding:0.85rem 0.8rem;text-align:center;color:#e8d5a3;font-weight:700;">Train Time</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background:linear-gradient(135deg,#eafaf1,#f5fdf7);border-bottom:2px solid #a9dfbf;">
            <td style="padding:0.9rem 1.1rem;font-weight:700;color:#2d5a3d;">
              🏆 DenseNet121 <span style="font-size:0.7rem;background:#27ae60;color:white;
                border-radius:20px;padding:0.1rem 0.5rem;margin-left:0.4rem;">BEST</span></td>
            <td style="padding:0.9rem 0.8rem;text-align:center;color:#5a4a3a;">Transfer Learning</td>
            <td style="padding:0.9rem 0.8rem;text-align:center;font-weight:700;color:#27ae60;font-family:'DM Mono',monospace;">94.27%</td>
            <td style="padding:0.9rem 0.8rem;text-align:center;font-weight:600;color:#27ae60;font-family:'DM Mono',monospace;">91.62%</td>
            <td style="padding:0.9rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">10</td>
            <td style="padding:0.9rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">32.3 min</td>
          </tr>
          <tr style="background:white;border-bottom:1px solid #ede8de;">
            <td style="padding:0.8rem 1.1rem;font-weight:600;color:#3d2b1f;">MobileNetV2</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#5a4a3a;">Transfer Learning</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#4a7c59;">86.45%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#4a7c59;">82.35%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">10</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">15.3 min</td>
          </tr>
          <tr style="background:#fafaf8;border-bottom:1px solid #ede8de;">
            <td style="padding:0.8rem 1.1rem;font-weight:600;color:#3d2b1f;">EfficientNetB0</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#5a4a3a;">Transfer Learning</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#5c3d2e;">84.23%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#5c3d2e;">82.69%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">10</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">17.3 min</td>
          </tr>
          <tr style="background:white;border-bottom:1px solid #ede8de;">
            <td style="padding:0.8rem 1.1rem;font-weight:600;color:#3d2b1f;">Baseline CNN</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#5a4a3a;">From Scratch</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#e67e22;">78.55%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#e67e22;">72.82%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">10</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">9.5 min</td>
          </tr>
          <tr style="background:#fafaf8;">
            <td style="padding:0.8rem 1.1rem;font-weight:600;color:#8b7355;">ResNet50</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;">Transfer Learning</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#c0392b;">71.15%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;font-family:'DM Mono',monospace;color:#c0392b;">69.82%</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">6</td>
            <td style="padding:0.8rem 0.8rem;text-align:center;color:#8b7355;font-family:'DM Mono',monospace;">27.9 min</td>
          </tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── DenseNet Architecture ────────────────────────────────────────────────────
    section_title("DenseNet121 Architecture", "🏗️")

    st.markdown("""
    <div style="background:#f5faf5;border:1px solid #a9dfbf;border-radius:12px;padding:1rem 1.4rem;
        margin-bottom:1.2rem;font-size:0.88rem;line-height:1.7;color:#3d2b1f;">
      <strong>DenseNet121</strong> (Densely Connected Convolutional Network) connects each layer to every
      subsequent layer in a feed-forward fashion. For L layers, there are <em>L(L+1)/2</em> direct connections.
      Each layer receives feature maps from all preceding layers as input and passes its own feature maps
      to all subsequent layers - enabling strong gradient flow, feature reuse, and substantially fewer parameters
      than conventional deep networks.
    </div>
    """, unsafe_allow_html=True)

    layers = [
        ("Input Layer",                "224 × 224 × 3",    "#4a7c59", "RGB image - normalized to [0, 1]"),
        ("Conv2D (64 filters) + BN",   "112 × 112 × 64",   "#5c8a6b", "7×7 kernel · stride 2 · ReLU"),
        ("MaxPooling2D",               "56 × 56 × 64",     "#5c3d2e", "3×3 pool · stride 2"),
        ("Dense Block 1 (6 layers)",   "56 × 56 × 256",    "#5c8a6b", "Growth rate k=32 · bottleneck"),
        ("Transition Layer 1",         "28 × 28 × 128",    "#7a4f6d", "BN + Conv 1×1 + AvgPool 2×2"),
        ("Dense Block 2 (12 layers)",  "28 × 28 × 512",    "#5c8a6b", "Growth rate k=32 · bottleneck"),
        ("Transition Layer 2",         "14 × 14 × 256",    "#7a4f6d", "BN + Conv 1×1 + AvgPool 2×2"),
        ("Dense Block 3 (24 layers)",  "14 × 14 × 1024",   "#5c8a6b", "Growth rate k=32 · bottleneck"),
        ("Transition Layer 3",         "7 × 7 × 512",      "#7a4f6d", "BN + Conv 1×1 + AvgPool 2×2"),
        ("Dense Block 4 (16 layers)",  "7 × 7 × 1024",     "#5c8a6b", "Growth rate k=32 · bottleneck"),
        ("GlobalAveragePooling2D",     "1024",              "#8b5e3c", "Spatial compression"),
        ("Dense (256) + Dropout 40%",  "256",               "#5c3d2e", "ReLU · regularization head"),
        ("Output Dense",               "38 classes",        "#c0392b", "Softmax - disease probabilities"),
    ]
    for name, shape, color, note in layers:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;background:white;border:1px solid #d4c5a0;
            border-left:5px solid {color};border-radius:10px;padding:0.7rem 1.1rem;margin-bottom:0.4rem;
            box-shadow:0 1px 5px rgba(61,43,31,0.05);">
          <div style="font-family:'DM Mono',monospace;font-size:0.82rem;font-weight:500;color:{color};min-width:240px;">{name}</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#8b7355;min-width:160px;">{shape}</div>
          <div style="font-size:0.8rem;color:#5a4a3a;">{note}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#faf6ee;border:1px solid #d4c5a0;border-radius:10px;
        padding:0.8rem 1.2rem;margin-top:0.8rem;font-size:0.82rem;color:#5a4a3a;display:flex;gap:2rem;flex-wrap:wrap;">
      <span>⚙️ <strong>Total Params:</strong> 7,037,504</span>
      <span>🚀 <strong>Optimizer:</strong> Adam (lr=0.001, ReduceLROnPlateau)</span>
      <span>📉 <strong>Loss:</strong> Categorical Cross-Entropy</span>
      <span>⏱️ <strong>Epochs:</strong> 10 (EarlyStopping patience=3)</span>
      <span>📦 <strong>Batch Size:</strong> 32</span>
      <span>🏋️ <strong>Pre-trained:</strong> ImageNet weights · top 30 layers unfrozen</span>
    </div>
    """, unsafe_allow_html=True)

    section_title("Dataset Statistics", "🗄️")
    sc1, sc2, sc3 = st.columns(3)
    for col, (name, count, pct, color, note) in zip([sc1, sc2, sc3], [
        ("Training Set",   "48,647", "70%", "#4a7c59", "38,013 original + 10,634 augmented"),
        ("Validation Set", "8,146",  "15%", "#5c3d2e", "Stratified split"),
        ("Test Set",       "8,146",  "15%", "#e67e22", "Held-out for final evaluation"),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:14px;border:1px solid #d4c5a0;
                border-top:4px solid {color};padding:1.3rem;box-shadow:0 2px 10px rgba(61,43,31,0.07);">
              <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:{color};margin-bottom:0.4rem;">{name}</div>
              <div style="font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:700;color:{color};line-height:1.1;">{count}</div>
              <div style="font-size:0.8rem;color:#8b7355;margin-top:0.2rem;">{pct} of total · {note}</div>
            </div>
            """, unsafe_allow_html=True)

    section_title("Data Augmentation Applied", "🔄")
    aug = [("🔄","Rotation","±40°"),("↔️","H-Flip","Random"),("↕️","V-Flip","Random"),
           ("🔍","Zoom","80–120%"),("↔️","Width Shift","±20%"),("↕️","Height Shift","±20%"),
           ("☀️","Brightness","80–120%"),("🖼️","Fill Mode","Nearest")]
    aug_cols = st.columns(4)
    for i, (icon, name, val) in enumerate(aug):
        with aug_cols[i % 4]:
            st.markdown(f"""
            <div style="background:#f5faf5;border:1px solid #a9dfbf;border-radius:10px;
                padding:0.8rem;text-align:center;margin-bottom:0.5rem;">
              <div style="font-size:1.3rem;">{icon}</div>
              <div style="font-size:0.78rem;font-weight:600;color:#2d5a3d;margin-top:0.2rem;">{name}</div>
              <div style="font-size:0.72rem;color:#8b7355;">{val}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  ABOUT PAGE
# ════════════════════════════════════════════════════════════════════════════════

elif "About" in app_mode:
    page_header("About This Project", "Agricultural crop disease detection powered by DenseNet121 transfer learning.", "📖")

    a1, a2 = st.columns([1.5, 1], gap="large")

    with a1:
        section_title("Project Overview", "🌾")
        st.markdown("""
        <div style="font-size:0.92rem;line-height:1.8;color:#3d2b1f;background:white;
            border-radius:14px;padding:1.5rem;border:1px solid #d4c5a0;box-shadow:0 2px 10px rgba(61,43,31,0.07);">
          This is an Agricultural crop disease detection system developed as a
          capstone project for <strong>CS 719 - Data Science Project</strong> at the
          <strong>University of Regina</strong>.<br><br>
          The system uses <strong>DenseNet121</strong> - a densely connected convolutional network
          pre-trained on ImageNet and fine-tuned on the <strong>PlantVillage Dataset</strong> (54,305 images,
          <strong>38 disease classes</strong>, 14 crop species) - achieving <strong>94.27% test accuracy</strong>
          and a <strong>91.62% macro F1-score</strong>,
          outperforming all baseline CNN and other transfer learning models evaluated in this study.<br><br>
          Beyond classification, the system provides a curated <strong>treatment recommendation
          database</strong> covering chemical, cultural, and organic management strategies
          for each identified disease - putting expert-level agricultural guidance directly
          into farmers' hands.
        </div>
        """, unsafe_allow_html=True)

        section_title("Treatment Database", "💊")
        st.markdown("""
        <div style="font-size:0.88rem;line-height:1.7;color:#3d2b1f;background:#f5faf5;
            border:1px solid #a9dfbf;border-radius:12px;padding:1.2rem 1.4rem;">
          Treatment recommendations were manually curated from agricultural extension resources
          and peer-reviewed literature. Each of the 38 disease classes includes:
          <ul style="margin:0.8rem 0 0 1.2rem;padding:0;">
            <li>Detailed symptom descriptions with visual identifiers</li>
            <li>Chemical treatment options with application timing guidance</li>
            <li>Cultural and preventive management practices</li>
            <li>Certified organic management alternatives</li>
            <li>Prevention tips tailored to each pathogen type</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        section_title("Project Details", "📋")
        for key, val in [
            ("Course", "CS 719 - Data Science Project"),
            ("Institution", "University of Regina"),
            ("Department", "Computer Science"),
            ("Term", "Winter 2026"),
            ("Student", "Viral Prajapati"),
            ("Student ID", "200499893"),
            ("Instructor", "Howard J. Hamilton"),
            ("Model", "DenseNet121 (ImageNet → PlantVillage fine-tune)"),
            ("Test Accuracy", "94.27%"),
            ("Macro F1-Score", "91.62%"),
            ("Dataset", "PlantVillage (Kaggle)"),
            ("Classes", "38 disease categories"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.55rem 0;
                border-bottom:1px solid #ede8de;font-size:0.84rem;">
              <span style="color:#8b7355;font-weight:600;text-transform:uppercase;
                  font-size:0.72rem;letter-spacing:0.05em;">{key}</span>
              <span style="color:#2c1810;font-weight:500;text-align:right;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        section_title("Tech Stack", "⚙️")
        techs = ["Python 3.9+","TensorFlow 2.15","Keras","Streamlit 1.28",
                 "NumPy","Pandas","scikit-learn","Matplotlib","Seaborn","Pillow"]
        pills = " ".join([f"""<span style="display:inline-block;background:#eafaf1;color:#2d5a3d;
            border:1px solid #a9dfbf;border-radius:20px;padding:0.2rem 0.8rem;font-size:0.78rem;
            font-weight:500;margin:0.15rem 0.2rem 0.15rem 0;">{t}</span>""" for t in techs])
        st.markdown(f"<div style='margin-top:0.5rem;'>{pills}</div>", unsafe_allow_html=True)

    section_title("Key References", "📚")
    for num, authors, title, journal in [
        ("[1]","Ngugi et al. (2024)","Revolutionizing crop disease detection with computational deep learning.","Environmental Monitoring and Assessment"),
        ("[2]","Goklani (2024)","Real-time plant disease detection using TensorFlow Lite and Flutter.","SSRN Electronic Journal"),
        ("[3]","Alghamdi & Turki (2023)","PDD-Net: Plant disease diagnoses using multilevel CNN features.","Agriculture"),
        ("[4]","Shafik et al. (2024)","Transfer learning-based plant disease classification for sustainable agriculture.","BMC Plant Biology"),
        ("[5]","Asani et al. (2023)","mPD-APP: A mobile-enabled plant disease diagnosis application using CNN.","Frontiers in Artificial Intelligence"),
    ]:
        st.markdown(f"""
        <div style="display:flex;gap:0.8rem;background:white;border:1px solid #d4c5a0;
            border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.5rem;font-size:0.82rem;">
          <span style="font-family:'DM Mono',monospace;color:#8b7355;flex-shrink:0;font-size:0.78rem;margin-top:0.1rem;">{num}</span>
          <div>
            <span style="font-weight:600;color:#2d5a3d;">{authors}</span>
            <span style="color:#3d2b1f;"> - {title} </span>
            <span style="font-style:italic;color:#8b7355;">{journal}.</span>
          </div>
        </div>
        """, unsafe_allow_html=True)