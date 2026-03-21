import streamlit as st
import time
from PIL import Image
import numpy as np
import json
import tensorflow as tf


# Configuring the Page
st.set_page_config(
    page_title="Oxford 102 Benchmark", 
    page_icon="🌸",
    layout="centered"
)

# Header and Description
st.markdown(
    """
    <h1 style='text-align: center;'>🌸 Oxford 102 Flower Classification</h1>
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
        <p style='line-height: 1.6; text-align: justify; color: #31333F; margin: 0;'>
            This web app showcases the application of Deep Neural Networks in flower classification. By leveraging Transfer Learning
            on the <strong>Oxford 102 Flower Dataset</strong>, the system overcomes high inter-class similarity to recognize unique 
            flower species. Use the sidebar to switch between architectures and compare their performance.
        </p>
    </div>
    """, unsafe_allow_html=True
)


# Model Selection Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.markdown("---")
    
    model_choice = st.radio(
        "**Select Architecture:**",
        ("ResNet50 (High Accuracy)", "MobileNetV2 (High Efficiency)"),
        help="ResNet50 focuses on precision, while MobileNetV2 is optimized for speed and mobile devices."
    )
    
    st.markdown("<br>", unsafe_allow_html=True) # Extra vertical spacing
    st.divider()
    
    # Model Specs Section
    st.subheader("📊 Model Specs")
    if "ResNet50" in model_choice:
        st.info("**Type:** Residual Network\n\n**Best for:** Server-side high-precision tasks.")
    else:
        st.success("**Type:** Inverted Residuals\n\n**Best for:** Real-time edge device inference.")
    
    st.markdown("---")
    st.caption("Developed for CDS6354 Machine Learning Project")


# Asset Loading
@st.cache_resource
def load_assets(choice):
    # Only show the progress bar during the initial load
    progress_bar = st.progress(0, text="Initializing Model Weights...")
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    if "ResNet50" in choice:
        model = tf.keras.models.load_model('models/resnet50_augmented_finetune_best.keras')
    else:
        model = tf.keras.models.load_model('models/mobilenetv2_baseline_finetune_best.keras')
        
    # Applying the alphabetical sorting logic for accurate labels
    with open('data/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        # Use string-based alphabetical sorting to match Keras internal indexing
        # This ensures index 0 matches folder '1', index 1 matches folder '10', etc.
        sorted_keys = sorted(cat_to_name.keys(), key=str) 
        labels = [cat_to_name[k] for k in sorted_keys]
        
    progress_bar.empty()
    return model, labels


model, labels = load_assets(model_choice)


# Selection of Inference
uploaded_file = st.file_uploader("Upload a Flower Image...", type=["jpg", "png"])

if uploaded_file and model is not None:

    image = Image.open(uploaded_file).convert("RGB")
        
    st.image(image, caption=f'Analyzing with {model_choice}', use_container_width=True)
    
    with st.status("Analyzing Image Features...", expanded=True) as status:
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        preds = model.predict(img_array)[0]
        top_indices = np.argsort(preds)[-5:][::-1]
        status.update(label="Classification Complete!", state="complete")
    
    st.subheader("🔬 Identification Results")
    for idx in top_indices:
        idx_int = idx.item()
        label_name = labels[idx_int]
        confidence = float(preds[idx_int])
        
        # Clean UI Layout
        # Creates two side-by-side containers where the first column takes up two-fifths (40%) of the available width, 
        # and the second column takes up the remaining three-fifths (60%)
        cols = st.columns([2, 3])
        with cols[0]:
            st.write(f"**{label_name.title()}**")
        with cols[1]:
            st.progress(confidence)
            st.caption(f"Confidence: {confidence*100:.2f}%")