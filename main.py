import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ===========================
# Load Environment Variables & Configure
# ===========================
def configure_genai_from_env():
    """Load `.env` (override) and configure the google.generativeai client.
    Returns (success: bool, message_or_model).
    """
    # Explicitly load the .env file from the project so we know which file is used
    load_dotenv(r"c:\Users\admin\Downloads\proj\.env", override=True)
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return False, "No API key found in .env"

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Quick test to verify the model works
        test = model.generate_content("Test connection")
        if not test or not test.text:
            return False, "Could not verify model (empty response)"
        return True, model
    except Exception as e:
        err = str(e).lower()
        if "leak" in err or "reported as leaked" in err or "403" in err or "blocked" in err:
            return False, "leaked"
        return False, str(e)

# Global model instance
model_gemini = None

# Initial configuration attempt
ok, result = configure_genai_from_env()
if ok:
    model_gemini = result
else:
    if result == "leaked":
        st.error(
            "❌ The API key used by this app has been flagged as leaked or blocked.\n"
            "Please revoke the compromised key in Google Cloud Console and create a new API key.\n\n"
            "Steps:\n1) Visit https://console.cloud.google.com/apis/credentials and delete the compromised key.\n"
            "2) Create and restrict a new API key (API & application restrictions).\n"
            "3) Update your `.env` file with the new key and use the 'Reload .env' button in the sidebar."
        )
    else:
        st.warning(f"⚠️ Could not configure Gemini client: {result}")

# ===========================
# Tensorflow Model Prediction
# ===========================
def model_prediction(test_image):
    """Predict plant disease from image using trained model.
    Args:
        test_image: UploadedFile object containing the image
    Returns:
        int: Index of predicted class, or None on error
    """
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) / 255.0  # Normalize pixel values
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# ===========================
# Gemini Remedy Generator
# ===========================
def get_disease_remedy(disease_name):
    """Generate remedy recommendations using Gemini AI.
    Args:
        disease_name: Name of the diagnosed plant disease
    Returns:
        str: AI-generated remedy text or error message
    """
    # Clean disease name for better prompting
    clean_disease = disease_name.replace('___', ' ').replace('_', ' ').strip()
    
    prompt = f"""
    You are an expert agricultural assistant.
    The plant has been diagnosed with: {clean_disease}.
    Provide:
    1. A brief description of the disease.
    2. Organic and chemical treatment options.
    3. Preventive measures.
    4. Recommended pesticides or fungicides (if any).
    Keep the response concise, well-structured, farmer-friendly, and actionable.
    """

    if not model_gemini:
        return "❌ Gemini API not configured. Please check API key and settings."

    try:
        response = model_gemini.generate_content(prompt)
        if not response or not response.text:
            return "❌ No response from Gemini API. Please try again."
        return response.text.strip()
    except Exception as e:
        err = str(e).lower()
        if "leak" in err or "reported as leaked" in err or "403" in err or "blocked" in err:
            return (
                "❌ The API key used by this app has been flagged as leaked or blocked.\n"
                "Please revoke the compromised key in Google Cloud Console and create a new API key.\n"
                "Steps:\n1) Visit https://console.cloud.google.com/apis/credentials and delete the compromised key.\n"
                "2) Create and restrict a new API key (API & application restrictions).\n"
                "3) Update your `.env` file and use the 'Reload .env' button in the sidebar."
            )
        return f"❌ Could not generate remedy: {str(e)}. Check API key or try again later."

# ===========================
# Custom CSS Styling
# ===========================
st.set_page_config(page_title="🌿 Plant Doctor AI", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f8f0;
    }
    h1, h2, h3 {
        color: #2e7d32;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        transform: scale(1.05);
    }
    .stImage {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .remedy-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
        line-height: 1.6;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #66bb6a, #43a047);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# Sidebar Navigation
# ===========================
st.sidebar.title("🌿 Dashboard")

# Use relative path for image
try:
    st.sidebar.image("WhatsApp Image 2025-09-18 at 15.28.12_66d95583.jpg", width=60)
except:
    st.sidebar.warning("⚠️ Logo image not found")

app_mode = st.sidebar.selectbox("🧭 Navigate", ["Home", "About", "Disease Recognition"])

# Admin tools for API key management
with st.sidebar.expander("🔑 API Settings", expanded=False):
    if st.button("🔄 Reload API Key"):
        ok2, res2 = configure_genai_from_env()
        if ok2:
            model_gemini = res2
            st.success("✅ API key reloaded successfully")
            st.experimental_rerun()
        else:
            if res2 == "leaked":
                st.error("❌ The loaded key has been flagged as leaked/blocked.")
            else:
                st.error(f"⚠️ Error loading key: {res2}")
    
    show_key = st.checkbox("👁️ Show masked key", value=False)
    if show_key:
        key = os.getenv("GOOGLE_API_KEY")
        if key:
            # Show only first/last 4 chars
            masked = f"{key[:4]}...{key[-4:]}"
            st.info(f"Current key (masked): {masked}")
        else:
            st.warning("No API key loaded")

st.sidebar.markdown("---")
st.sidebar.info("🛠️ Built with TensorFlow + Gemini AI")

# ===========================
# Home Page
# ===========================
if app_mode == "Home":
    st.title("🌱 Welcome to Plant Doctor AI")
    st.image("imgplant1.webp", use_container_width=True, caption="Smart Plant Disease Detection System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🌿 Protect Your Crops with AI-Powered Diagnosis

        Our intelligent system helps farmers, gardeners, and researchers quickly identify plant diseases from leaf images — and provides **real-time, AI-generated remedies** powered by **Google Gemini**.

        Whether you're managing a backyard garden or a commercial farm, early detection can save your harvest!

        #### ✅ How It Works
        1. **Upload** a clear photo of a diseased leaf.
        2. **AI analyzes** it using deep learning.
        3. **Get diagnosis + smart remedies** in seconds.

        #### 🚀 Why Choose Us?
        - ✔️ Trained on 87,000+ real plant images
        - ✔️ 38+ disease categories supported
        - ✔️ Organic & chemical treatment advice
        - ✔️ Simple, mobile-friendly interface
        """)

    with col2:
        st.info("📌 Supported Plants:")
        plants = [
            "🍎 Apple", "🍇 Grape", "🍊 Orange",
            "🍒 Cherry", "🌽 Corn", "🍑 Peach",
            "🌶️ Pepper", "🥔 Potato", "🍓 Strawberry",
            "🍅 Tomato", "🫐 Blueberry", "🍂 Soybean"
        ]
        for plant in plants:
            st.markdown(f"- {plant}")

    st.markdown("---")
    st.subheader("🚀 Ready to Diagnose?")
    st.write("Go to ➤ **Disease Recognition** in the sidebar to upload an image and start!")

    # Call-to-action button that switches to Disease Recognition
    if st.button("👉 Start Diagnosis Now", use_container_width=True):
        st.session_state.app_mode = "Disease Recognition"
        st.experimental_rerun()

# ===========================
# About Page
# ===========================
elif app_mode == "About":
    st.title("📖 About Plant Doctor AI")

    st.markdown("""
    <div style='background: linear-gradient(to right, #e8f5e8, #c8e6c9); padding: 20px; border-radius: 15px;'>
        <h3>🌾 Empowering Farmers with AI</h3>
        <p>Plant Doctor AI is designed to bring cutting-edge machine learning to agriculture — helping detect diseases early and recommend proven remedies.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("📸 Total Images", "87,000+")
    col2.metric("🦠 Disease Classes", "38")
    col3.metric("🍃 Healthy Variants", "10+")

    st.markdown("""
    #### 🧠 Technical Details
    - **Model Architecture**: CNN-based classifier (TensorFlow/Keras)
    - **Input Size**: 128×128 pixels
    - **Training Split**: 80% train, 20% validation
    - **Test Set**: 33 curated images for final evaluation

    #### 🤖 AI Remedy Engine
    Powered by **Google Gemini**, we generate:
    - Disease descriptions in simple language
    - Step-by-step organic & chemical treatments
    - Prevention tips tailored to crop type
    - Pesticide/fungicide recommendations (with safety notes)

    > ⚠️ *Note: AI advice is for informational purposes. Always consult a local agricultural expert before applying chemicals.*

    #### 👥 Developed By
    _Your Name / Team Name_  
    Committed to sustainable farming and food security through technology.
    """)

    st.image("https://img.icons8.com/external-vitaliy-gorbachev-lineal-color-vitaly-gorbachev/60/000000/external-farmer-business-vitaliy-gorbachev-lineal-color-vitaly-gorbachev.png", width=80)
    st.caption("Supporting farmers worldwide with smarter tools.")

# ===========================
# Disease Recognition Page
# ===========================
elif app_mode == "Disease Recognition":
    st.title("🩺 Plant Disease Diagnosis & Remedies")

    st.markdown("""
    <div style='background: #fff8e1; padding: 15px; border-radius: 10px; border-left: 4px solid #ffc107;'>
        📸 Upload a clear, close-up image of a plant leaf to begin diagnosis.
    </div>
    """, unsafe_allow_html=True)

    test_image = st.file_uploader("📤 Upload Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if test_image is not None:
        col1, col2 = st.columns([1, 2])

        try:
            with col1:
                st.image(test_image, caption="📸 Uploaded Image", use_container_width=True, output_format="PNG")
        except Exception as e:
            st.error(f"❌ Error displaying image: {str(e)}")
            st.stop()

        with col2:
            st.markdown("### 🧪 Ready to Analyze?")
            st.write("Click below to run AI diagnosis and get instant remedies.")

            if st.button("🔍 Analyze Leaf & Get Remedies", use_container_width=True):
                with st.spinner("🧠 AI is analyzing your plant..."):
                    result_index = model_prediction(test_image)
                    
                    if result_index is None:
                        st.error("❌ Error during image analysis. Please try again with a different image.")
                        st.stop()

                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                predicted_disease = class_name[result_index]
                clean_name = predicted_disease.replace('___', ' - ').replace('_', ' ').title()

                # Display Diagnosis Card
                st.markdown(f"""
                <div class="diagnosis-card">
                    <h3>✅ Diagnosis Result</h3>
                    <h4>{clean_name}</h4>
                    <p><em>Confidence: High (based on trained model)</em></p>
                </div>
                """, unsafe_allow_html=True)

                # Fetch Remedy
                with st.spinner("🌿 Generating personalized care plan..."):
                    remedy_text = get_disease_remedy(predicted_disease)

                # Handle potential None or empty response
                if not remedy_text:
                    st.error("❌ Could not generate remedy. Please check API settings and try again.")
                    st.stop()

                # PRE-PROCESS outside f-string to avoid SyntaxError
                remedy_html = remedy_text.replace('\n', '<br>')

                # Display Remedy Box
                st.subheader("💡 AI-Generated Care Plan")
                st.markdown(f"""
                <div class="remedy-box">
                    {remedy_html}
                </div>
                """, unsafe_allow_html=True)

                st.caption("ℹ️ *Powered by Google Gemini AI — always verify with local experts before treatment.*")

                # Shareable summary with timestamp
                summary_text = f"""Plant Disease Diagnosis Report
Generated: {st.session_state.get('diagnosis_time', '')}

DIAGNOSIS
---------
{clean_name}

TREATMENT PLAN
-------------
{remedy_text}

Note: This is an AI-generated report. Please consult with agricultural experts before applying any treatments.
"""
                try:
                    st.download_button(
                        "📥 Download Diagnosis Summary",
                        data=summary_text,
                        file_name="plant_diagnosis.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Error creating download: {str(e)}")

    else:
        st.info("👈 Please upload a leaf image to begin analysis.")
        st.image("https://img.icons8.com/clouds/100/000000/leaf.png", width=120)