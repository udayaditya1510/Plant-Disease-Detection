# 🌿 Plant Disease Recognition System with AI Remedies

> *Early detection. Smart remedies. Healthier harvests.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-green?logo=streamlit)
![Gemini API](https://img.shields.io/badge/Gemini_API-Integrated-purple?logo=google)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Project Screenshots

## Screenshots
# Dashboard
Here's a view of the interface when detecting a plant disease:
![image alt](https://github.com/suraj-6/plant-disease-detection/blob/main/Screenshot%202025-06-14%20174626.png?raw=true)

picture of an diseased plant:
![image alt](https://github.com/suraj-6/plant-disease-detection/blob/main/Screenshot%202025-06-14%20174729.png?raw=true)

Another view showing a diseased plant recognition by predicting the disease:
![image alt](https://github.com/suraj-6/plant-disease-detection/blob/main/Screenshot%202025-06-14%20174712.png?raw=true)

> 💡 *Replace placeholder image paths above with actual screenshots of your app. Store images in `/screenshots/` folder.*

---

## 🌍 Overview

Plant diseases cause massive losses in global agriculture every year — reducing yields, increasing costs, and threatening food security. Traditional diagnosis methods are slow, subjective, and often inaccessible to small farmers.

This project introduces an **AI-powered plant disease detection system** that:
- ✅ Uses **Convolutional Neural Networks (CNN)** to classify 38+ plant diseases from leaf images.
- ✅ Integrates **Google Gemini API** to generate real-time, actionable treatment advice.
- ✅ Provides a **user-friendly web interface** built with Streamlit for farmers, researchers, and gardeners.

Empowering users with instant diagnosis + smart remedies = faster intervention → healthier crops → better harvests.

---

## 🧠 Technical Stack

| Component          | Technology Used                     |
|--------------------|--------------------------------------|
| **Frontend/UI**    | Streamlit                            |
| **Backend Model**  | TensorFlow/Keras (CNN)               |
| **AI Remedies**    | Google Gemini 1.5 Flash              |
| **Image Processing** | Pillow, NumPy                        |
| **Secrets Mgmt**   | python-dotenv                        |
| **Deployment**     | Streamlit Cloud / Local              |

---

## 🚀 Features

- 📷 Upload leaf image → Get instant disease prediction
- 🤖 AI-generated organic & chemical treatment plans
- 📥 Downloadable diagnosis report (.txt)
- 🎨 Modern, responsive UI with cards and animations
- 🔐 Secure API key handling via `.env`
- 🌱 Supports 14+ crops: Tomato, Apple, Potato, Grape, Pepper, etc.
- 🧪 Trained on 87,000+ labeled plant images

---

## 🗃️ Dataset

The model was trained on a curated dataset containing:

- **Total Images**: ~87,000 RGB images
- **Classes**: 38 disease categories + healthy variants
- **Crops Covered**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **Split**:
  - `train`: 70,295 images
  - `validation`: 17,572 images
  - `test`: 33 images (for final evaluation)

Dataset source: Publicly available plant disease datasets (e.g., PlantVillage), augmented offline.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9–3.11 (recommended)
- pip

### Steps

1. **Clone the repo**

```bash
git clone https://github.com/udayaditya1510/Plant-Disease-Detection.git
cd plant-disease-detection
Create virtual environment (optional but recommended)
Bash

python -m venv plant-env
plant-env\Scripts\activate  # Windows
# source plant-env/bin/activate  # Mac/Linux
Install dependencies
Create requirements.txt if not present:

txt

streamlit
tensorflow
numpy
pillow
google-generativeai
python-dotenv
Then install:

Bash

pip install -r requirements.txt
Set up API Key
Create a .env file in the root folder:

env

GOOGLE_API_KEY=your_gemini_api_key_here
🔑 Get your free API key: https://aistudio.google.com/app/apikey


run the ipynb files of train and test data accordingly
Place your model & assets
Ensure these files are in the project root:

trained_plant_disease_model.keras
imgplant1.webp (homepage banner)
logo.jpg (sidebar logo — optional)
Screenshots in /screenshots/ folder (optional)
Run the app
Bash

streamlit run main.py
🌐 Open http://localhost:8501 and start diagnosing
