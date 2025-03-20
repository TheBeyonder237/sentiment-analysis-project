import streamlit as st
import torch
from transformers import pipeline

# Set device (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Custom Streamlit Styles
st.markdown("""
    <style>
        /* Center everything */
        .block-container {
            max-width: 650px;
            text-align: center;
        }
        
        /* Title styling */
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #FF4B4B;
            text-shadow: 2px 2px 10px rgba(255, 75, 75, 0.5);
        }

        /* Text input styling */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #FF4B4B;
            background-color: #1E1E1E;
            color: white;
            font-size: 16px;
        }

        /* Button styling */
        div.stButton > button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #E63E3E;
        }

        /* Result display */
        .result {
            font-size: 22px;
            font-weight: bold;
            color: #FF4B4B;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Load Model from Hugging Face
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"  
    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, device=0 if DEVICE == "cuda" else -1)
    return classifier
#
#@st.cache_resource
#def load_model():
    #model_path = "./models"  # Path to the trained model  directory
    #classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0 if DEVICE == "cuda" else -1)
    #return classifier
# Initialize model
classifier = load_model()

# Streamlit UI
st.markdown('<p class="title">Sentiment Analysis App 💬</p>', unsafe_allow_html=True)
st.write("Enter a review below and let AI analyze its sentiment! 🚀")

# User Input
text = st.text_area("Enter text:", "", height=150)

if st.button("Analyze"):
    if text.strip():
        result = classifier(text)[0]
        sentiment = result['label']
        confidence = result['score']

        # Display sentiment result
        st.markdown(f'<p class="result">Sentiment: {sentiment} ({confidence:.2%} confidence)</p>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text!")
