import streamlit as st
import torch
import torch.nn as nn
from feature_extractor import extract_features
import os

class Model(nn.Module):
    def __init__(self, in_features=57, h1=8, h2=9, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

@st.cache_resource
def load_model():
    model = Model()
    model_path = 'model.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found. Please train the model first.")
        st.stop()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

st.set_page_config(page_title="Spam Mail Classifier", page_icon="📧", layout="centered")

st.title("📧 Spam Mail Classifier")
st.markdown("Paste your email below to check if it's spam or not.")

email_input = st.text_area("Email Content", height=200, placeholder="Paste your email here...")

if st.button("Check Spam", type="primary"):
    if not email_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Analyzing email..."):
            features = extract_features(email_input)
            features_tensor = torch.tensor([features], dtype=torch.float32)
            
            model = load_model()
            with torch.no_grad():
                prediction = model(features_tensor).item()
            
            st.markdown("### Result")
            if prediction >= 0.5:
                st.error("🚨 **SPAM DETECTED!**")
                st.write(f"Confidence: {prediction:.2%}")
                st.progress(prediction)
                st.info("Tip: Do not click on suspicious links or provide personal information.")
            else:
                st.success("✅ **NOT SPAM**")
                st.write(f"Confidence: {1 - prediction:.2%}")
                st.progress(1 - prediction)
