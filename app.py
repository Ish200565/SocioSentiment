import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. SETUP & MODEL LOADING ---
# We use st.cache_resource so the models are downloaded/loaded only once
@st.cache_resource
def load_models():
    print("Loading pre-trained models... please wait.")
    
    # --- SENTIMENT MODEL ---
    # Model: cardiffnlp/twitter-roberta-base-sentiment
    # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    sent_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
    sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)

    # --- EMOTION MODEL ---
    # Model: j-hartmann/emotion-english-distilroberta-base
    # Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    emo_model_name = "j-hartmann/emotion-english-distilroberta-base"
    emo_tokenizer = AutoTokenizer.from_pretrained(emo_model_name)
    emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_name)
    
    return sent_tokenizer, sent_model, emo_tokenizer, emo_model

# Load models immediately
sent_tokenizer, sent_model, emo_tokenizer, emo_model = load_models()

# --- 2. HELPER FUNCTIONS ---

def clean_text(text):
    """Basic text cleaning to remove URLs and handles."""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    return text.strip()

def highlight_confidence(val):
    """Color code the confidence table based on score."""
    if val >= 70:
        return 'background-color: #90ee90' # Light green
    elif val >= 40:
        return 'background-color: #f0e68c' # Khaki
    else:
        return 'background-color: #f08080' # Light coral

# --- 3. STREAMLIT UI LAYOUT ---

st.title("Social Issue Analyzer 🧠")
st.markdown("Analyze the **Sentiment** and **Emotion** of text using pre-trained AI models.")

# Input Layer
user_input = st.text_area("Enter a sentence about a social issue:", height=100, placeholder="e.g., I am worried about climate change, but hopeful for the future.")

if st.button("Analyze Text"):
    if user_input:
        with st.spinner('Analyzing...'):
            # Preprocessing
            cleaned_input = clean_text(user_input)
            
            # --- SENTIMENT PREDICTION ---
            # Tokenize specifically for the sentiment model
            sent_inputs = sent_tokenizer(cleaned_input, return_tensors="pt")
            
            with torch.no_grad():
                sent_outputs = sent_model(**sent_inputs)
                
            sent_probs = softmax(sent_outputs.logits, dim=1)
            conf_sent, sent_idx = torch.max(sent_probs, dim=1)
            
            # Labels for cardiffnlp/twitter-roberta-base-sentiment
            sentiment_labels = ["Negative", "Neutral", "Positive"]
            sentiment_pred = sentiment_labels[sent_idx.item()]
            
            # --- EMOTION PREDICTION ---
            # Tokenize specifically for the emotion model
            emo_inputs = emo_tokenizer(cleaned_input, return_tensors="pt")
            
            with torch.no_grad():
                emo_outputs = emo_model(**emo_inputs)
                
            emo_probs = softmax(emo_outputs.logits, dim=1)
            conf_emo, emo_idx = torch.max(emo_probs, dim=1)
            
            # Get labels directly from the model config (ensures accuracy)
            emotion_labels = list(emo_model.config.id2label.values())
            emotion_pred = emotion_labels[emo_idx.item()]

            # --- KEYWORD EXTRACTION ---
            vectorizer = TfidfVectorizer(stop_words="english")
            try:
                vectorizer.fit_transform([cleaned_input])
                keywords = vectorizer.get_feature_names_out()
            except:
                keywords = ["(Text too short)"]

            # --- DISPLAY RESULTS ---
            
            # Metrics Row
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Sentiment", value=sentiment_pred, delta=f"{conf_sent.item()*100:.1f}% Conf.")
            with col2:
                st.metric(label="Predicted Emotion", value=emotion_pred, delta=f"{conf_emo.item()*100:.1f}% Conf.")

            st.write(f"**Keywords detected:** {', '.join(keywords[:5])}")

            # --- VISUALIZATION ---
            st.divider()
            st.subheader("Confidence Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Sentiment Chart
            sent_df = pd.DataFrame({'Label': sentiment_labels, 'Score': sent_probs.detach().numpy()[0]})
            sns.barplot(x='Label', y='Score', data=sent_df, ax=ax1, palette="viridis")
            ax1.set_title("Sentiment Probabilities")
            ax1.set_ylim(0, 1)

            # Emotion Chart
            emo_df = pd.DataFrame({'Label': emotion_labels, 'Score': emo_probs.detach().numpy()[0]})
            sns.barplot(x='Label', y='Score', data=emo_df, ax=ax2, palette="magma")
            ax2.set_title("Emotion Probabilities")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45) # Rotate emotion labels for readability

            st.pyplot(fig)
            
            # --- SUMMARY TABLE ---
            st.divider()
            st.subheader("Detailed Breakdown")
            summary_data = {
                "Type": ["Sentiment", "Emotion"],
                "Prediction": [sentiment_pred, emotion_pred],
                "Confidence (%)": [conf_sent.item()*100, conf_emo.item()*100]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.applymap(highlight_confidence, subset=["Confidence (%)"]))
            
    else:
        st.warning("Please enter some text to analyze.")
