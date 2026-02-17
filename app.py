# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st 
# from transformers import pipeline
# os.path.join("Text-Based-Emotion-Classification", "datasets", "go_emotions", "data")
# # Navigate directly into the dataset folder
# dataset_path = "Text-Based-Emotion-Classification/datasets/go_emotions/data"

# # Emotion dataset
# train_emotions = pd.read_csv("Text-Based-Emotion-Classification/datasets/go_emotions/data/train.tsv", sep="\t", header=None)
# dev_emotions   = pd.read_csv("Text-Based-Emotion-Classification/datasets/go_emotions/data/dev.tsv", sep="\t", header=None)
# test_emotions  = pd.read_csv("Text-Based-Emotion-Classification/datasets/go_emotions/data/test.tsv", sep="\t", header=None)

# train_emotions.columns = ["text", "label", "id"]
# dev_emotions.columns   = ["text", "label", "id"]
# test_emotions.columns  = ["text", "label", "id"]

# print("Emotion dataset sample:")
# print(train_emotions.head())

# # Sentiment dataset
# train_sentiments = pd.read_csv("Text-Based-Emotion-Classification/datasets/sentiment/us_airlines_tweets/train.tsv", sep="\t", header=None)
# dev_sentiments   = pd.read_csv("Text-Based-Emotion-Classification/datasets/sentiment/us_airlines_tweets/dev.tsv", sep="\t", header=None)
# test_sentiments  = pd.read_csv("Text-Based-Emotion-Classification/datasets/sentiment/us_airlines_tweets/test.tsv", sep="\t", header=None)


# # Assign column names
# train_sentiments.columns = ["text", "label","id"]
# dev_sentiments.columns   = ["text", "label","id"]
# test_sentiments.columns  = ["text", "label","id"]

# print("Sentiment dataset sample:")
# print(train_sentiments.head())

# # !pip install streamlit

# #1Input Layer
# # Input Layer
# user_input = st.text_area("Enter a sentence about a social issue:")

# # Run the classifier when user provides input
# if user_input:
#     sentiment_classifier = pipeline("text-classification", model=sentiment_model, tokenizer=tokenizer, return_all_scores=True)
#     emotion_classifier   = pipeline("text-classification", model=emotion_model, tokenizer=tokenizer, return_all_scores=True)

#     sentiment_results = sentiment_classifier(user_input)
#     emotion_results   = emotion_classifier(user_input)

#     st.subheader("Sentiment Predictions")
#     for res in sentiment_results[0]:
#         st.write(f"{res['label']}: {res['score']:.2f}")

#     st.subheader("Emotion Predictions")
#     for res in emotion_results[0]:
#         st.write(f"{res['label']}: {res['score']:.2f}")



# # 2 Preprocessing
# import re
# def clean_text(text):
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r'\@\w+|\#','', text)
#     return text.strip()

# #Feature Extraction
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# #Model Training into Sentiment(Positive,NEgative,Neutral) and Emotion Model(Happy,Sad,Angry,Fear,Surprise)
# sentiment_model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
# emotion_model   = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)

# # 5 Confidence Scoring
# import torch
# from torch.nn.functional import softmax

# inputs = tokenizer(user_input, return_tensors="pt")
# outputs = sentiment_model(**inputs)
# probs = softmax(outputs.logits, dim=1)
# confidence, prediction = torch.max(probs, dim=1)

# #6 Explainability Layer
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words="english")
# # Assign a sample value to user_input for demonstration purposes
# # In a live Streamlit app, this would come from user interaction
# if not user_input:
#     user_input = "This is a sample sentence about social issues and community involvement."
# X = vectorizer.fit_transform([user_input])
# keywords = vectorizer.get_feature_names_out()

# # 7 Output Layer -> Sentiment + Emotion + Confidence + Keywords

# # --- Sentiment prediction ---
# sentiment_outputs = sentiment_model(**inputs)
# sentiment_probs = softmax(sentiment_outputs.logits, dim=1)
# confidence_sentiment, sentiment_prediction_idx = torch.max(sentiment_probs, dim=1)

# sentiment_labels = ["Positive", "Negative", "Neutral"]
# sentiment_prediction = sentiment_labels[sentiment_prediction_idx.item()]

# # --- Emotion prediction ---
# emotion_outputs = emotion_model(**inputs)
# emotion_probs = softmax(emotion_outputs.logits, dim=1)
# confidence_emotion, emotion_prediction_idx = torch.max(emotion_probs, dim=1)

# emotion_labels = ["Happy", "Sad", "Angry", "Fear", "Surprise"]
# emotion_prediction = emotion_labels[emotion_prediction_idx.item()]

# # --- Display results in Streamlit ---
# st.write(f"Sentiment: {sentiment_prediction} (Confidence: {confidence_sentiment.item()*100:.2f}%)")
# st.write(f"Emotion: {emotion_prediction} (Confidence: {confidence_emotion.item()*100:.2f}%)")
# st.write(f"Keywords influencing prediction: {keywords[:5]}")

# # --- Plot sentiment confidence chart ---
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.bar(sentiment_labels, sentiment_probs.detach().numpy()[0])
# ax.set_title("Sentiment Confidence Distribution")
# st.pyplot(fig)

# # --- Plot emotion confidence chart ---
# fig2, ax2 = plt.subplots()
# ax2.bar(emotion_labels, emotion_probs.detach().numpy()[0])
# ax2.set_title("Emotion Confidence Distribution")
# st.pyplot(fig2)

# # --- Plot sentiment confidence chart ---
# import matplotlib.pyplot as plt

# sentiment_labels = ["Positive", "Negative", "Neutral"]
# fig1, ax1 = plt.subplots()
# ax1.bar(sentiment_labels, sentiment_probs.detach().numpy()[0])
# ax1.set_title("Sentiment Confidence Distribution")
# st.pyplot(fig1)

# # --- Plot emotion confidence chart ---
# emotion_labels = ["Happy", "Sad", "Angry", "Fear", "Surprise"]
# fig2, ax2 = plt.subplots()
# ax2.bar(emotion_labels, emotion_probs.detach().numpy()[0])
# ax2.set_title("Emotion Confidence Distribution")
# st.pyplot(fig2)

# # --- Combined Sentiment + Emotion Confidence Charts ---
# import matplotlib.pyplot as plt

# sentiment_labels = ["Positive", "Negative", "Neutral"]
# emotion_labels   = ["Happy", "Sad", "Angry", "Fear", "Surprise"]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Sentiment chart
# ax1.bar(sentiment_labels, sentiment_probs.detach().numpy()[0])
# ax1.set_title("Sentiment Confidence Distribution")
# ax1.set_ylabel("Probability")

# # Emotion chart
# ax2.bar(emotion_labels, emotion_probs.detach().numpy()[0])
# ax2.set_title("Emotion Confidence Distribution")
# ax2.set_ylabel("Probability")

# plt.tight_layout()
# st.pyplot(fig)

# # --- Summary Table with Color Coding ---
# import pandas as pd

# summary_data = {
#     "Prediction Type": ["Sentiment", "Emotion"],
#     "Label": [sentiment_prediction, emotion_prediction],
#     "Confidence (%)": [confidence_sentiment.item()*100, confidence_emotion.item()*100]
# }
# summary_df = pd.DataFrame(summary_data)

# def highlight_confidence(val):
#     if val >= 70:
#         color = 'background-color: lightgreen'
#     elif val >= 40:
#         color = 'background-color: khaki'
#     else:
#         color = 'background-color: lightcoral'
#     return color

# st.subheader("Prediction Summary")
# st.dataframe(summary_df.style.applymap(highlight_confidence, subset=["Confidence (%)"]))

# #What This Does
# '''- Two charts side by side: one for sentiment probabilities, one for emotion probabilities.
# - Summary table: neatly shows the predicted sentiment and emotion labels with their confidence scores.
# '''

# import matplotlib.pyplot as plt

# labels = ["Positive","Negative","Neutral"]
# plt.bar(labels, sentiment_probs.detach().numpy()[0])
# st.pyplot(plt)

# print("Working directory:", os.getcwd())

# # # Basic setup
# # # !pip install pandas numpy matplotlib seaborn scikit-learn transformers datasets
# # # 1. Clone the repository
# # # !git clone https://github.com/lightbooster/Text-Based-Emotion-Classification.git

# # # 2. Navigate into the datasets folder
# # import os
# # os.chdir("Text-Based-Emotion-Classification/datasets")

# # # Check available folders
# # print("Folders inside datasets:", os.listdir())



# # # Navigate into emotions folder
# # os.chdir("go_emotions/data")
# # # List files
# # print("Files inside go_emotions/data:", os.listdir())
























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