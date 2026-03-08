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
from langdetect import detect, LangDetectException
from googletrans import Translator
import emoji
import requests
import datetime

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # set this in your environment
# NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# --- 1. SETUP & MODEL LOADING ---
# We use st.cache_resource so the models are downloaded/loaded only once
@st.cache_resource
def load_models():
    print("Loading pre-trained models... please wait.")
    
    # --- ENGLISH SENTIMENT MODEL ---
    # Model: cardiffnlp/twitter-roberta-base-sentiment
    # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    sent_model_name_en = "cardiffnlp/twitter-roberta-base-sentiment"
    sent_tokenizer_en = AutoTokenizer.from_pretrained(sent_model_name_en)
    sent_model_en = AutoModelForSequenceClassification.from_pretrained(sent_model_name_en)

    # --- MULTILINGUAL SENTIMENT MODEL ---
    # Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
    # Supports: Arabic, English, French, German, Hindi, Italian, Portuguese, Spanish
    sent_model_name_multi = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sent_tokenizer_multi = AutoTokenizer.from_pretrained(sent_model_name_multi)
    sent_model_multi = AutoModelForSequenceClassification.from_pretrained(sent_model_name_multi)

    # --- EMOTION MODEL (English) ---
    # Model: j-hartmann/emotion-english-distilroberta-base
    # Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    emo_model_name = "j-hartmann/emotion-english-distilroberta-base"
    emo_tokenizer = AutoTokenizer.from_pretrained(emo_model_name)
    emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_name)
    
    return (sent_tokenizer_en, sent_model_en, 
            sent_tokenizer_multi, sent_model_multi,
            emo_tokenizer, emo_model)

# Load models immediately
(sent_tokenizer_en, sent_model_en, 
 sent_tokenizer_multi, sent_model_multi,
 emo_tokenizer, emo_model) = load_models()

# Initialize translator
translator = Translator()

# --- 2. HELPER FUNCTIONS ---

# Common abbreviations/slang dictionary
ABBREVIATIONS = {
    "u": "you", "ur": "your", "r": "are", "y": "why",
    "ppl": "people", "govt": "government", "gov": "government",
    "b4": "before", "2day": "today", "2morrow": "tomorrow",
    "msg": "message", "plz": "please", "pls": "please",
    "thx": "thanks", "ty": "thank you", "tnx": "thanks",
    "omg": "oh my god", "wtf": "what the hell", "idk": "i don't know",
    "imo": "in my opinion", "tbh": "to be honest", "ngl": "not gonna lie",
    "btw": "by the way", "fyi": "for your information",
    "asap": "as soon as possible", "dm": "direct message",
    "rt": "retweet", "fb": "facebook", "ig": "instagram",
    "bc": "because", "w/": "with", "w/o": "without",
    "abt": "about", "rn": "right now", "smh": "shaking my head",
    "lol": "laughing", "lmao": "laughing hard", "rofl": "laughing",
    "brb": "be right back", "gtg": "got to go",
    # Hindi romanized common words
    "kya": "what", "hai": "is", "nahi": "no", "haan": "yes",
    "accha": "good", "theek": "okay", "bahut": "very",
}

# Supported languages for multilingual sentiment model
SUPPORTED_LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'ar': 'Arabic', 
    'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'es': 'Spanish'
}

# Common Hinglish/Romanized Hindi words for detection
HINGLISH_WORDS = {
    'kya', 'hai', 'nahi', 'haan', 'acha', 'accha', 'theek', 'thik', 'bahut', 
    'bura', 'acha', 'kaise', 'kaisa', 'kyun', 'kyu', 'kab', 'kahan', 'yaar',
    'bhai', 'dost', 'tera', 'mera', 'tumhara', 'hamara', 'unka', 'iska',
    'woh', 'yeh', 'tum', 'hum', 'main', 'mujhe', 'tujhe', 'unhe', 'isko',
    'karo', 'karna', 'kiya', 'kar', 'raha', 'rahi', 'rahe', 'gaya', 'gayi',
    'hota', 'hoti', 'hote', 'hua', 'hui', 'hue', 'laga', 'lagi', 'lage',
    'sab', 'sabhi', 'kuch', 'kaafi', 'zyada', 'kam', 'bohot', 'bohat',
    'aur', 'lekin', 'par', 'magar', 'phir', 'abhi', 'tab', 'jab',
    'achha', 'bura', 'sahi', 'galat', 'pakka', 'sacchi', 'jhooth',
    'paisa', 'kaam', 'ghar', 'log', 'desh', 'sarkar', 'sarkaar',
    'pasand', 'nafrat', 'pyaar', 'gussa', 'dukh', 'sukh', 'khushi',
    'chinta', 'darr', 'dar', 'umeed', 'bharosa', 'vishwas',
    'samajh', 'soch', 'dekh', 'sun', 'bol', 'baat', 'jawab',
    'sawal', 'problem', 'dikkat', 'mushkil', 'aasan', 'mushkil',
    'sach', 'jhoot', 'matlab', 'yaani', 'waise', 'aise', 'wala', 'wali',
    'ne', 'ko', 'se', 'ka', 'ki', 'ke', 'mein', 'pe', 'tak'
}

def detect_language(text):
    """Detect language of input text."""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'en'  # Default to English if detection fails

def is_hinglish(text):
    """Detect if text is Hinglish (Hindi written in Roman script)."""
    words = text.lower().split()
    hindi_word_count = sum(1 for word in words if word.strip('.,!?') in HINGLISH_WORDS)
    # If more than 20% of words are Hindi, consider it Hinglish
    if len(words) > 0 and (hindi_word_count / len(words)) >= 0.2:
        return True
    return False

def translate_to_english(text, source_lang):
    """Translate text to English for emotion analysis."""
    try:
        if source_lang == 'en':
            return text, False
        result = translator.translate(text, src=source_lang, dest='en')
        return result.text, True
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Using original text.")
        return text, False

def convert_emojis(text):
    """Convert emojis to their text description."""
    return emoji.demojize(text, delimiters=(" ", " "))

def expand_abbreviations(text):
    """Expand common abbreviations and slang."""
    words = text.split()
    expanded = []
    for word in words:
        lower_word = word.lower().strip('.,!?')
        if lower_word in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[lower_word])
        else:
            expanded.append(word)
    return ' '.join(expanded)

def clean_text(text, expand_abbrev=True, handle_emoji=True):
    """Enhanced text cleaning with abbreviation expansion and emoji handling."""
    # Convert emojis to text
    if handle_emoji:
        text = convert_emojis(text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # Remove Twitter handles but keep the text after @
    text = re.sub(r'\@\w+', '', text)
    
    # Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Expand abbreviations
    if expand_abbrev:
        text = expand_abbreviations(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def highlight_confidence(val):
    """Color code the confidence table based on score."""
    if val >= 70:
        return 'background-color: #90ee90' # Light green
    elif val >= 40:
        return 'background-color: #f0e68c' # Khaki
    else:
        return 'background-color: #f08080' # Light coral
    
def fetch_news(query="social issues", language="en", page_size=5):
    """Fetch latest news articles from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("status") == "ok":
            return data["articles"]
        else:
            st.error(f"News API error: {data.get('message')}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {str(e)}")
        return []


# --- 3. STREAMLIT UI LAYOUT ---

st.title("SocioSentiment-Social Issue Analyzer 🧠")
st.markdown("Analyze the **Sentiment** and **Emotion** of text using pre-trained AI models.")
st.markdown("✨ **Now supports Hindi, Arabic, French, German, Italian, Portuguese & Spanish!**")

# --- Sidebar News Section ---
st.sidebar.subheader("📰 Latest News on Social Issues")

articles = fetch_news(query="climate change OR poverty OR healthcare", language="en", page_size=5)

# Input Layer
if articles:
    titles = [article['title'] for article in articles]
    selected_title = st.sidebar.selectbox("Select a headline to analyze:", titles)

user_input = st.text_area(
    "Enter a sentence about a social issue:",
    value=selected_title if selected_title else "",
    height=100,
    placeholder="e.g., I am worried about climate change, but hopeful for the future.\nया हिंदी में: मुझे जलवायु परिवर्तन की चिंता है।"
)
# Advanced options
with st.expander("⚙️ Advanced Options"):
    expand_abbrev = st.checkbox("Expand abbreviations (u→you, govt→government)", value=True)
    handle_emoji = st.checkbox("Convert emojis to text", value=True)

if st.button("Analyze Text"):
    if user_input:
        with st.spinner('Analyzing...'):
            # Preprocessing
            cleaned_input = clean_text(user_input, expand_abbrev=expand_abbrev, handle_emoji=handle_emoji)
            
            # --- LANGUAGE DETECTION ---
            detected_lang = detect_language(cleaned_input)
            
            # --- HINGLISH DETECTION ---
            # Check if text is Hinglish (Hindi in Roman script) even if detected as English
            hinglish_detected = is_hinglish(cleaned_input)
            
            if hinglish_detected:
                detected_lang = 'hi-latn'  # Hindi in Latin script
                lang_name = "Hinglish (Hindi in Roman script)"
                is_english = False
                is_supported = True
            else:
                lang_name = SUPPORTED_LANGUAGES.get(detected_lang, f"Other ({detected_lang})")
                is_english = detected_lang == 'en'
                is_supported = detected_lang in SUPPORTED_LANGUAGES
            
            # --- TRANSLATE IF HINGLISH or NON-ENGLISH ---
            text_for_analysis = cleaned_input
            was_translated_for_sentiment = False
            
            if hinglish_detected or (not is_english and is_supported):
                # Translate to English for better analysis
                try:
                    src_lang = 'hi' if hinglish_detected else detected_lang
                    result = translator.translate(cleaned_input, src=src_lang, dest='en')
                    text_for_analysis = result.text
                    was_translated_for_sentiment = True
                except Exception as e:
                    st.warning(f"Translation failed, using original text: {str(e)}")
                    text_for_analysis = cleaned_input
            
            # --- SENTIMENT PREDICTION ---
            if was_translated_for_sentiment or is_english:
                # Use English model on translated/English text for best accuracy
                sent_inputs = sent_tokenizer_en(text_for_analysis, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    sent_outputs = sent_model_en(**sent_inputs)
                model_used_sent = "English (cardiffnlp/twitter-roberta-base-sentiment)"
            else:
                # Use multilingual model for non-English that wasn't translated
                sent_inputs = sent_tokenizer_multi(cleaned_input, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    sent_outputs = sent_model_multi(**sent_inputs)
                model_used_sent = "Multilingual (cardiffnlp/twitter-xlm-roberta-base-sentiment)"
                
            sent_probs = softmax(sent_outputs.logits, dim=1)
            conf_sent, sent_idx = torch.max(sent_probs, dim=1)
            
            # Labels for sentiment models
            sentiment_labels = ["Negative", "Neutral", "Positive"]
            sentiment_pred = sentiment_labels[sent_idx.item()]
            
            # --- EMOTION PREDICTION ---
            # Use already translated text if available, otherwise translate
            if was_translated_for_sentiment:
                text_for_emotion = text_for_analysis
                was_translated = True
            elif not is_english:
                text_for_emotion, was_translated = translate_to_english(cleaned_input, detected_lang)
            else:
                text_for_emotion = cleaned_input
                was_translated = False
            
            # Tokenize for emotion model
            emo_inputs = emo_tokenizer(text_for_emotion, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                emo_outputs = emo_model(**emo_inputs)
                
            emo_probs = softmax(emo_outputs.logits, dim=1)
            conf_emo, emo_idx = torch.max(emo_probs, dim=1)
            
            # Get labels directly from the model config
            emotion_labels = list(emo_model.config.id2label.values())
            emotion_pred = emotion_labels[emo_idx.item()]

            # --- KEYWORD EXTRACTION ---
            # Use translated text for keywords if available
            keyword_text = text_for_emotion if was_translated else cleaned_input
            vectorizer = TfidfVectorizer(stop_words="english")
            try:
                vectorizer.fit_transform([keyword_text])
                keywords = vectorizer.get_feature_names_out()
            except:
                keywords = ["(Text too short)"]

            # --- DISPLAY RESULTS ---
            
            # Language Info
            st.divider()
            lang_col1, lang_col2 = st.columns(2)
            with lang_col1:
                st.info(f"🌐 **Detected Language:** {lang_name}")
            with lang_col2:
                if was_translated:
                    st.info(f"🔄 **Translated for emotion analysis**")
                elif not is_supported and not is_english:
                    st.warning(f"⚠️ Language may not be fully supported")
            
            # Metrics Row
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Sentiment", value=sentiment_pred, delta=f"{conf_sent.item()*100:.1f}% Conf.")
            with col2:
                st.metric(label="Predicted Emotion", value=emotion_pred.capitalize(), delta=f"{conf_emo.item()*100:.1f}% Conf.")

            st.write(f"**Keywords detected:** {', '.join(keywords[:5])}")
            
            # Show translated text if applicable
            if was_translated:
                with st.expander("📝 View Translated Text"):
                    st.write(f"**Original:** {cleaned_input}")
                    st.write(f"**Translated:** {text_for_emotion}")

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
                "Prediction": [sentiment_pred, emotion_pred.capitalize()],
                "Confidence (%)": [conf_sent.item()*100, conf_emo.item()*100],
                "Model Used": [model_used_sent, "j-hartmann/emotion-english-distilroberta-base"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.applymap(highlight_confidence, subset=["Confidence (%)"]))
            

            # Technical Info Expander
            with st.expander("🔧 Technical Details"):
                st.write(f"**Input Language:** {lang_name}")
                st.write(f"**Text was translated:** {'Yes' if was_translated else 'No'}")
                st.write(f"**Cleaned Input:** {cleaned_input}")
                st.write(f"**Abbreviations Expanded:** {'Yes' if expand_abbrev else 'No'}")
                st.write(f"**Emojis Converted:** {'Yes' if handle_emoji else 'No'}")


            # --- REPORT GENERATION & DOWNLOAD BUTTON ---
            # Format timestamp for readability
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            # Limit confidence to two decimals
            sent_conf = round(conf_sent.item()*100, 2)
            emo_conf = round(conf_emo.item()*100, 2)
            # Use semicolons for keywords
            keywords_str = "; ".join(keywords[:5])

            report_data = {
                "Input Text": [user_input],
                "Detected Language": [lang_name],
                "Sentiment Prediction": [sentiment_pred],
                "Sentiment Confidence (%)": [sent_conf],
                "Emotion Prediction": [emotion_pred.capitalize()],
                "Emotion Confidence (%)": [emo_conf],
                "Top Keywords": [keywords_str],
                "Was Translated": ["Yes" if was_translated else "No"],
                "Timestamp": [timestamp_str]
            }

            report_df = pd.DataFrame(report_data)

            csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Analysis Report (CSV)",
                data=csv,
                file_name="socio_sentiment_report.csv",
                mime="text/csv"
            )

            
            
    else:
        st.warning("Please enter some text to analyze.")



