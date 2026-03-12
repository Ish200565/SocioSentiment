import os
HISTORY_FILE = "history.csv"

def log_analysis(sentiment):
    import pandas as pd
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=["sentiment"])
        df.to_csv(HISTORY_FILE, index=False)
    df = pd.read_csv(HISTORY_FILE)
    df = pd.concat([df, pd.DataFrame([{"sentiment": sentiment.lower()}])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time

import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, LangDetectException
from googletrans import Translator
import emoji
import requests
import datetime

# Must be the first Streamlit command
st.set_page_config(page_title="SocioSentiment", layout="wide")

# Load NEWS_API_KEY from environment or Streamlit secrets
try:
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except (KeyError, FileNotFoundError):
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- 1. SETUP & MODEL LOADING ---
@st.cache_resource
def load_models():
    print("Loading pre-trained models... please wait.")
    
    sent_model_name_en = "cardiffnlp/twitter-roberta-base-sentiment"
    sent_tokenizer_en = AutoTokenizer.from_pretrained(sent_model_name_en)
    sent_model_en = AutoModelForSequenceClassification.from_pretrained(sent_model_name_en)

    sent_model_name_multi = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sent_tokenizer_multi = AutoTokenizer.from_pretrained(sent_model_name_multi)
    sent_model_multi = AutoModelForSequenceClassification.from_pretrained(sent_model_name_multi)

    emo_model_name = "j-hartmann/emotion-english-distilroberta-base"
    emo_tokenizer = AutoTokenizer.from_pretrained(emo_model_name)
    emo_model = AutoModelForSequenceClassification.from_pretrained(emo_model_name)
    
    return (sent_tokenizer_en, sent_model_en, 
            sent_tokenizer_multi, sent_model_multi,
            emo_tokenizer, emo_model)

(sent_tokenizer_en, sent_model_en, 
 sent_tokenizer_multi, sent_model_multi,
 emo_tokenizer, emo_model) = load_models()

translator = Translator()

# --- 2. HELPER FUNCTIONS ---
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
    "kya": "what", "hai": "is", "nahi": "no", "haan": "yes",
    "accha": "good", "theek": "okay", "bahut": "very",
}

SUPPORTED_LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'ar': 'Arabic', 
    'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'es': 'Spanish'
}

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
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def is_hinglish(text):
    words = text.lower().split()
    hindi_word_count = sum(1 for word in words if word.strip('.,!?') in HINGLISH_WORDS)
    if len(words) > 0 and (hindi_word_count / len(words)) >= 0.2:
        return True
    return False

def translate_to_english(text, source_lang):
    try:
        if source_lang == 'en':
            return text, False
        result = translator.translate(text, src=source_lang, dest='en')
        return result.text, True
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Using original text.")
        return text, False

def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def expand_abbreviations(text):
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
    if handle_emoji:
        text = convert_emojis(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    if expand_abbrev:
        text = expand_abbreviations(text)
    return ' '.join(text.split()).strip()

def highlight_confidence(val):
    if val >= 70:
        return 'background-color: #90ee90'
    elif val >= 40:
        return 'background-color: #f0e68c'
    else:
        return 'background-color: #f08080'
    
def fetch_news(query="social issues", language="en", page_size=5):
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
    except:
        pass
    return []

# --- CORE PROCESSING FUNCTION ---
def perform_analysis(text, expand_abbrev=True, handle_emoji=True):
    cleaned_input = clean_text(text, expand_abbrev=expand_abbrev, handle_emoji=handle_emoji)
    
    detected_lang = detect_language(cleaned_input)
    hinglish_detected = is_hinglish(cleaned_input)
    
    if hinglish_detected:
        detected_lang = 'hi-latn'
        lang_name = "Hinglish (Hindi in Roman script)"
        is_english = False
        is_supported = True
    else:
        lang_name = SUPPORTED_LANGUAGES.get(detected_lang, f"Other ({detected_lang})")
        is_english = detected_lang == 'en'
        is_supported = detected_lang in SUPPORTED_LANGUAGES
    
    text_for_analysis = cleaned_input
    was_translated_for_sentiment = False
    
    if hinglish_detected or (not is_english and is_supported):
        try:
            src_lang = 'hi' if hinglish_detected else detected_lang
            result = translator.translate(cleaned_input, src=src_lang, dest='en')
            text_for_analysis = result.text
            was_translated_for_sentiment = True
        except:
            text_for_analysis = cleaned_input
    
    # Sentiment
    if was_translated_for_sentiment or is_english:
        sent_inputs = sent_tokenizer_en(text_for_analysis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            sent_outputs = sent_model_en(**sent_inputs)
        model_used_sent = "English (cardiffnlp/twitter-roberta-base-sentiment)"
    else:
        sent_inputs = sent_tokenizer_multi(cleaned_input, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            sent_outputs = sent_model_multi(**sent_inputs)
        model_used_sent = "Multilingual (cardiffnlp/twitter-xlm-roberta-base-sentiment)"
        
    sent_probs = softmax(sent_outputs.logits, dim=1)
    conf_sent, sent_idx = torch.max(sent_probs, dim=1)
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment_pred = sentiment_labels[sent_idx.item()]
    
    # Emotion
    if was_translated_for_sentiment:
        text_for_emotion = text_for_analysis
        was_translated = True
    elif not is_english:
        text_for_emotion, was_translated = translate_to_english(cleaned_input, detected_lang)
    else:
        text_for_emotion = cleaned_input
        was_translated = False
    
    emo_inputs = emo_tokenizer(text_for_emotion, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        emo_outputs = emo_model(**emo_inputs)
        
    emo_probs = softmax(emo_outputs.logits, dim=1)
    conf_emo, emo_idx = torch.max(emo_probs, dim=1)
    emotion_labels = list(emo_model.config.id2label.values())
    emotion_pred = emotion_labels[emo_idx.item()]
    
    # Keywords
    keyword_text = text_for_emotion if was_translated else cleaned_input
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectorizer.fit_transform([keyword_text])
        keywords = vectorizer.get_feature_names_out()
    except:
        keywords = ["(Text too short)"]
        
    return {
        "text_original": text,
        "text_cleaned": cleaned_input,
        "text_translated": text_for_emotion if was_translated else "",
        "lang_detected": lang_name,
        "was_translated": was_translated,
        "is_supported": is_supported,
        "sentiment": sentiment_pred,
        "sentiment_conf": conf_sent.item() * 100,
        "sentiment_probs": sent_probs.detach().numpy()[0],
        "sentiment_labels": sentiment_labels,
        "model_used_sent": model_used_sent,
        "emotion": emotion_pred.capitalize(),
        "emotion_conf": conf_emo.item() * 100,
        "emotion_probs": emo_probs.detach().numpy()[0],
        "emotion_labels": emotion_labels,
        "keywords": keywords,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# --- 3. STREAMLIT UI LAYOUT ---

st.title("SocioSentiment: Social Issue Analyzer 🧠")
st.markdown("Analyze the **Sentiment** and **Emotion** of text.")

# --- Sidebar News Section ---
st.sidebar.subheader("📰 Latest News on Social Issues")
articles = fetch_news(query="climate change OR poverty OR healthcare", language="en", page_size=5)
selected_title = ""
if articles:
    titles = [article['title'] for article in articles]
    selected_title = st.sidebar.selectbox("Select a headline to analyze:", titles)

# UI Tabs
tab_single, tab_batch, tab_dash = st.tabs(["📝 Single Analysis", "📂 Batch Analysis", "📊 Dashboard"])

with tab_single:
    st.subheader("📝 Analyze Single Text")
    st.caption("Supported Languages: Hindi, Hinglish, Arabic, French, German, Italian, Portuguese, Spanish")

    user_input = st.text_area(
        "Enter a sentence about a social issue:",
        value=selected_title if selected_title else "",
        height=100,
        placeholder="e.g., I am worried about climate change.\nया हिंदी में: मुझे जलवायु परिवर्तन की चिंता है।"
    )

    with st.expander("⚙️ Advanced Options"):
        st.markdown("Customize preprocessing before analysis:")
        expand_abbrev_single = st.checkbox("Expand abbreviations (u→you, govt→government)", value=True, key="ea_single")
        handle_emoji_single = st.checkbox("Convert emojis to text", value=True, key="he_single")

    if st.button("Analyze Text", type="primary"):
        if user_input.strip():
            with st.spinner('Analyzing...'):
                result = perform_analysis(
                    user_input,
                    expand_abbrev=expand_abbrev_single,
                    handle_emoji=handle_emoji_single
                )
                log_analysis(result["sentiment"])
                
                # Language Info
                st.divider()
                lang_col1, lang_col2 = st.columns(2)
                with lang_col1:
                    st.info(f"🌐 **Detected Language:** {result['lang_detected']}")
                with lang_col2:
                    if result['was_translated']:
                        st.info(f"🔄 **Translated for emotion analysis**")
                    elif not result['is_supported']:
                        st.warning(f"⚠️ Language may not be fully supported")
                
                # Metrics Row
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Sentiment", value=result["sentiment"], delta=f"{result['sentiment_conf']:.1f}% Conf.")
                with col2:
                    st.metric(label="Predicted Emotion", value=result["emotion"], delta=f"{result['emotion_conf']:.1f}% Conf.")

                st.write(f"**Keywords detected:** {', '.join(result['keywords'][:5])}")
                
                if result['was_translated']:
                    with st.expander("📝 View Translated Text"):
                        st.write(f"**Original:** {result['text_cleaned']}")
                        st.write(f"**Translated:** {result['text_translated']}")

                # Visualizations
                st.divider()
                st.subheader("Confidence Distribution")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                sent_df = pd.DataFrame({'Label': result['sentiment_labels'], 'Score': result['sentiment_probs']})
                sns.barplot(x='Label', y='Score', data=sent_df, ax=ax1, palette="viridis", hue='Label', legend=False)
                ax1.set_title("Sentiment Probabilities")
                ax1.set_ylim(0, 1)

                emo_df = pd.DataFrame({'Label': result['emotion_labels'], 'Score': result['emotion_probs']})
                sns.barplot(x='Label', y='Score', data=emo_df, ax=ax2, palette="magma", hue='Label', legend=False)
                ax2.set_title("Emotion Probabilities")
                ax2.set_ylim(0, 1)
                ax2.tick_params(axis='x', rotation=45) 

                st.pyplot(fig)
                
                # Summary Table
                st.divider()
                st.subheader("Detailed Breakdown")
                summary_data = {
                    "Type": ["Sentiment", "Emotion"],
                    "Prediction": [result["sentiment"], result["emotion"]],
                    "Confidence (%)": [result['sentiment_conf'], result['emotion_conf']],
                    "Model Used": [result['model_used_sent'], "j-hartmann/emotion-english-distilroberta-base"]
                }
                st.dataframe(pd.DataFrame(summary_data).style.map(highlight_confidence, subset=["Confidence (%)"]))
                
                with st.expander("🔧 Technical Details"):
                    st.write(f"**Cleaned Input:** {result['text_cleaned']}")
                    st.write(f"**Abbreviations Expanded:** {'Yes' if expand_abbrev_single else 'No'}")

                # Download Single Report
                report_data = {
                    "Input Text": [result["text_original"]],
                    "Detected Language": [result["lang_detected"]],
                    "Sentiment": [result["sentiment"]],
                    "Sentiment Confidence (%)": [round(result["sentiment_conf"], 2)],
                    "Emotion": [result["emotion"]],
                    "Emotion Confidence (%)": [round(result["emotion_conf"], 2)],
                    "Top Keywords": ["; ".join(result["keywords"][:5])],
                    "Timestamp": [result["timestamp"]]
                }
                csv = pd.DataFrame(report_data).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Analysis Report (CSV)",
                    data=csv,
                    file_name="socio_sentiment_report.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please enter some text to analyze.")

with tab_batch:
    st.subheader("📂 Batch Upload & Analysis")
    st.write("Upload a CSV file (with a specific text column) or a TXT file (one entry per line) to analyze multiple texts efficiently.")
    
    uploaded_file = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # Load Data
        df_batch = None
        text_column = None
        
        if file_ext == "csv":
            df_batch = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df_batch.head())
            text_column = st.selectbox("Select the column containing the text to analyze:", df_batch.columns)
        elif file_ext == "txt":
            content = uploaded_file.getvalue().decode("utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            df_batch = pd.DataFrame({"Text": lines})
            text_column = "Text"
            st.write(f"Found {len(lines)} lines of text.")
            st.dataframe(df_batch.head())

        with st.expander("⚙️ Advanced Options"):
            expand_abbrev_batch = st.checkbox("Expand abbreviations", value=True, key="ea_batch")
            handle_emoji_batch = st.checkbox("Convert emojis to text", value=True, key="he_batch")

        if st.button("Start Batch Analysis", type="primary"):
            if df_batch is not None and text_column is not None:
                texts = df_batch[text_column].tolist()
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                for idx, text in enumerate(texts):
                    # Update status
                    status_text.text(f"Analyzing {idx+1}/{len(texts)}: {str(text)[:50]}...")
                    
                    # Ensure text is string
                    if pd.isna(text):
                        text = ""
                    text_str = str(text)
                    
                    if not text_str.strip():
                        continue
                        
                    try:
                        res = perform_analysis(text_str, expand_abbrev=expand_abbrev_batch, handle_emoji=handle_emoji_batch)
                        log_analysis(res["sentiment"])
                        
                        results.append({
                            "Original Text": res["text_original"],
                            "Language": res["lang_detected"],
                            "Sentiment": res["sentiment"],
                            "Sentiment Confidence": round(res["sentiment_conf"], 2),
                            "Emotion": res["emotion"],
                            "Emotion Confidence": round(res["emotion_conf"], 2),
                            "Keywords": ", ".join(res["keywords"][:3])
                        })
                    except Exception as e:
                        st.error(f"Error processing text on row {idx+1}: {e}")
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(texts))
                
                end_time = time.time()
                
                status_text.text(f"✅ Analysis complete in {round(end_time - start_time, 1)} seconds!")
                
                # Display Results
                if results:
                    results_df = pd.DataFrame(results)
                    st.success("Batch Analysis Completed!")
                    st.dataframe(results_df)

                    # Visualization for batch
                    st.subheader("Batch Results Summary")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.write("Sentiment Distribution")
                        st.bar_chart(results_df['Sentiment'].value_counts())
                    with col_b2:
                        st.write("Emotion Distribution")
                        st.bar_chart(results_df['Emotion'].value_counts())

                    # Merging with original data if CSV
                    if file_ext == "csv":
                        final_df = pd.concat([df_batch, results_df.drop(columns=["Original Text"])], axis=1)
                    else:
                        final_df = results_df

                    csv_b = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Batch Results (CSV)",
                        data=csv_b,
                        file_name=f"batch_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

with tab_dash:
    st.subheader("📊 Analysis Stats Dashboard")
    st.write("Historical data of all tracked sentiments.")
    
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        if not df_hist.empty:
            total_analyses = len(df_hist)
            sentiment_counts = df_hist["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
            
            st.metric("Total Overall Analyses Logged", total_analyses)
            
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            sentiment_df = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values})
            sns.barplot(x='Sentiment', y='Count', data=sentiment_df, ax=ax_hist, palette=["#2ecc71", "#95a5a6", "#e74c3c"], hue='Sentiment', legend=False)
            ax_hist.set_title("Historical Sentiment Distribution")
            ax_hist.set_ylabel("Count")
            st.pyplot(fig_hist)
            
            with st.expander("View Raw History Data"):
                st.dataframe(df_hist)
                
            if st.button("Clear History"):
                os.remove(HISTORY_FILE)
                st.success("History cleared. Reloading...")
                time.sleep(1)
                st.rerun()
        else:
            st.info("No analyses have been logged yet.")
    else:
        st.info("No analyses have been logged yet (history file not found).")
