SocioSentiment – Social Issue Sentiment & Emotion Analyzer
SocioSentiment is a Streamlit-based web application that analyzes user-input text related to social issues and predicts both sentiment (Positive, Neutral, Negative) and emotion (e.g., joy, anger, fear, sadness, surprise).
The system leverages pre-trained and fine-tuned Transformer models from Hugging Face:
- cardiffnlp/twitter-roberta-base-sentiment → Sentiment Analysis
- j-hartmann/emotion-english-distilroberta-base → Emotion Classification

⚙ How It Works
- User enters a sentence in the web interface.
- Text is cleaned (removing URLs, mentions, etc.).
- Each model tokenizes the i
