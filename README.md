🧠 SocioSentiment – Social Issue Sentiment & Emotion Analyzer
SocioSentiment is a Streamlit-based web application that analyzes user-input text related to social issues and predicts both sentiment (Positive, Neutral, Negative) and emotion (e.g., joy, anger, fear, sadness, surprise).
The system leverages pre-trained and fine-tuned Transformer models from Hugging Face:
- cardiffnlp/twitter-roberta-base-sentiment → Sentiment Analysis
- j-hartmann/emotion-english-distilroberta-base → Emotion Classification

⚙ How It Works
- User enters a sentence in the web interface.
- Text is cleaned (removing URLs, mentions, etc.).
- Each model tokenizes the input into numerical representations.
- Transformer models perform inference (no training during runtime).
- Softmax converts outputs into probability scores.
The system displays:
- Predicted sentiment & emotion
- Confidence scores
- Probability distribution charts
- Extracted keywords (TF-IDF based)

🚀 Key Features
- Transfer learning with Hugging Face models
- Real-time inference
- Confidence visualization
- Explainability via keyword extraction
- Optimized with st.cache_resource for faster performance

🔮 Upcoming Improvements
- Stats Board → Aggregate insights across multiple inputs for trend analysis
- Real-Time Issue Fetching → Integration with NewsAPI to analyze current social issues dynamically
- Downloadable CSV Reports → Export predictions and insights for offline use
- Emoji Detection → Recognize and interpret emojis as part of sentiment/emotion analysis
- Multilingual Support → Language detection for German, Portuguese, French, Hinglish
- Automatic Translation → If another language is detected, the text is translated to English for analysis

🌐 SocioSentiment App Deployment
This is a temporary deployed link: https://sociosentiment-3kxktkrdd5bnlppxsv3zme.streamlit.app/

