🧠 SocioSentiment – Social Issue Sentiment & Emotion Analyzer
SocioSentiment is a Streamlit-based web application that analyzes
user-input text related to social issues and predicts both
sentiment (Positive, Neutral, Negative) and emotion (e.g., joy, anger, fear, sadness, surprise).

The system uses pre-trained and fine-tuned Transformer models from Hugging Face:
cardiffnlp/twitter-roberta-base-sentiment for sentiment analysis
j-hartmann/emotion-english-distilroberta-base for emotion classification.

⚙ How It Works
->User enters a sentence in the web interface.
->The text is cleaned (removing URLs, mentions, etc.).
->Each model tokenizes the input into numerical representations.
->The transformer models perform inference (no training during runtime).
->Softmax converts outputs into probability scores.

The system displays:
->Predicted sentiment & emotion
->Confidence scores
->Probability distribution charts
->Extracted keywords (TF-IDF based)

🚀 Key Features
1) Uses transfer learning
2) Real-time inference
3) Confidence visualization
4) Explainability via keyword extraction
5) Optimized with st.cache_resource for faster performance

My deployed app link =[https://sociosentiment-3kxktkrdd5bnlppxsv3zme.streamlit.app/]
