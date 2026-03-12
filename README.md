🧠 SocioSentiment – Social Issue Sentiment & Emotion Analyzer

SocioSentiment is a **production-ready Streamlit web application** that analyzes text about social issues and predicts **sentiment** (Positive, Neutral, Negative) and **emotion** (joy, anger, fear, sadness, surprise, etc.). It supports **multiple languages** with automatic translation and real-time news integration.

## 🎯 Core Capabilities

The system uses state-of-the-art Transformer models from Hugging Face:
- **cardiffnlp/twitter-roberta-base-sentiment** → English sentiment analysis
- **cardiffnlp/twitter-xlm-roberta-base-sentiment** → Multilingual sentiment analysis  
- **j-hartmann/emotion-english-distilroberta-base** → Emotion classification

### 🌐 Supported Languages
English, Hindi, Hinglish (Roman Urdu), Arabic, French, German, Italian, Portuguese, Spanish

## 🚀 Key Features

### 📝 Single Text Analysis
- Analyze individual texts with detailed predictions
- Real-time sentiment & emotion detection with confidence scores
- Interactive probability distribution charts
- Automatic keyword extraction (TF-IDF)
- Optional text preprocessing:
  - Emoji conversion to text
  - Abbreviation expansion (u→you, govt→government, etc.)
  - URL & mention removal
  - Hinglish detection

### 📂 Batch Processing
- Upload CSV or TXT files for bulk analysis
- Process multiple texts efficiently
- Download comprehensive reports with predictions
- Real-time progress tracking
- Aggregated visualization (sentiment & emotion distribution)

### 📊 Analytics Dashboard
- Historical sentiment tracking
- Aggregate statistics across all analyses
- Visual sentiment distribution charts
- CSV export of analysis history
- Clear history option for fresh start

### 📰 News Integration
- Real-time social issue headlines from NewsAPI
- Pre-populated headlines for quick analysis
- Search news on climate change, poverty, healthcare, and more

### ⚙️ Advanced Text Processing
- **Language Detection** → Automatically identifies input language
- **Auto-Translation** → Translates non-English text for emotion analysis
- **Emoji Handling** → Converts emojis to text representations
- **Abbreviation Expansion** → Expands common internet slang
- **Text Cleaning** → Removes URLs, mentions, hashtags

## 📋 How It Works

1. **Text Input** → User enters or uploads text
2. **Language Detection** → System identifies the language
3. **Text Cleaning** → Preprocessing removes noise
4. **Translation (if needed)** → Non-English text translated to English
5. **Model Inference** → Transformer models predict sentiment & emotion
6. **Visualization** → Results displayed with confidence scores & charts
7. **History Logging** → All analyses tracked in CSV format

## 💾 Installation & Setup

### Local Development
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SocioSentiment.git
cd SocioSentiment

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add your NEWS API key to .streamlit/secrets.toml
# NEWS_API_KEY = "your_api_key_here"

# Run locally
streamlit run app.py
```

### Environment Variables
Create `.streamlit/secrets.toml`:
```toml
NEWS_API_KEY = "your_news_api_key_here"
```

Get your FREE API key from: https://newsapi.org/


## 🌐 Live Deployment

**Try the live app here:** https://sociosentiment-3kxktkrdd5bnlppxsv3zme.streamlit.app/

### Deploy Your Own Version

**Step 1:** Push to GitHub
```bash
git add .
git commit -m "Deploy SocioSentiment"
git push origin main
```

**Step 2:** Deploy on Streamlit Cloud
- Visit: https://share.streamlit.io/
- Click "Deploy an app"
- Select your GitHub repo: `YOUR_USERNAME/SocioSentiment`
- Main file: `SocioSentiment/app.py`
- Click "Deploy"

**Step 3:** Add Secrets
- Go to your app → ⋮ menu → "Settings"
- Click "Secrets" tab
- Add: `NEWS_API_KEY = "your_api_key"`
- Save (app will restart)

## 📊 Use Cases

✅ **Social Issue Monitoring** → Track public sentiment on climate, healthcare, poverty  
✅ **Multilingual Analysis** → Analyze global conversations in native languages  
✅ **Trend Analysis** → Identify emotional patterns across large datasets  
✅ **Research & Insights** → Export data for further analysis  
✅ **Real-time News Analysis** → Understand public mood about breaking news  

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend | Python 3.8+ |
| ML Models | Transformers (Hugging Face) |
| GPU Support | PyTorch |
| Data Processing | Pandas, Scikit-Learn |
| Translation | Google Translate API (googletrans) |
| News | NewsAPI |
| Language Detection | LangDetect |

## 📦 Dependencies

```
torch
transformers
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
googletrans==4.0.0-rc1
langdetect
emoji
requests
```

## ⚡ Performance Tips

- **First Load:** Models take 2-5 minutes to download & cache
- **GPU:** Recommended for batch processing (check PyTorch GPU setup)
- **CPU:** Works fine for single-text analysis
- **Caching:** Leverages `st.cache_resource` for fast subsequent runs

## 📝 File Structure

```
SocioSentiment/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── history.csv                 # Analysis history (auto-generated)
├── .streamlit/
│   └── secrets.toml           # API keys (add your NEWS_API_KEY)
└── .gitignore                 # Excludes secrets from Git
```

## 🔐 Security

- ✅ Secrets stored in `.streamlit/secrets.toml` (excluded from Git)
- ✅ No data sent to external servers except translation & news API
- ✅ All models run locally on your machine/server
- ✅ API keys loaded from environment at runtime

## 🐛 Troubleshooting

**Q: "ModuleNotFoundError: No module named 'streamlit'"**  
A: Run `pip install -r requirements.txt`

**Q: "NEWS API not working"**  
A: Make sure `NEWS_API_KEY` is set in `.streamlit/secrets.toml`

**Q: "Slow inference"**  
A: First run downloads models (~2GB). Subsequent runs are faster. Consider GPU support.

**Q: "Language not supported"**  
A: Fallback to English translation. Check supported languages list above.

## 📄 License

MIT License - feel free to use and modify

## 🤝 Contributing

Issues, suggestions, and PRs welcome! Help improve SocioSentiment.

---

**Made with ❤️ for understanding global sentiment on social issues**

