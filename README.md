# News.AI: Intelligent News Summarization & Sentiment Analysis System

News.AI is a premium, interactive web application designed to help users quickly digest online news articles. By combining traditional Natural Language Processing (NLP) pipelines with modern Large Language Models (LLMs), the system parses news articles (via direct text input or web URL scraping), performs linguistic preprocessing, generates diverse summaries, conducts deep sentiment/emotion analysis, and allows readers to interact with a grounded chatbot to ask questions about the article.

---

## 🏗️ System Architecture & Workflow

The system coordinates standard NLP pipelines with generative LLM chains using LangChain and Google Gemini models.

```mermaid
graph TD
    A[User Input: URL, Text, or Sample] --> B{Input Type}
    
    B -->|URL| C[BeautifulSoup Web Scraper]
    B -->|Text / Sample| D[Raw Article Text]
    
    C --> D
    
    D --> E[NLP Preprocessing Engine]
    D --> F[LangChain LLM Chains]
    
    %% NLP Pipeline
    subgraph Traditional NLP Pipeline (nlp_utils.py)
        E --> E1[Sentence & Word Tokenization]
        E1 --> E2[Stopword Filtering]
        E2 --> E3[WordNet Lemmatization]
        E3 --> E4[POS Tagging & Word Freq]
    end
    
    %% LLM Chains
    subgraph LangChain & Gemini AI (llm_utils.py)
        F --> F1[Summarization Chain]
        F1 --> F1a[TL;DR, Bullets, Narrative]
        
        F --> F2[Structured Sentiment & Entity Chain]
        F2 --> F2a[JSON: Sentiment Label, Confidence, Emotion Breakdown, Entity Mapping]
        
        F --> F3[Contextual Q&A Chain]
        F3 --> F3a[Chatbot responses grounded by article text]
    end
    
    %% UI Presentation
    E4 --> G[Streamlit Dashboard UI]
    F1a --> G
    F2a --> G
    F3a --> G
    
    style G fill:#0072FF,stroke:#333,stroke-width:2px,color:#fff
```

---

## 🌟 Key Features

1. **Flexible Ingestion**: Scrape and extract clean text from any online news URL (e.g. CNN, TechCrunch, BBC) or paste text directly.
2. **Traditional NLP Pipeline Stage-Viewer**: Inspect how text is processed at different levels—from sentence/word tokenization, stopword removal, lemmatization, to grammatical Part-of-Speech (POS) tagging.
3. **Structured Sentiment & Emotion Analysis**: Extract positive/negative/neutral labels with confidence levels, alongside a 6-dimension emotional signature (Joy, Sadness, Anger, Fear, Surprise, Analytical) visualized on charts.
4. **Multiple Summarization Formats**: Instantly switch between three summary formats generated on demand:
   - **Quick TL;DR**: A brief, high-impact 2-3 sentence overview.
   - **Key Takeaways**: Bullet points highlighting core facts (Who, What, When, Where, Why).
   - **Comprehensive Narrative**: A structured editorial summary detailing background and implications.
5. **Grounded Q&A Chatbot**: Chat with an LLM that is strictly grounded by the article's text, preventing hallucinations or out-of-context assumptions.
6. **Sleek Premium UI**: Glassmorphic, dark-themed responsive layout with custom fonts, colors, and interactive Plotly visualization charts.

---

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (enhanced with custom CSS)
- **LLM Orchestration**: [LangChain](https://www.langchain.com/) (using LCEL chains)
- **Generative AI Model**: Google Gemini (`gemini-1.5-flash` for high-speed reasoning and JSON output)
- **Traditional NLP**: [NLTK](https://www.nltk.org/) (Tokenizers, Stopwords, WordNet Lemmatizer, POS Tagger)
- **Data Scraping**: [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) & [Requests](https://requests.readthedocs.io/)
- **Visualizations**: [Plotly Express](https://plotly.com/), [WordCloud](https://github.com/amueller/word_cloud), and [Matplotlib](https://matplotlib.org/)

---

## 🚀 Setup and Local Installation

### Prerequisites
- Python 3.9 to 3.11 installed.
- A **Google Gemini API Key** (Get one free from [Google AI Studio](https://aistudio.google.com/)).

### Steps

1. **Clone or Copy the Repository**
   Ensure all files are placed in a project directory:
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Rename `.env.example` to `.env` and add your API key:
   ```bash
   copy .env.example .env
   # Edit .env and set:
   # GEMINI_API_KEY=your_actual_api_key_here
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   The application will automatically download required NLTK assets on its initial launch and open in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment via GitHub & Streamlit Community Cloud

You can host this application for free on [Streamlit Community Cloud](https://streamlit.io/cloud) directly from your GitHub repository:

1. **Push your code to GitHub**:
   Make sure you commit the following files:
   - `app.py`
   - `nlp_utils.py`
   - `llm_utils.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore` (ensure `.env` and your `venv/` folder are ignored!)

2. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **Create App** (or **New App**).
   - Select your repository, branch, and set the Main file path to `app.py`.
   - Click **Advanced Settings**.
   - In the **Secrets** section, configure your Gemini API Key so the app has access without requiring a sidebar input:
     ```toml
     GEMINI_API_KEY = "your_actual_api_key_here"
     ```
   - Click **Deploy**. Your app will be live on a public URL in a few minutes!
