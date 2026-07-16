import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

# Load local environment variables if present (override=True updates keys dynamically)
load_dotenv(override=True)

# Import our custom utilities
import nlp_utils
import llm_utils

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="News.AI | Intelligent News Analyst",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & GLASSMORPHISM AESTHETICS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Base font override */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    .subtitle-text {
        color: #8A99AD;
        font-size: 1.1rem;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Premium Glassmorphic Card Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Sentiment Badges */
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10B981;
        border-radius: 30px;
        padding: 6px 16px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.1);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #EF4444;
        border-radius: 30px;
        padding: 6px 16px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(239, 68, 68, 0.1);
    }
    .badge-neutral {
        background-color: rgba(100, 116, 139, 0.12);
        border: 1px solid rgba(100, 116, 139, 0.3);
        color: #94A3B8;
        border-radius: 30px;
        padding: 6px 16px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(100, 116, 139, 0.1);
    }
    
    /* Entity Tags */
    .entity-tag {
        background-color: rgba(0, 114, 255, 0.08);
        border: 1px solid rgba(0, 114, 255, 0.2);
        color: #38BDF8;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 4px;
        display: inline-block;
    }
    
    /* Metric Boxes */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Clean chat UI elements */
    .chat-bubble {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- WEB SCRAPING UTILITY ---
def fetch_article_text(url: str) -> dict:
    """Scrapes clean text from a news article URL with robust fallback parsing."""
    try:
        # Standard headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Try fetching with verification, fallback to unverified if SSL error
        try:
            res = requests.get(url, headers=headers, timeout=12)
        except requests.exceptions.SSLError:
            res = requests.get(url, headers=headers, timeout=12, verify=False)
            
        res.raise_for_status()
        
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # 1. Clean up boilerplates and noise
        for noise in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
            noise.decompose()
            
        # 2. Extract title
        title = ""
        # Look for typical article title elements
        title_tag = soup.find('h1') or soup.find('meta', property='og:title')
        if title_tag:
            title = title_tag.get_text().strip() if hasattr(title_tag, 'get_text') else title_tag.get('content', '')
            
        if not title:
            title = soup.title.get_text().strip() if soup.title else "Extracted Article"
            
        # 3. Extract core text content
        # Check standard article containers first to find focused text
        article_body = soup.find('article')
        if not article_body:
            article_body = soup.find(attrs={"itemprop": "articleBody"})
        if not article_body:
            article_body = soup.find('main')
        if not article_body:
            common_classes = ['article-body', 'story-body', 'post-content', 'entry-content', 'article-content', 'story-content', 'post-body']
            for cls in common_classes:
                article_body = soup.find(class_=cls)
                if article_body:
                    break
                    
        container = article_body if article_body else soup
        
        # Find paragraphs inside the selected container
        paragraphs = container.find_all('p')
        texts = []
        for p in paragraphs:
            pt = p.get_text().strip()
            # Eliminate extremely short lines (ads, sharing prompts)
            if len(pt) > 40:
                texts.append(pt)
                
        full_text = "\n\n".join(texts)
        
        # Fallback if no paragraph tags yielded enough content
        if not full_text or len(full_text.split()) < 50:
            lines = [line.strip() for line in container.get_text().splitlines() if line.strip()]
            # Filter out lines that look like headers/footers (too short or menu items)
            text_blocks = [line for line in lines if len(line) > 60]
            full_text = "\n\n".join(text_blocks)
            
        if not full_text or len(full_text.strip()) == 0:
            return {"success": False, "error": "Could not extract readable text content from this webpage.", "title": "", "text": ""}
            
        return {"success": True, "title": title, "text": full_text}
    except Exception as e:
        return {"success": False, "error": str(e), "title": "", "text": ""}

# --- APP INITIALIZATION ---
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "title" not in st.session_state:
    st.session_state.title = ""
if "nlp_results" not in st.session_state:
    st.session_state.nlp_results = None
if "summary_tldr" not in st.session_state:
    st.session_state.summary_tldr = ""
if "summary_bullets" not in st.session_state:
    st.session_state.summary_bullets = ""
if "summary_detailed" not in st.session_state:
    st.session_state.summary_detailed = ""
if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prev_source_type" not in st.session_state:
    st.session_state.prev_source_type = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

# --- SIDEBAR: CONFIG & INPUTS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("News.AI Engine")
    st.markdown("Select your news input source below.")
    
    # Input source configuration
    st.subheader("Source Selection")
    source_type = st.radio(
        "Choose Input Method",
        ["Enter Article URL", "Paste Raw Text"]
    )
    
    # Clear analysis state if input method changes to avoid mixing content
    if source_type != st.session_state.prev_source_type:
        st.session_state.raw_text = ""
        st.session_state.title = ""
        st.session_state.nlp_results = None
        st.session_state.summary_tldr = ""
        st.session_state.summary_bullets = ""
        st.session_state.summary_detailed = ""
        st.session_state.llm_analysis = None
        st.session_state.chat_history = []
        st.session_state.prev_source_type = source_type
        
    staged_title = ""
    staged_text = ""
    
    if source_type == "Enter Article URL":
        url_input = st.text_input(
            "News Article URL", 
            placeholder="https://techcrunch.com/..."
        )
        if st.button("Fetch & Parse URL", use_container_width=True):
            if url_input:
                with st.spinner("Scraping webpage..."):
                    scrape_result = fetch_article_text(url_input)
                    if scrape_result["success"]:
                        st.session_state.raw_text = scrape_result["text"]
                        st.session_state.title = scrape_result["title"]
                        st.session_state.nlp_results = None
                        st.session_state.summary_tldr = ""
                        st.session_state.summary_bullets = ""
                        st.session_state.summary_detailed = ""
                        st.session_state.llm_analysis = None
                        st.session_state.chat_history = []
                        st.toast("Article fetched successfully!", icon="✅")
                    else:
                        st.error(f"Failed to fetch: {scrape_result['error']}")
            else:
                st.warning("Please enter a valid URL first.")
                
        staged_title = st.session_state.title
        staged_text = st.session_state.raw_text
                
    elif source_type == "Paste Raw Text":
        pasted_title = st.text_input("Article Title (Optional)", value="Pasted News Article")
        pasted_text = st.text_area("Paste Full Article Text Here", height=250)
        
        staged_title = pasted_title
        staged_text = pasted_text

    st.divider()
    st.info(
        "💡 **💡 Tip:** This application processes text locally using NLTK and connects to Gemini models using LangChain for summaries, structured sentiment data, and contextual question-answering."
    )

# --- MAIN PAGE HUB ---
st.markdown("<h1 class='gradient-text'>News.AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Intelligent News Summarization, Sentiment Analysis, and Grounded QA powered by LLMs</p>", unsafe_allow_html=True)

# Determine if we should show input text status
if staged_text:
    # Quick Summary Metrics of Input Text
    char_count = len(staged_text)
    word_count = len(staged_text.split())
    
    # Display details of the active text in an expander
    with st.expander(f"📖 Active Article: {staged_title or 'Untitled'} ({word_count} words)", expanded=False):
        if staged_title:
            st.markdown(f"### {staged_title}")
        st.write(staged_text)
        
    # Process trigger button
    trigger_col1, trigger_col2 = st.columns([1, 3])
    with trigger_col1:
        start_analysis = st.button("🚀 Analyze Article", type="primary", use_container_width=True)
    
    if start_analysis:
        # Clear past chat history on new article run
        st.session_state.chat_history = []
        st.session_state.raw_text = staged_text
        st.session_state.title = staged_title or "News Article"
        
        with st.spinner("Analyzing text and generating summaries..."):
            # 1. Traditional NLP Preprocessing
            st.session_state.nlp_results = nlp_utils.perform_nlp_preprocessing(st.session_state.raw_text)
            
            # Check for API key
            if not st.session_state.api_key:
                st.warning("⚠️ Traditional NLP completed. Please configure the GEMINI_API_KEY environment variable for LLM summarization, sentiment extraction, and QA.")
            else:
                # 2. LLM Summaries (Parallelizable or sequential. Let's fetch them using our chains)
                st.session_state.summary_tldr = llm_utils.generate_summary(st.session_state.raw_text, "Quick TL;DR", st.session_state.api_key)
                st.session_state.summary_bullets = llm_utils.generate_summary(st.session_state.raw_text, "Bullet Points", st.session_state.api_key)
                st.session_state.summary_detailed = llm_utils.generate_summary(st.session_state.raw_text, "Detailed Summary", st.session_state.api_key)
                
                # 3. Structured Sentiment and Entity Extraction
                st.session_state.llm_analysis = llm_utils.analyze_sentiment_and_entities(st.session_state.raw_text, st.session_state.api_key)
                st.toast("Analysis complete!", icon="🎉")
                st.rerun()

else:
    # Empty state instructions
    st.markdown("""
    <div class='glass-card' style='text-align: center; padding: 50px 20px;'>
        <img src='https://img.icons8.com/color/96/news.png' width='100'/><br/><br/>
        <h3>Welcome to News.AI Engine</h3>
        <p style='color: #8A99AD;'>Please load a news article using the left sidebar, and click <b>Analyze Article</b> to begin.</p>
    </div>
    """, unsafe_allow_html=True)

# --- DISPLAY ANALYSIS RESULTS ---
if st.session_state.raw_text and st.session_state.nlp_results:
    # Setup tabs
    tab_summary, tab_nlp, tab_entities, tab_qa = st.tabs([
        "📊 Summary & Sentiment Insights", 
        "⚙️ NLP Preprocessing Analysis", 
        "🏷️ Keyword & Entity Hub",
        "💬 Interactive Q&A Assistant"
    ])
    
    # ------------------- TAB 1: SUMMARY & SENTIMENT -------------------
    with tab_summary:
        col_sum, col_sent = st.columns([3, 2])
        
        with col_sum:
            st.markdown("### 📝 Generated Summaries")
            
            if st.session_state.summary_tldr:
                # Segmented Summary Tabs
                sum_style = st.radio(
                    "Select Summary Format",
                    ["Quick TL;DR", "Bullet Points", "Detailed Narrative"],
                    horizontal=True
                )
                
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                if sum_style == "Quick TL;DR":
                    st.markdown("**TL;DR Summary:**")
                    st.write(st.session_state.summary_tldr)
                elif sum_style == "Bullet Points":
                    st.markdown("**Key Takeaways:**")
                    st.write(st.session_state.summary_bullets)
                else:
                    st.markdown("**Comprehensive Narrative:**")
                    st.write(st.session_state.summary_detailed)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Summaries require a GEMINI_API_KEY in the environment.")
                
            # Original text basic metrics
            nlp_res = st.session_state.nlp_results
            st.markdown("#### Text Statistics")
            met1, met2, met3, met4 = st.columns(4)
            with met1:
                st.markdown(f"<div class='glass-card' style='padding:15px; text-align:center;'><div class='metric-label'>Sentences</div><div class='metric-value'>{nlp_res['sentence_count']}</div></div>", unsafe_allow_html=True)
            with met2:
                st.markdown(f"<div class='glass-card' style='padding:15px; text-align:center;'><div class='metric-label'>Raw Words</div><div class='metric-value'>{nlp_res['raw_token_count']}</div></div>", unsafe_allow_html=True)
            with met3:
                st.markdown(f"<div class='glass-card' style='padding:15px; text-align:center;'><div class='metric-label'>Cleaned Words</div><div class='metric-value'>{nlp_res['filtered_token_count']}</div></div>", unsafe_allow_html=True)
            with met4:
                st.markdown(f"<div class='glass-card' style='padding:15px; text-align:center;'><div class='metric-label'>Vocabulary Size</div><div class='metric-value'>{nlp_res['vocabulary_size']}</div></div>", unsafe_allow_html=True)

        with col_sent:
            st.markdown("### 🔮 Sentiment Analysis")
            
            if st.session_state.llm_analysis:
                analysis = st.session_state.llm_analysis
                sent = analysis.get("sentiment", "Neutral")
                score = analysis.get("confidence_score", 0.0)
                explanation = analysis.get("sentiment_explanation", "")
                
                # Sentiment badge
                badge_html = ""
                if sent == "Positive":
                    badge_html = f"<div class='badge-positive'>🟢 Positive ({score*100:.1f}%)</div>"
                elif sent == "Negative":
                    badge_html = f"<div class='badge-negative'>🔴 Negative ({score*100:.1f}%)</div>"
                else:
                    badge_html = f"<div class='badge-neutral'>⚪ Neutral ({score*100:.1f}%)</div>"
                    
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(f"<h4>Document Sentiment:</h4> {badge_html}", unsafe_allow_html=True)
                st.markdown(f"<p style='margin-top:15px; color:#E2E8F0;'><i>\"{explanation}\"</i></p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Emotion Breakdown Chart
                emotions = analysis.get("emotion_breakdown", {})
                if emotions:
                    st.markdown("#### Emotion Signature")
                    emo_df = pd.DataFrame({
                        "Emotion": [k.capitalize() for k in emotions.keys()],
                        "Score": list(emotions.values())
                    }).sort_values(by="Score", ascending=True)
                    
                    fig = px.bar(
                        emo_df, 
                        x="Score", 
                        y="Emotion", 
                        orientation='h',
                        color="Score",
                        color_continuous_scale="Viridis",
                        range_x=[0.0, 1.0],
                        height=250,
                        labels={"Score": "Confidence Level"}
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#F8FAFC',
                        margin=dict(l=10, r=10, t=10, b=10),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Sentiment insights require a GEMINI_API_KEY in the environment.")
                
    # ------------------- TAB 2: NLP PREPROCESSING -------------------
    with tab_nlp:
        st.markdown("### ⚙️ NLP Pipelines & Preprocessing Steps")
        st.markdown("Traditional NLP pipeline transforms raw paragraphs into clean, structured tokens ready for machine learning.")
        
        nlp_res = st.session_state.nlp_results
        
        col_stages, col_wc = st.columns([3, 2])
        
        with col_stages:
            st.markdown("#### NLP Pipeline Stage Inspection")
            
            stage = st.selectbox(
                "Select Pipeline Stage to Inspect",
                ["1. Original vs. Cleaned Text", "2. Tokenized Words", "3. Stopwords Removed & Lemmatized", "4. Part-of-Speech Tagging (POS)"]
            )
            
            st.markdown("<div class='glass-card' style='max-height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)
            if stage == "1. Original vs. Cleaned Text":
                st.markdown("**Original Text Sample (First 400 chars):**")
                st.write(st.session_state.raw_text[:400] + "...")
                st.divider()
                st.markdown("**Cleaned Formatting & Whitespace:**")
                st.write(nlp_res["sentences"][0] + " ... (sentence tokenized)")
                
            elif stage == "2. Tokenized Words":
                st.markdown(f"**Raw Word Tokens (Count: {nlp_res['raw_token_count']}):**")
                st.write(nlp_res["raw_tokens"][:100])
                st.markdown("*Showing first 100 tokens. Normalization: Lowercasing, splitting on whitespace and punctuation.*")
                
            elif stage == "3. Stopwords Removed & Lemmatized":
                st.markdown(f"**Cleaned Lemmas (Count: {nlp_res['filtered_token_count']}):**")
                st.write(nlp_res["preprocessed_tokens"][:100])
                st.markdown("*Filtered out English stopwords, symbols, short character noises. Reduced terms to base forms (lemmas) via WordNet.*")
                
            elif stage == "4. Part-of-Speech Tagging (POS)":
                st.markdown("**Grammatical POS Tagging (First 50 terms):**")
                tags_df = pd.DataFrame(nlp_res["pos_tags"][:50], columns=["Word", "POS Tag"])
                
                # Add descriptions for popular tags
                tag_meanings = {
                    'NN': 'Noun, singular', 'NNS': 'Noun, plural', 'NNP': 'Proper noun, singular',
                    'VB': 'Verb, base form', 'VBD': 'Verb, past tense', 'VBG': 'Verb, gerund/present participle',
                    'VBN': 'Verb, past participle', 'JJ': 'Adjective', 'RB': 'Adverb', 'CD': 'Cardinal number'
                }
                tags_df["Description"] = tags_df["POS Tag"].map(lambda t: tag_meanings.get(t, 'Other'))
                st.dataframe(tags_df, use_container_width=True, height=250)
                st.markdown("*POS Tags help LLMs and parser systems understand grammatical relationships in sentences.*")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_wc:
            st.markdown("#### ☁️ Text Word Cloud")
            if nlp_res["word_frequencies"]:
                # Generate Word Cloud
                try:
                    wc = WordCloud(
                        width=800, 
                        height=500, 
                        background_color='black', 
                        colormap='Blues',
                        max_words=50
                    ).generate_from_frequencies(nlp_res["word_frequencies"])
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    fig.patch.set_facecolor('black')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")
            else:
                st.info("No words to construct word cloud.")

    # ------------------- TAB 3: KEYWORD & ENTITY HUB -------------------
    with tab_entities:
        col_ent, col_freq = st.columns([3, 2])
        
        with col_ent:
            st.markdown("### 🏷️ Extracted Entities (Gemini + LLM)")
            st.markdown("Key items, organizations, people, and tech mentioned in the news article.")
            
            if st.session_state.llm_analysis and "entities" in st.session_state.llm_analysis:
                entities = st.session_state.llm_analysis["entities"]
                if entities:
                    # Group entities by type
                    ent_groups = {}
                    for e in entities:
                        etype = e.get("type", "Other")
                        ename = e.get("name", "")
                        erelevance = e.get("relevance", "")
                        
                        if etype not in ent_groups:
                            ent_groups[etype] = []
                        ent_groups[etype].append((ename, erelevance))
                        
                    # Display entity lists
                    for etype, items in ent_groups.items():
                        st.markdown(f"**{etype}s**")
                        for ename, erelevance in items:
                            st.markdown(
                                f"<span class='entity-tag'>{ename}</span> <span style='font-size:0.85rem; color:#8A99AD;'>- {erelevance}</span>", 
                                unsafe_allow_html=True
                            )
                        st.divider()
                else:
                    st.info("No notable entities extracted.")
            else:
                st.info("Entity extraction requires a GEMINI_API_KEY in the environment.")

        with col_freq:
            st.markdown("### 📊 Traditional Word Frequency")
            st.markdown("Top 10 most common words processed by our NLTK system.")
            
            common_words = nlp_res["most_common_words"][:10]
            if common_words:
                words = [w[0] for w in common_words]
                freqs = [w[1] for w in common_words]
                
                fig = px.bar(
                    x=freqs, 
                    y=words, 
                    orientation='h', 
                    color=freqs,
                    color_continuous_scale="Blues",
                    labels={"x": "Occurrence Count", "y": "Word"}
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No frequency records available.")

    # ------------------- TAB 4: INTERACTIVE Q&A CHAT -------------------
    with tab_qa:
        st.markdown("### 💬 Grounded Q&A Assistant")
        st.markdown("Ask the LLM any question about the article. The assistant is grounded and will only answer based on facts provided in the text.")
        
        # Display chat interface
        if not st.session_state.api_key:
            st.info("Please configure the GEMINI_API_KEY environment variable to use the Q&A Assistant.")
        else:
            # Suggest questions layout
            st.markdown("💡 **Suggested Questions:**")
            sug_col1, sug_col2, sug_col3 = st.columns(3)
            
            sug_q1 = "What are the core arguments or facts presented?"
            sug_q2 = "Who/what is the main focus of this article?"
            sug_q3 = "What are the future implications or issues?"
            
            clicked_sug = ""
            with sug_col1:
                if st.button(sug_q1, key="sug1", use_container_width=True):
                    clicked_sug = sug_q1
            with sug_col2:
                if st.button(sug_q2, key="sug2", use_container_width=True):
                    clicked_sug = sug_q2
            with sug_col3:
                if st.button(sug_q3, key="sug3", use_container_width=True):
                    clicked_sug = sug_q3
            
            st.divider()
            
            # Message box container
            chat_container = st.container(height=350)
            
            with chat_container:
                # Render existing chat
                for role, message in st.session_state.chat_history:
                    with st.chat_message("user" if role == "User" else "assistant"):
                        st.write(message)
            
            # Handle new inputs
            user_input = st.chat_input("Ask a question about the article...")
            
            # If a user typed a question OR clicked a suggestion
            active_question = user_input or clicked_sug
            
            if active_question:
                # Add to chat history immediately
                st.session_state.chat_history.append(("User", active_question))
                
                # Render user message
                with chat_container:
                    with st.chat_message("user"):
                        st.write(active_question)
                        
                # Generate AI response
                with st.spinner("Analyzing article context..."):
                    response = llm_utils.answer_article_question(
                        text=st.session_state.raw_text,
                        question=active_question,
                        chat_history=st.session_state.chat_history[:-1], # excluding the current question itself
                        api_key=st.session_state.api_key
                    )
                
                # Add response to history
                st.session_state.chat_history.append(("Assistant", response))
                
                # Render assistant message
                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(response)
                        
                st.rerun()
