import re
import collections
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Programmatic NLTK resource downloader
def download_nltk_resources():
    """Ensure all required NLTK resources are downloaded."""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }
    
    for resource_id, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource_id, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download NLTK resource {resource_id}: {e}")

# Run downloader on module import
download_nltk_resources()

def clean_html_and_formatting(text: str) -> str:
    """Removes HTML tags, extra whitespaces, and basic cleaning."""
    if not text:
        return ""
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', '', text)
    # Remove excess whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def perform_nlp_preprocessing(text: str) -> dict:
    """
    Performs standard NLP preprocessing steps:
    1. Sentence Tokenization
    2. Word Tokenization & Lowercasing
    3. Punctuation and Non-alphabetic token removal
    4. Stopword Removal
    5. Lemmatization
    6. Part-of-Speech Tagging
    
    Returns a dictionary with all intermediate representations and statistics.
    """
    cleaned_text = clean_html_and_formatting(text)
    
    # 1. Sentence Tokenization
    try:
        sentences = sent_tokenize(cleaned_text)
    except Exception:
        # Fallback if sentence tokenizer fails
        sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]

    # 2. Word Tokenization
    try:
        raw_tokens = word_tokenize(cleaned_text)
    except Exception:
        # Fallback split-based tokenization
        raw_tokens = re.findall(r'\b\w+\b', cleaned_text)
        
    # Lowercase tokens
    tokens_lower = [token.lower() for token in raw_tokens]

    # 3 & 4. Punctuation removal, Non-alphabetic token removal and Stopword filtering
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Fallback basic stopword list if nltk fails
        stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                      "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", 
                      "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
                      "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", 
                      "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", 
                      "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", 
                      "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", 
                      "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"}

    filtered_tokens = [
        token for token in tokens_lower 
        if token.isalpha() and token not in stop_words and len(token) > 1
    ]

    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in filtered_tokens:
        try:
            lemmatized = lemmatizer.lemmatize(token)
        except Exception:
            lemmatized = token
        lemmatized_tokens.append(lemmatized)

    # 6. Part-of-Speech Tagging (on pre-filtered tokens or original cleaned tokens)
    # We'll tag a sample of the first 100 raw tokens to keep it fast and readable,
    # and also POS tag the filtered tokens.
    pos_tagged_filtered = []
    try:
        pos_tagged_filtered = pos_tag(lemmatized_tokens)
    except Exception as e:
        print(f"POS tagging failed: {e}")
        # Mock tag as 'NOUN' as fallback
        pos_tagged_filtered = [(t, 'N/A') for t in lemmatized_tokens]

    # Count word frequencies
    word_counts = collections.Counter(lemmatized_tokens)
    most_common_words = word_counts.most_common(20)

    # Preprocessed output pack
    return {
        "original_char_count": len(text),
        "cleaned_char_count": len(cleaned_text),
        "sentence_count": len(sentences),
        "raw_token_count": len(raw_tokens),
        "filtered_token_count": len(filtered_tokens),
        "vocabulary_size": len(set(lemmatized_tokens)),
        "sentences": sentences,
        "raw_tokens": raw_tokens[:200],  # Limit display list length
        "preprocessed_tokens": lemmatized_tokens,
        "pos_tags": pos_tagged_filtered[:100],  # Limit display list length
        "word_frequencies": dict(word_counts),
        "most_common_words": most_common_words
    }
