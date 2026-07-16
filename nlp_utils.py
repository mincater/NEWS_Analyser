import re
import collections
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def download_nltk_resources():
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

download_nltk_resources()

def clean_html_and_formatting(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<[^>]+>', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def perform_nlp_preprocessing(text: str) -> dict:
    """Clean and tokenize text, filter stopwords, perform lemmatization and POS tagging."""
    cleaned_text = clean_html_and_formatting(text)
    
    try:
        sentences = sent_tokenize(cleaned_text)
    except Exception:
        sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]

    try:
        raw_tokens = word_tokenize(cleaned_text)
    except Exception:
        raw_tokens = re.findall(r'\b\w+\b', cleaned_text)
        
    tokens_lower = [token.lower() for token in raw_tokens]

    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
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

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in filtered_tokens:
        try:
            lemmatized = lemmatizer.lemmatize(token)
        except Exception:
            lemmatized = token
        lemmatized_tokens.append(lemmatized)

    pos_tagged_filtered = []
    try:
        pos_tagged_filtered = pos_tag(lemmatized_tokens)
    except Exception as e:
        print(f"POS tagging failed: {e}")
        pos_tagged_filtered = [(t, 'N/A') for t in lemmatized_tokens]

    word_counts = collections.Counter(lemmatized_tokens)
    most_common_words = word_counts.most_common(20)

    return {
        "original_char_count": len(text),
        "cleaned_char_count": len(cleaned_text),
        "sentence_count": len(sentences),
        "raw_token_count": len(raw_tokens),
        "filtered_token_count": len(filtered_tokens),
        "vocabulary_size": len(set(lemmatized_tokens)),
        "sentences": sentences,
        "raw_tokens": raw_tokens[:200],
        "preprocessed_tokens": lemmatized_tokens,
        "pos_tags": pos_tagged_filtered[:100],
        "word_frequencies": dict(word_counts),
        "most_common_words": most_common_words
    }
