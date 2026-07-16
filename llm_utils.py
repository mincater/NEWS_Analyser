import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser

def get_llm(api_key: str = None, model_name: str = "gemini-3.5-flash", temperature: float = 0.2):
    """
    Initializes and returns the ChatGoogleGenerativeAI model.
    Prioritizes the passed API key, otherwise falls back to GEMINI_API_KEY env variable.
    """
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_api_key:
        raise ValueError("Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable.")
        
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=effective_api_key,
        temperature=temperature,
        max_tokens=2048
    )

def generate_summary(text: str, summary_type: str, api_key: str = None) -> str:
    """
    Generates a summary based on the requested format.
    Types: 'Quick TL;DR', 'Detailed Summary', 'Bullet Points'
    """
    try:
        llm = get_llm(api_key=api_key, model_name="gemini-3.5-flash", temperature=0.3)
        
        # Define prompts based on type
        if summary_type == "Quick TL;DR":
            system_prompt = (
                "You are a professional editor. Write a concise, 2-3 sentence summary (TL;DR) "
                "capturing the core event and impact of the following news article. Keep it brief and high-impact."
            )
        elif summary_type == "Bullet Points":
            system_prompt = (
                "You are an analytical assistant. Extract the key takeaways and facts from the following "
                "news article and list them as bullet points (max 6-8 bullets). Focus on who, what, when, where, and why."
            )
        else: # Detailed Summary
            system_prompt = (
                "You are a journalist. Provide a comprehensive summary of the following news article. "
                "Structure it with a main summary paragraph, followed by subheadings or paragraphs "
                "explaining the context/background, major players, and potential future implications. Keep it readable and narrative."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Article Content:\n\n{article_text}")
        ])
        
        # Simple LCEL Chain
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"article_text": text})
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def analyze_sentiment_and_entities(text: str, api_key: str = None) -> dict:
    """
    Performs sentiment analysis and key entity extraction in a single structured JSON response.
    """
    try:
        # Initialize LLM with temperature 0 for consistent structured outputs
        llm = get_llm(api_key=api_key, model_name="gemini-3.5-flash", temperature=0.0)
        
        system_prompt = (
            "You are a precise data extraction agent. Analyze the provided news article "
            "and extract sentiment and key entities. You MUST return ONLY a valid JSON object matching the schema below. "
            "Do not wrap the JSON in markdown code blocks like ```json ... ```, return ONLY the raw JSON text.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "sentiment": "Positive" | "Negative" | "Neutral",\n'
            '  "confidence_score": 0.0 to 1.0 (float),\n'
            '  "sentiment_explanation": "brief explanation why this sentiment was chosen based on tone/words",\n'
            '  "emotion_breakdown": {\n'
            '    "joy": 0.0 to 1.0,\n'
            '    "sadness": 0.0 to 1.0,\n'
            '    "anger": 0.0 to 1.0,\n'
            '    "fear": 0.0 to 1.0,\n'
            '    "surprise": 0.0 to 1.0,\n'
            '    "analytical": 0.0 to 1.0\n'
            "  },\n"
            '  "entities": [\n'
            "    {\n"
            '      "name": "Entity Name (e.g. Google, Joe Biden)",\n'
            '      "type": "Person" | "Organization" | "Location" | "Date" | "Technology" | "Other",\n'
            '      "relevance": "Brief description of their role/context in this article"\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Article Content:\n\n{article_text}")
        ])
        
        # Simple JSON parser chain
        chain = prompt | llm | SimpleJsonOutputParser()
        result = chain.invoke({"article_text": text})
        return result
        
    except Exception as e:
        # Fallback dictionary structure in case of parsing errors or api limits
        error_msg = str(e)
        # Attempt to see if we can clean up common json wraps if it returned string instead
        try:
            # Check if it was returned as string inside the exception or message
            if "{" in error_msg:
                start = error_msg.find("{")
                end = error_msg.rfind("}") + 1
                return json.loads(error_msg[start:end])
        except Exception:
            pass
            
        return {
            "sentiment": "Neutral",
            "confidence_score": 0.0,
            "sentiment_explanation": f"Failed to perform sentiment analysis. Error: {error_msg}",
            "emotion_breakdown": {"joy": 0, "sadness": 0, "anger": 0, "fear": 0, "surprise": 0, "analytical": 0},
            "entities": []
        }

def answer_article_question(text: str, question: str, chat_history: list = None, api_key: str = None) -> str:
    """
    Answers user's questions about the article context.
    Uses the Gemini model grounded by the article text.
    """
    try:
        # Pro has a bit better reasoning for QA, but Flash is fine and faster. Let's use Flash.
        llm = get_llm(api_key=api_key, model_name="gemini-3.5-flash", temperature=0.2)
        
        system_prompt = (
            "You are an intelligent QA assistant specializing in news analysis.\n"
            "You are given the following news article:\n"
            "---------------------\n"
            "{article_context}\n"
            "---------------------\n"
            "Using ONLY the article text above, answer the user's question. "
            "If the answer is not mentioned, or cannot be directly inferred from the article, "
            "respond with: 'I am sorry, but the article does not provide enough information to answer this question.' "
            "Do not make up facts or use external knowledge. Be factual, concise, and professional."
        )
        
        # We can pass chat history if present for multi-turn conversation
        messages = [("system", system_prompt)]
        
        if chat_history:
            for speaker, msg in chat_history[-6:]: # Keep last 3 turns
                role = "user" if speaker == "User" else "assistant"
                messages.append((role, msg))
                
        messages.append(("user", "{question}"))
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({
            "article_context": text,
            "question": question
        })
        
    except Exception as e:
        return f"Error answering question: {str(e)}"
