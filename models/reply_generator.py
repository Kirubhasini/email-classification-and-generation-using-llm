import cohere
import logging
import os
import re
import pickle
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cohere API key
COHERE_API_KEY = "cohere key"  # Replace with your actual key
#KfRekS4l6ZtIsjIhiu8SMjAVnz3OXsMz5X5S6GP2
# Model paths
MODEL_PATH = "models/svm_priority_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Load stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    logger.warning("NLTK stopwords not available. Using empty set.")
    stop_words = set()

def preprocess_text(text):
    """Preprocess text for classification."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def classify_priority(email_body):
    """Classifies email priority as urgent or normal."""
    try:
        # Check if model files exist
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            logger.error("Model or vectorizer file not found.")
            return "normal"  # Default to normal if models aren't available
            
        # Load model and vectorizer
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Preprocess and transform email
        cleaned_text = preprocess_text(email_body)
        features = vectorizer.transform([cleaned_text])
        
        # Predict and return priority
        prediction = model.predict(features)[0]
        priority = "urgent" if prediction == 1 else "normal"
        logger.info(f"Email classified as {priority}")
        return priority
        
    except Exception as e:
        logger.exception(f"Error in priority classification: {e}")
        return "normal"  # Default to normal on error

def generate_reply(email_body, email_subject=None):
    """Generate reply for urgent emails only."""
    try:
        # First determine priority
        priority = classify_priority(email_body)
        
        # Only generate reply for urgent emails
        if priority == "urgent":
            co = cohere.Client(COHERE_API_KEY)
            
            # Create prompt for LLM
            prompt = f"""
            Email Subject: {email_subject if email_subject else 'No Subject'}
            Email Body: {email_body}
            
            This is an URGENT email that requires immediate attention. Generate a professional reply:
            """
            
            # Generate reply using Cohere
            response = co.generate(
                model='command-xlarge',
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,
            )
            
            reply = response.generations[0].text.strip()
            logger.info("Reply generated for urgent email")
            return reply
        else:
            # No reply for normal priority
            logger.info("Normal priority email - no reply generated")
            return ""
            
    except Exception as e:
        logger.exception(f"Error in reply generation: {e}")
        return "Error: Unable to generate reply."