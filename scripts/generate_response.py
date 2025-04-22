import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import json
from typing import List, Dict

# --- Configuration ---
MODEL_DIR = "models/sentiment_model"
THERAPY_DATA_PATH = "data/processed/combined_balanced_dataset.csv"
FALLBACK_RESPONSES = [
    "I'm here to listen if you want to share more.",
    "That sounds important. Would you like to elaborate?",
    "I want to understand better. Could you tell me more about that?"
]

# --- Load Sentiment Analysis Resources ---
def load_sentiment_model() -> tuple:
    """Load trained sentiment analysis model and label mapping"""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Load label mapping
    with open(os.path.join(MODEL_DIR, "label_mapping.json"), "r") as f:
        label_mapping = json.load(f)
    
    index_to_label = {v: k for k, v in label_mapping.items()}
    return model, tokenizer, index_to_label

# --- Load and Prepare Therapy Responses ---
def load_therapy_data() -> pd.DataFrame:
    """Load and preprocess therapy response dataset"""
    df = pd.read_csv(THERAPY_DATA_PATH)
    
    # Clean and prepare data
    df = df.dropna(subset=['response', 'Sentiment'])
    df['response'] = df['response'].str.split('|')  # Split multiple responses
    df = df.explode('response')  # Create separate rows for each response
    df['response'] = df['response'].str.strip()
    df = df[df['response'] != '']
    
    return df

# --- Sentiment Classification ---
def classify_sentiment(text: str, 
                      model: AutoModelForSequenceClassification,
                      tokenizer: AutoTokenizer,
                      index_to_label: Dict[int, str]) -> str:
    """Classify user input sentiment with confidence checking"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = probs.argmax().item()
    confidence = probs[0][predicted_class_id].item()
    
    # Handle low confidence predictions
    if confidence < 0.6:
        return "neutral"
    
    return index_to_label[predicted_class_id]

# --- Response Generation Logic ---
def generate_response(user_input: str,
                     model: AutoModelForSequenceClassification,
                     tokenizer: AutoTokenizer,
                     index_to_label: Dict[int, str],
                     therapy_df: pd.DataFrame) -> str:
    """Generate context-aware response based on sentiment analysis"""
    # 1. Analyze sentiment with confidence check
    sentiment = classify_sentiment(user_input, model, tokenizer, index_to_label)
    print(f"[DEBUG] Detected Sentiment: {sentiment} (User input: '{user_input}')")
    
    # 2. Filter responses by sentiment and context
    filtered = therapy_df[
        (therapy_df['Sentiment'] == sentiment) &
        (therapy_df['pattern'].str.lower().str.contains(user_input.lower()))
    ]
    
    # 3. Fallback to general sentiment-matched responses
    if filtered.empty:
        filtered = therapy_df[therapy_df['Sentiment'] == sentiment]
    
    # 4. Final fallback to neutral responses
    if filtered.empty:
        filtered = therapy_df[therapy_df['Sentiment'] == 'neutral']
    
    # 5. Select and format response
    if not filtered.empty:
        response = random.choice(filtered['response'].tolist())
    else:
        response = random.choice(FALLBACK_RESPONSES)
    
    # 6. Ensure proper punctuation
    return response.strip() + ('' if response.endswith(('.','!','?')) else '.')

# --- Main Application ---
def main():
    # Initialize resources
    model, tokenizer, index_to_label = load_sentiment_model()
    therapy_df = load_therapy_data()
    
    print("Mental Health Support Chatbot (Type 'exit' to quit)")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Take care of yourself. Remember help is always available.")
                break
                
            if not user_input:
                print("Chatbot: I'm here to listen when you're ready to share.")
                continue
                
            response = generate_response(
                user_input=user_input,
                model=model,
                tokenizer=tokenizer,
                index_to_label=index_to_label,
                therapy_df=therapy_df
            )
            print(f"Chatbot: {response}")
            
        except KeyboardInterrupt:
            print("\nChatbot: Session ended. Please reach out to a professional if needed.")
            break
        except Exception as e:
            print(f"Chatbot: I'm having trouble understanding. Could you rephrase that?")
            print(f"[System Error] {str(e)}")

if __name__ == "__main__":
    main()