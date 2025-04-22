import pandas as pd
import json
import re
from transformers import pipeline
from sklearn.utils import resample

# Load datasets
conv_df = pd.read_csv("data/raw/mental_health_conv/train.csv")
nlp_df = pd.read_csv("data/raw/nlp_mental_health/train.csv")
therapy_data = pd.read_json("data/raw/therapy/intents.json")
print("Datasets loaded successfully.")

# Clean NLP data
nlp_df.columns = nlp_df.columns.str.strip()
nlp_df['Context'] = nlp_df['Context'].str.replace(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', regex=True
)
nlp_df['Context'] = nlp_df['Context'].str.replace(r'[^A-Za-z0-9\s.,!?]', '', regex=True).str.strip()

# Sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english",
                              truncation=True)

def label_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        label = result["label"].lower()
        confidence = result["score"]

        # Confidence-adjusted labeling
        if label == "positive" and confidence > 0.7:
            return "positive"
        elif label == "negative" and confidence > 0.55:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        return "neutral"

print("Applying sentiment analysis...")
nlp_df['Sentiment'] = nlp_df['Context'].apply(label_sentiment)

# Expand known negative expressions (with regex)
negative_patterns = [
    r"\b(i feel )?(hopeless|worthless|ugly|helpless|lost)\b",
    r"\b(i hate myself|can't stop crying|want to disappear)\b",
    r"\babusive\b", r"\bsuicidal\b"
]

def flag_negative(text):
    for pat in negative_patterns:
        if re.search(pat, text.lower()):
            return True
    return False

nlp_df.loc[nlp_df['Context'].apply(flag_negative), 'Sentiment'] = 'negative'

# Balance dataset: limit neutral dominance
def balance_dataset(df):
    grouped = [df[df.Sentiment == s] for s in ['positive', 'negative', 'neutral']]
    min_len = min(len(g) for g in grouped)
    target_n = min(min_len, 5000)
    balanced_parts = [resample(g, n_samples=target_n, random_state=42) for g in grouped]
    return pd.concat(balanced_parts)

print("Balancing dataset...")
balanced_df = balance_dataset(nlp_df)

# Process therapy intents
therapy_intents = []
for intent in therapy_data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    responses = intent['response']
    
    override_sentiment = "negative" if tag in ['self-esteem-A', 'self-esteem-B', 'domesticviolence'] else None
    
    for pattern in patterns:
        cleaned = re.sub(r'[^A-Za-z0-9\s.,!?]', '', pattern).strip()
        therapy_intents.append({
            'tag': tag,
            'pattern': cleaned,
            'response': '|'.join(responses),
            'Sentiment': override_sentiment or label_sentiment(cleaned)
        })

therapy_df = pd.DataFrame(therapy_intents)

# Combine all
print("Combining datasets...")
balanced_df['tag'] = 'user'
balanced_df['pattern'] = balanced_df['Context']
balanced_df['response'] = ''
combined = pd.concat([balanced_df[['tag', 'pattern', 'response', 'Sentiment']], therapy_df], ignore_index=True)

# Final clean-up
combined = combined.drop_duplicates(subset=['pattern']).dropna()
combined = combined.sample(frac=1).reset_index(drop=True)

# Save final dataset
combined.to_csv("data/processed/combined_balanced_dataset.csv", index=False)
print("Processing complete. Final dataset size:", len(combined))
