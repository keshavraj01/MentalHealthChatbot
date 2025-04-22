from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import torch
from imblearn.over_sampling import RandomOverSampler

# --- 1. Load Dataset ---
df = pd.read_csv("data/processed/combined_balanced_dataset.csv")

# --- 2. Clean Data ---
df.columns = df.columns.str.strip()

# Use 'pattern' instead of 'Context'
if 'pattern' not in df.columns:
    print("Available columns:", df.columns)
    raise KeyError("Expected column 'pattern' not found in dataset.")

# Drop rows with missing or empty 'pattern'
df['pattern'] = df['pattern'].astype(str).str.strip()
df = df[df['pattern'].notnull() & (df['pattern'] != '')]

# --- 3. Label Encoding ---
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['Sentiment'].map(label_mapping)

if df['label'].isnull().any():
    print("Unmapped sentiment values:", df[df['label'].isnull()]['Sentiment'].unique())
    raise ValueError("Found unmapped sentiment labels.")

# --- 4. Oversample for Balance ---
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(df[['pattern']], df['label'])
resampled_df = pd.DataFrame({'pattern': X_res['pattern'], 'label': y_res})

# Recover sentiment text (optional)
inv_label_mapping = {v: k for k, v in label_mapping.items()}
resampled_df['Sentiment'] = resampled_df['label'].map(inv_label_mapping)

print("\nBalanced Class Distribution:")
print(resampled_df['Sentiment'].value_counts())

# --- 5. Train-Test Split ---
train_df, test_df = train_test_split(
    resampled_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=resampled_df['label']
)

# --- 6. Convert to Hugging Face Dataset ---
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
    'test': Dataset.from_pandas(test_df.reset_index(drop=True))
})

# --- 7. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["pattern"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --- 8. Model ---
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=inv_label_mapping,
    label2id=label_mapping
)

# --- 9. TrainingArguments ---
training_args = TrainingArguments(
    output_dir="models/sentiment_model",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available()
)

# --- 10. Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }

# --- 11. Train ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

train_results = trainer.train()
eval_results = trainer.evaluate()

# --- 12. Save Model & Tokenizer ---
trainer.save_model("models/sentiment_model")
tokenizer.save_pretrained("models/sentiment_model")

with open("models/sentiment_model/label_mapping.json", "w") as f:
    json.dump(label_mapping, f)

# --- 13. Final Report ---
print("\nTraining Complete!")
print(f"Final Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Final F1 (Macro): {eval_results['eval_f1_macro']:.4f}")
print(f"Final F1 (Weighted): {eval_results['eval_f1_weighted']:.4f}")
