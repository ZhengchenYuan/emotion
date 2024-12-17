import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import rel_entr
import numpy as np

# Step 1: Load Data (Simulated)
data = {
    'Text': [
        "I am so happy!", "This is terrifying.", "I feel sad.", "What an amazing experience!",
        "This is so frustrating.", "I am shocked and surprised!"
    ] * 20,  # Repeat to create more data
    'Joy': [3, 0, 0, 3, 0, 2] * 20,
    'Fear': [0, 3, 0, 0, 0, 1] * 20,
    'Sadness': [0, 0, 3, 0, 0, 0] * 20,
    'Anger': [0, 0, 0, 0, 3, 0] * 20,
    'Surprise': [0, 0, 0, 0, 0, 3] * 20
}

df = pd.DataFrame(data)
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Step 2: BERT Feature Extraction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_bert_features(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract [CLS] token features

X_train = extract_bert_features(train['Text'].tolist())
X_test = extract_bert_features(test['Text'].tolist())

y_train = train[['Joy', 'Fear', 'Sadness', 'Anger', 'Surprise']]
y_test = test[['Joy', 'Fear', 'Sadness', 'Anger', 'Surprise']]

# Step 3: Statistical Inference + Logistic Regression
predictions = {}
kl_divergence = {}

for emotion in y_train.columns:
    print(f"Training for {emotion}...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train[emotion])
    y_pred = model.predict_proba(X_test)
    
    # Store predictions
    predictions[emotion] = np.argmax(y_pred, axis=1)
    
    # Calculate KL Divergence between prediction and actual distributions
    actual_dist = y_test[emotion].value_counts(normalize=True).sort_index()
    pred_dist = pd.Series(predictions[emotion]).value_counts(normalize=True).sort_index()
    kl_div = sum(rel_entr(actual_dist, pred_dist))
    kl_divergence[emotion] = kl_div

# Step 4: Results
for emotion in predictions:
    print(f"\n{emotion} - KL Divergence: {kl_divergence[emotion]:.4f}")
    comparison = pd.DataFrame({
        'Text': test['Text'].values,
        'Actual': y_test[emotion].values,
        'Predicted': predictions[emotion]
    })
    print(comparison.head())
