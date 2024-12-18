import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.nn.functional as F

# Step 1: Load Data
file_path = "eng(b).csv"  # 修改为你的CSV路径
df = pd.read_csv(file_path)

# Step 2: Define Emotion and Process Data
emotion = 'Joy'  # 选择情绪列: 'Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'
df = df[['text', emotion]]

# Step 3: Split Data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 4: Dataset Class for BERT
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Step 5: Initialize Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # 4强度级别：0, 1, 2, 3

# Step 6: Prepare DataLoaders
train_dataset = EmotionDataset(train_df['text'].tolist(), train_df[emotion].tolist(), tokenizer)
test_dataset = EmotionDataset(test_df['text'].tolist(), test_df[emotion].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 7: Training Function
optimizer = AdamW(model.parameters(), lr=5e-5)

def train_model(model, dataloader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Step 8: Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(F.softmax(logits, dim=1), axis=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels

# Step 9: Train and Evaluate Model
train_model(model, train_loader, epochs=2)
predictions, true_labels = evaluate_model(model, test_loader)

# Step 10: Output Results
print(f"\nClassification Report for {emotion}:")
print(classification_report(true_labels, predictions, digits=4))
