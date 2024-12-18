import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Step 1: Load CSV Data
file_path = "eng(b).csv"  # 修改为您的CSV文件路径
df = pd.read_csv(file_path)

# Step 2: Use First 1500 Rows for Training
df = df.iloc[:1500]

# Step 3: Splitting Data
train, temp = train_test_split(df, test_size=0.3, random_state=42)
validation, test = train_test_split(temp, test_size=0.33, random_state=42)

print(f"Training Set: {len(train)} | Validation Set: {len(validation)} | Test Set: {len(test)}")

# Step 4: Prepare Dataset for BERT
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 5: Initialize BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # 4 levels: 0, 1, 2, 3

# Prepare DataLoaders for Training and Testing
def prepare_dataloader(data, targets, tokenizer, batch_size=16):
    dataset = EmotionDataset(data['text'].tolist(), targets.tolist(), tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

y_train = train['intensity'].values  # Ensure you have an 'intensity' column
X_train = train[['text']]
train_loader = prepare_dataloader(X_train, y_train, tokenizer)

y_test = test['intensity'].values
X_test = test[['text']]
test_loader = prepare_dataloader(X_test, y_test, tokenizer, batch_size=32)

# Step 6: Training Function
def train_model(model, dataloader, epochs=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Step 7: Prediction Function
def predict_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels

# Step 8: Train and Evaluate Model
train_model(model, train_loader, epochs=1)
preds, labels = predict_model(model, test_loader)

# Step 9: Evaluate Results
print("\nClassification Report:")
print(classification_report(labels, preds))
