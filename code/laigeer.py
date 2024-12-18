import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.nn.functional as F

# Step 1: Load Data
file_path = "eng(b).csv"  # 修改为你的CSV路径
df = pd.read_csv(file_path)

# Step 2: Split Data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: Balance Data for Training
def balance_data(df, emotion):
    class_0 = df[df[emotion] == 0]
    class_1 = df[df[emotion] == 1]
    class_2 = df[df[emotion] == 2]
    class_3 = df[df[emotion] == 3]

    min_size = min(len(class_0), len(class_1), len(class_2), len(class_3))
    balanced_df = pd.concat([
        resample(class_0, n_samples=min_size, random_state=42),
        resample(class_1, n_samples=min_size, random_state=42),
        resample(class_2, n_samples=min_size, random_state=42),
        resample(class_3, n_samples=min_size, random_state=42)
    ])
    return balanced_df

# Apply balancing on the training set
emotion = 'Joy'  # 选择目标情绪列
balanced_train_df = balance_data(train_df, emotion)

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
train_dataset = EmotionDataset(balanced_train_df['text'].tolist(), balanced_train_df[emotion].tolist(), tokenizer)
test_dataset = EmotionDataset(test_df['text'].tolist(), test_df[emotion].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 7: Training Function
def train_model(model, dataloader, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
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
train_model(model, train_loader, epochs=5)
bert_preds, bert_labels = evaluate_model(model, test_loader)

# Step 10: Output Results
print(f"\nBERT Fine-tuned Report for {emotion}:")
print(classification_report(bert_labels, bert_preds, digits=4))

# Step 11: Baseline Logistic Regression
vectorizer = TfidfVectorizer(max_features=500)
X_train_lr = vectorizer.fit_transform(balanced_train_df['text'])
y_train_lr = balanced_train_df[emotion]

X_test_lr = vectorizer.transform(test_df['text'])
y_test_lr = test_df[emotion]

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_lr, y_train_lr)
lr_preds = lr_model.predict(X_test_lr)

print("\nBaseline Logistic Regression Report:")
print(classification_report(y_test_lr, lr_preds, digits=4))
