import pandas as pd
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer, BertModel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels.iloc[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label.values, dtype=torch.float)
        }

# Define the model class
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.dropout(outputs.pooler_output)
        return self.out(output)

# Load the data
data = pd.read_csv("eng(a).csv")

# Step 1: Prepare the data
train_data = data.iloc[:100]
validation_data = data.iloc[100:120]
test_data = data.iloc[120:130]

X_train = train_data['text']
y_train = train_data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]
X_validation = validation_data['text']
y_validation = validation_data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]
X_test = test_data['text']
y_test = test_data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]

# Step 2: Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128

train_dataset = EmotionDataset(X_train.tolist(), y_train, tokenizer, MAX_LEN)
validation_dataset = EmotionDataset(X_validation.tolist(), y_validation, tokenizer, MAX_LEN)
test_dataset = EmotionDataset(X_test.tolist(), y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the Emotion Classifier
model = EmotionClassifier(n_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# Train the model
for epoch in range(3):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Tune threshold on the validation set
thresholds = np.arange(0.1, 0.9, 0.1)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs)
            binary_preds = (preds > threshold).int()

            predictions.append(binary_preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)
    f1 = f1_score(true_labels, predictions, average='macro')

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best Threshold: {best_threshold}, Best F1 Score: {best_f1:.4f}")

# Evaluate on the test set using the best threshold
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        preds = torch.sigmoid(outputs)
        binary_preds = (preds > best_threshold).int()

        predictions.append(binary_preds.cpu().numpy())
        true_labels.append(labels.cpu().numpy())

predictions = np.vstack(predictions)
true_labels = np.vstack(true_labels)

# Print metrics
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']))

# Calculate F1 score
f1 = f1_score(true_labels, predictions, average='macro')
print(f"Macro F1 Score: {f1:.4f}")
