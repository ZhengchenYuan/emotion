import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import BertTokenizer, BertModel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
dev_data = data.iloc[100:120]
test_data = data.iloc[120:180]

X_train = train_data['text']
y_train = train_data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]
X_dev = dev_data['text']
y_dev = dev_data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]
X_test = test_data['text']
y_test = test_data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]

# Step 2: Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128

train_dataset = EmotionDataset(X_train.tolist(), y_train, tokenizer, MAX_LEN)
dev_dataset = EmotionDataset(X_dev.tolist(), y_dev, tokenizer, MAX_LEN)
test_dataset = EmotionDataset(X_test.tolist(), y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Step 3: Initialize the model
model = EmotionClassifier(n_classes=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Step 4: Training setup
EPOCHS = 3
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(EPOCHS):
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

# Step 5: Determine best threshold on development set
def evaluate_threshold(loader, threshold):
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            sigmoid_outputs = torch.sigmoid(outputs)
            binary_predictions = (sigmoid_outputs > threshold).float()
            predictions.append(binary_predictions.cpu().numpy())
            true_values.append(labels.cpu().numpy())

    predictions = np.vstack(predictions)
    true_values = np.vstack(true_values)
    f1 = f1_score(true_values, predictions, average='micro')
    return f1

best_threshold = 0.0
best_f1 = 0.0
thresholds = np.arange(0.1, 0.9, 0.1)

for threshold in thresholds:
    f1 = evaluate_threshold(dev_loader, threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best F1 score: {best_f1}")

# Step 6: Evaluate on test set
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        sigmoid_outputs = torch.sigmoid(outputs)
        binary_predictions = (sigmoid_outputs > best_threshold).float()
        predictions.append(binary_predictions.cpu().numpy())
        true_values.append(labels.cpu().numpy())

predictions = np.vstack(predictions)
true_values = np.vstack(true_values)

# Step 7: Calculate accuracy
accuracy = accuracy_score(true_values.flatten(), predictions.flatten())
print(f"Test Accuracy: {accuracy:.2f}")

# Step 8: Calculate matching labels
predictions_str = [','.join(map(str, pred.astype(int))) for pred in predictions]
true_values_str = [','.join(map(str, true.astype(int))) for true in true_values]

final_predictions_df = pd.DataFrame({
    'Predicted_Labels': predictions_str,
    'True_Labels': true_values_str
})

matches = 0
for _, row in final_predictions_df.iterrows():
    true = list(map(int, row['True_Labels'].split(',')))
    pred = list(map(int, row['Predicted_Labels'].split(',')))
    if true == pred:
        matches += 1

total_samples = len(final_predictions_df)
match_rate = matches / total_samples * 100

print(f"\nNumber of samples where predicted labels match true labels: {matches}/{total_samples}")
print(f"Match rate: {match_rate:.2f}%")

# Add match information to DataFrame
final_predictions_df['Match'] = final_predictions_df.apply(
    lambda row: 1 if list(map(int, row['True_Labels'].split(','))) == list(map(int, row['Predicted_Labels'].split(','))) else 0,
    axis=1
)

# Save results to CSV
output_file_path = "results_comparison_with_matches.csv"
final_predictions_df.to_csv(output_file_path, index=False)
print(f"Updated results with match information saved to {output_file_path}")
