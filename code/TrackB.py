import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch import nn, optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Set random seeds for reproducibility
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

# Define Dataset Class
class EmotionIntensityDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        targets = self.targets[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'targets': torch.tensor(targets, dtype=torch.float)
        }

# Define Intensity Prediction Model
class IntensityModel(nn.Module):
    def __init__(self, base_model_name, num_outputs, dropout_rate=0.3):
        super(IntensityModel, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.output(output)

# Load Data
data = pd.read_csv("trackb_emotions.csv")  # Assumes data with 'text' and 'emotion intensities'

# Prepare Dataset
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
BASE_MODEL = 'bert-base-uncased'
LEARNING_RATE = 2e-5

# Prepare Data
texts = data['text'].tolist()
targets = data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].values

train_texts, train_targets = texts[:80], targets[:80]
val_texts, val_targets = texts[80:100], targets[80:100]
test_texts, test_targets = texts[100:], targets[100:]

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Create DataLoaders
train_dataset = EmotionIntensityDataset(train_texts, train_targets, tokenizer, MAX_LEN)
val_dataset = EmotionIntensityDataset(val_texts, val_targets, tokenizer, MAX_LEN)
test_dataset = EmotionIntensityDataset(test_texts, test_targets, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize Model
model = IntensityModel(BASE_MODEL, num_outputs=5).cuda()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training Function
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    losses = []
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        targets = batch['targets'].cuda()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return np.mean(losses)

# Evaluation Function
def eval_model(model, data_loader, criterion):
    model.eval()
    losses = []
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            targets = batch['targets'].cuda()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.cpu().numpy())
    return np.mean(losses), np.vstack(predictions), np.vstack(true_values)

# Training Loop
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_targets = eval_model(model, val_loader, criterion)
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_loss = val_loss

# Load Best Model
model.load_state_dict(torch.load('best_model.pth'))

# Test Model
test_loss, test_preds, test_targets = eval_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')

# Post-process Predictions
def post_process_predictions(preds):
    return np.round(np.clip(preds, 0, 3)).astype(int)

test_preds = post_process_predictions(test_preds)

# Display Results
predictions_df = pd.DataFrame(test_preds, columns=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])
true_values_df = pd.DataFrame(test_targets, columns=['Anger_True', 'Fear_True', 'Joy_True', 'Sadness_True', 'Surprise_True'])
results_df = pd.concat([predictions_df, true_values_df], axis=1)

print("Predictions vs True Values:")
print(results_df)

# Calculate Evaluation Metrics for Each Emotion
for i, emotion in enumerate(['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']):
    mse = mean_squared_error(test_targets[:, i], test_preds[:, i])
    mae = mean_absolute_error(test_targets[:, i], test_preds[:, i])
    print(f'{emotion}: MSE={mse:.4f}, MAE={mae:.4f}')
