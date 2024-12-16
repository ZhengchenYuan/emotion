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
    def __init__(self, base_model_name, dropout_rate=0.3):
        super(IntensityModel, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(self.bert.config.hidden_size, 1)  # Predict a single value

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.output(output)

# Training Function
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    losses = []
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        targets = batch['targets'].cuda()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).squeeze(-1)
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

            outputs = model(input_ids, attention_mask).squeeze(-1)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            predictions.append(outputs.cpu().numpy())
            true_values.append(targets.cpu().numpy())
    return np.mean(losses), np.hstack(predictions), np.hstack(true_values)

# Training and Evaluation Pipeline for Each Emotion
def train_and_evaluate_emotion(emotion, texts, targets, base_model, max_len, batch_size, epochs, lr):
    print(f"\nTraining for Emotion: {emotion}")
    train_texts, train_targets = texts[:80], targets[:80]
    val_texts, val_targets = texts[80:100], targets[80:100]
    test_texts, test_targets = texts[100:], targets[100:]

    # Create datasets and dataloaders
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_dataset = EmotionIntensityDataset(train_texts, train_targets, tokenizer, max_len)
    val_dataset = EmotionIntensityDataset(val_texts, val_targets, tokenizer, max_len)
    test_dataset = EmotionIntensityDataset(test_texts, test_targets, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = IntensityModel(base_model).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = eval_model(model, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f'best_model_{emotion}.pth')
            best_val_loss = val_loss

    # Test the model
    model.load_state_dict(torch.load(f'best_model_{emotion}.pth'))
    test_loss, test_preds, test_targets = eval_model(model, test_loader, criterion)
    print(f"Test Loss for {emotion}: {test_loss:.4f}")

    # Post-process predictions
    test_preds = np.round(np.clip(test_preds, 0, 3)).astype(int)
    return test_preds, test_targets

# Main Pipeline
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("trackb_emotions.csv")
    texts = data['text'].tolist()
    emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

    # Parameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    BASE_MODEL = 'bert-base-uncased'
    LEARNING_RATE = 2e-5

    # Train and Evaluate for Each Emotion
    results = {}
    for emotion in emotions:
        targets = data[emotion].values
        preds, true_values = train_and_evaluate_emotion(emotion, texts, targets, BASE_MODEL, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE)
        results[emotion] = {'predictions': preds, 'true_values': true_values}

    # Combine Results
    final_preds = pd.DataFrame({emotion: results[emotion]['predictions'] for emotion in emotions})
    final_true = pd.DataFrame({emotion + '_True': results[emotion]['true_values'] for emotion in emotions})
    results_df = pd.concat([final_preds, final_true], axis=1)

    # Display Results
    print("\nFinal Predictions vs True Values:")
    print(results_df)

    # Evaluate Metrics
    for emotion in emotions:
        mse = mean_squared_error(results[emotion]['true_values'], results[emotion]['predictions'])
        mae = mean_absolute_error(results[emotion]['true_values'], results[emotion]['predictions'])
        print(f"{emotion}: MSE={mse:.4f}, MAE={mae:.4f}")
