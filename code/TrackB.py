import pandas as pd
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertModel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define custom dataset class
class IntensityDataset(Dataset):
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

# Define the model class for intensity prediction
class IntensityRegressor(nn.Module):
    def __init__(self, n_classes):
        super(IntensityRegressor, self).__init__()
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

# Load Track A predictions (replace with actual Track A results)
track_a_predictions = pd.DataFrame({
    'Anger': [0, 1, 0, 1],
    'Fear': [1, 0, 1, 1],
    'Joy': [0, 0, 1, 1],
    'Sadness': [1, 0, 1, 0],
    'Surprise': [0, 1, 0, 1]
})

# Load the corresponding texts and ground truth intensity labels (for training intensity model)
data = pd.read_csv("eng_intensity.csv")  # Assume this file contains text and intensity labels
X = data['text']
y = data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]

# Filter data where Track A predicted the emotion as 1
selected_indices = track_a_predictions[track_a_predictions == 1].stack().index
filtered_X = X.iloc[[i[0] for i in selected_indices]].reset_index(drop=True)
filtered_y = y.iloc[[i[0] for i in selected_indices]].reset_index(drop=True)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128

intensity_dataset = IntensityDataset(filtered_X.tolist(), filtered_y, tokenizer, MAX_LEN)
intensity_loader = DataLoader(intensity_dataset, batch_size=16, shuffle=True)

# Initialize the intensity model
intensity_model = IntensityRegressor(n_classes=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intensity_model = intensity_model.to(device)

# Training setup
EPOCHS = 3
optimizer = optim.Adam(intensity_model.parameters(), lr=2e-5)
criterion = nn.MSELoss()

# Training loop for intensity prediction
for epoch in range(EPOCHS):
    intensity_model.train()
    for batch in intensity_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = intensity_model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Predict intensities for Track A results
intensity_model.eval()
intensity_predictions = []

with torch.no_grad():
    for batch in intensity_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = intensity_model(input_ids, attention_mask)
        intensity_predictions.append(outputs.cpu().numpy())

intensity_predictions = np.vstack(intensity_predictions)

# Combine Track A predictions with intensity predictions
final_results = track_a_predictions.copy()
for i, col in enumerate(['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']):
    final_results[col] = final_results[col].replace(1, pd.Series(intensity_predictions[:, i]).round().clip(1, 3))

print("Final Predictions (Combined with Intensity):")
print(final_results)
