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

# Define GAN components
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
data = pd.read_csv("eng(a).csv")

# Step 1: Prepare the data
train_data = data.iloc[:100]
dev_data = data.iloc[100:120]
test_data = data.iloc[120:130]

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

# Initialize GAN
NOISE_DIM = 100
LABEL_DIM = 5
generator = Generator(NOISE_DIM, LABEL_DIM).to(device)
discriminator = Discriminator(LABEL_DIM).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
criterion_gan = nn.BCELoss()

# Train GAN
for epoch in range(100):  # Simplified GAN training loop
    for batch in train_loader:
        # Train Discriminator
        real_labels = batch['labels'].to(device)
        batch_size = real_labels.size(0)
        noise = torch.randn(batch_size, NOISE_DIM).to(device)
        fake_labels = generator(noise)

        real_targets = torch.ones(batch_size, 1).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        optimizer_D.zero_grad()
        real_loss = criterion_gan(discriminator(real_labels), real_targets)
        fake_loss = criterion_gan(discriminator(fake_labels.detach()), fake_targets)
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        generated_targets = torch.ones(batch_size, 1).to(device)
        loss_G = criterion_gan(discriminator(fake_labels), generated_targets)
        loss_G.backward()
        optimizer_G.step()

# Use GAN-enhanced labels for training classifier
enhanced_labels = []
generator.eval()
with torch.no_grad():
    for batch in train_loader:
        noise = torch.randn(len(batch['labels']), NOISE_DIM).to(device)
        enhanced_labels.append(generator(noise).cpu().numpy())

enhanced_labels = np.vstack(enhanced_labels)
X_train = np.hstack([X_train.values.reshape(-1, 1), enhanced_labels])

# Initialize the Emotion Classifier
model = EmotionClassifier(n_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# Training classifier with enhanced data
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

# Testing and evaluation
model.eval()

# Store predictions and true labels
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Get model predictions
        outputs = model(input_ids, attention_mask)
        preds = torch.sigmoid(outputs)  # Convert logits to probabilities
        predictions.append(preds.cpu().numpy())
        true_labels.append(labels.cpu().numpy())

# Convert predictions and true labels to numpy arrays
predictions = np.vstack(predictions)
true_labels = np.vstack(true_labels)

# Apply threshold to convert probabilities to binary predictions
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Compare predictions and true labels
for i in range(len(binary_predictions)):
    print(f"Sample {i+1}:")
    print(f"Predicted: {binary_predictions[i]}")
    print(f"Actual: {true_labels[i]}")
    print()

# Calculate and print evaluation metrics
print("Classification Report:")
print(classification_report(true_labels, binary_predictions, target_names=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']))

# Calculate F1 score
f1 = f1_score(true_labels, binary_predictions, average='macro')
print(f"Macro F1 Score: {f1:.4f}")

# Calculate and print match rate
matches = np.sum(binary_predictions == true_labels)
total_elements = binary_predictions.size
match_rate = matches / total_elements

print(f"Match Rate: {match_rate:.4f} ({matches}/{total_elements})")

