from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Step 1: Balance Data for Training
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

emotion = 'Joy'
balanced_train_df = balance_data(train_df, emotion)

# Step 2: Baseline Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=500)
X_train_lr = vectorizer.fit_transform(balanced_train_df['text'])
y_train_lr = balanced_train_df[emotion]

X_test_lr = vectorizer.transform(test_df['text'])
y_test_lr = test_df[emotion]

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_lr, y_train_lr)
lr_preds = lr_model.predict(X_test_lr)

print("\nBaseline Logistic Regression Report:")
print(classification_report(y_test_lr, lr_preds))

# Step 3: Train BERT with More Epochs
train_dataset = EmotionDataset(balanced_train_df['text'].tolist(), balanced_train_df[emotion].tolist(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

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

train_model(model, train_loader, epochs=5)
bert_preds, bert_labels = evaluate_model(model, test_loader)

# Step 4: BERT Report
print("\nBERT Fine-tuned Report:")
print(classification_report(bert_labels, bert_preds))

