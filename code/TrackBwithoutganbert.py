import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random

# Step 1: Simulated Data Creation (Ensure 0,1,2,3 appear)
data = {
    'Text': [
        "Never saw him again.", "I love telling this story.", "How stupid of him.",
        "None of us did.", "I can't believe it! I won the scholarship! This is amazing!"
    ]*20,  # Replicate data to meet training, validation, test size
    'Anger': [0, 0, 2, 0, 0]*20,
    'Fear': [0, 0, 0, 0, 0]*20,
    'Joy': [0, 2, 0, 0, 3]*20,
    'Sadness': [2, 0, 0, 0, 0]*20,
    'Surprise': [0, 0, 0, 0, 3]*20
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Splitting Data
train, temp = train_test_split(df, test_size=0.3, random_state=42)
validation, test = train_test_split(temp, test_size=0.33, random_state=42)

print(f"Training Set: {len(train)} | Validation Set: {len(validation)} | Test Set: {len(test)}")

# Step 3: Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train['Text'])
X_val = vectorizer.transform(validation['Text'])
X_test = vectorizer.transform(test['Text'])

# Target Emotion Columns
targets = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Step 4: Model Training and Prediction
def train_and_predict(X_train, y_train, X_val, X_test):
    predictions = {}
    for emotion in targets:
        print(f"Training model for {emotion}...")
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train[emotion])
        
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        predictions[emotion] = test_pred
        
        print(f"Validation Classification Report for {emotion}:")
        print(classification_report(y_val[emotion], val_pred))
    return predictions

# Extract Targets
y_train = train[targets]
y_val = validation[targets]
y_test = test[targets]

# Train Models and Predict
test_predictions = train_and_predict(X_train, y_train, X_val, X_test)

# Step 5: Compare Predictions to Test Set
print("\nTest Set vs Predictions:")
for emotion in targets:
    print(f"Emotion: {emotion}")
    comparison = pd.DataFrame({
        'Text': test['Text'],
        'Actual': y_test[emotion],
        'Predicted': test_predictions[emotion]
    })
    print(comparison)
    print("\n")
