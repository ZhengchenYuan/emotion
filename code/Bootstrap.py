import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import statsmodels.api as sm
import numpy as np
from sklearn.utils import resample

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
X_train = vectorizer.fit_transform(train['Text']).toarray()
X_val = vectorizer.transform(validation['Text']).toarray()
X_test = vectorizer.transform(test['Text']).toarray()

# Target Emotion Columns
targets = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Step 4: Bootstrap Uncertainty Estimation
def bootstrap_model_uncertainty(X_train, y_train, X_test, n_iterations=100):
    predictions = np.zeros((n_iterations, X_test.shape[0]))
    for i in range(n_iterations):
        X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_resampled, y_resampled)
        predictions[i, :] = model.predict(X_test)
    
    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)
    return mean_prediction, std_prediction

# Step 5: Train Models and Predict with Uncertainty
def train_and_infer_with_uncertainty(X_train, y_train, X_test):
    results = {}
    for emotion in targets:
        print(f"Bootstrapping for uncertainty in {emotion} predictions...")
        mean_pred, std_pred = bootstrap_model_uncertainty(X_train, y_train[emotion], X_test)
        results[emotion] = {'mean': mean_pred, 'std': std_pred}
    return results

# Extract Targets
y_train = train[targets]
y_test = test[targets]

# Run Bootstrap with Uncertainty
test_results = {}
for emotion in targets:
    print(f"\nProcessing emotion: {emotion}")
    mean_pred, std_pred = bootstrap_model_uncertainty(X_train, y_train[emotion], X_test)
    test_results[emotion] = {'mean': mean_pred, 'std': std_pred}

# Step 6: Compare Results with Uncertainty
for emotion in targets:
    print(f"\nEmotion: {emotion}")
    comparison = pd.DataFrame({
        'Text': test['Text'],
        'Actual': y_test[emotion].values,
        'Predicted Mean': test_results[emotion]['mean'].round().astype(int),
        'Uncertainty (Std)': test_results[emotion]['std']
    })
    print(comparison.head())
