import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import numpy as np

# Step 1: Load CSV Data
file_path = "eng(b).csv"  # 修改为您的CSV文件路径
df = pd.read_csv(file_path)

# Step 2: Use First 1500 Rows for Training
df = df.iloc[:1500]

# Step 3: Splitting Data
train, temp = train_test_split(df, test_size=0.3, random_state=42)
validation, test = train_test_split(temp, test_size=0.33, random_state=42)

print(f"Training Set: {len(train)} | Validation Set: {len(validation)} | Test Set: {len(test)}")

# Step 4: Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train['text']).toarray()
X_val = vectorizer.transform(validation['text']).toarray()
X_test = vectorizer.transform(test['text']).toarray()

# Step 5: Target Emotion Columns
targets = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
y_train = train[targets]
y_val = validation[targets]
y_test = test[targets]

# Step 6: Bootstrap Uncertainty Estimation
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

# Step 7: Train Models and Predict with Uncertainty
def train_and_infer_with_uncertainty(X_train, y_train, X_test, y_test):
    results = {}
    accuracies = {}
    for emotion in targets:
        print(f"Bootstrapping for uncertainty in {emotion} predictions...")
        mean_pred, std_pred = bootstrap_model_uncertainty(X_train, y_train[emotion], X_test)
        predicted = mean_pred.round().astype(int)
        accuracy = accuracy_score(y_test[emotion], predicted)
        accuracies[emotion] = accuracy
        results[emotion] = {'mean': mean_pred, 'std': std_pred, 'accuracy': accuracy}
    return results, accuracies

# Run Bootstrap with Uncertainty and Accuracy
test_results, accuracies = train_and_infer_with_uncertainty(X_train, y_train, X_test, y_test)

# Step 8: Compare Results with Uncertainty and Output Accuracy
for emotion in targets:
    print(f"\nEmotion: {emotion} (Accuracy: {accuracies[emotion]:.4f})")
    comparison = pd.DataFrame({
        'Text': test['text'],
        'Actual': y_test[emotion].values,
        'Predicted Mean': test_results[emotion]['mean'].round().astype(int),
        'Uncertainty (Std)': test_results[emotion]['std']
    })
    print(comparison.head())

print("\nOverall Accuracy for Each Emotion:")
for emotion, acc in accuracies.items():
    print(f"{emotion}: {acc:.4f}")
