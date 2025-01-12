import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import numpy as np

# Load the dataset
data = pd.read_csv(r"C:\Users\ziyue\nlp\emotion\eng(b).csv")
print("Dataset Loaded Successfully!")
print(data.head())

# Define emotions to predict
target_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Split data into training and test sets
train_data = data.iloc[:2000]  # First 2000 entries as training data
test_data = data.iloc[2000:].reset_index(drop=True)  # Remaining data as test data

# Use TF-IDF to vectorize the text column
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

# Train a two-stage model with data augmentation for minority classes
final_predictions_df = test_data[['text']].copy()
overall_true_labels = []
overall_pred_labels = []

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")
    
    # Define target variable
    y_train = train_data[emotion]
    y_test = test_data[emotion]
    
    # Stage 1: Predict binary outcome (0 vs >0)
    y_train_binary = (y_train > 0).astype(int)
    X_resampled, y_resampled = resample(X_train, y_train_binary, random_state=42)
    
    stage1_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    stage1_model.fit(X_resampled, y_resampled)
    y_pred_binary = stage1_model.predict(X_test)
    
    # Stage 2: Predict intensities (1, 2, 3) for non-zero samples
    X_train_stage2 = X_train[y_train > 0]
    y_train_stage2 = y_train[y_train > 0]
    X_test_stage2 = X_test[y_pred_binary > 0]

    final_predictions = np.zeros(len(y_test), dtype=int)
    if X_test_stage2.shape[0] > 0:
        stage2_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        stage2_model.fit(X_train_stage2, y_train_stage2)
        y_pred_stage2 = stage2_model.predict(X_test_stage2)
        final_predictions[y_pred_binary > 0] = y_pred_stage2.astype(int)
    
    overall_true_labels.append(y_test.values.astype(int))
    overall_pred_labels.append(final_predictions.astype(int))

    # Calculate metrics for the current emotion
    print(f"Metrics for {emotion}:")
    print(classification_report(y_test.values.astype(int), final_predictions))

# Combine true and predicted labels for display and format for easy reading
final_predictions_df['True_Labels'] = ["[" + ", ".join(map(str, labels)) + "]" for labels in zip(*overall_true_labels)]
final_predictions_df['Predicted_Labels'] = ["[" + ", ".join(map(str, labels)) + "]" for labels in zip(*overall_pred_labels)]

# Calculate the number of matching labels between True_Labels and Predicted_Labels
matches = sum([1 for true, pred in zip(final_predictions_df['True_Labels'], final_predictions_df['Predicted_Labels']) if true == pred])
total_samples = len(final_predictions_df)
match_rate = matches / total_samples * 100
print(f"\nNumber of samples where predicted labels match true labels: {matches}/{total_samples}")
print(f"Match rate: {match_rate:.2f}%")

# Save the cleaned results to a CSV file
output_file_path = 'emotion_predictions_cleaned_results.csv'
final_predictions_df.to_csv(output_file_path, index=False)
print(f"Updated results with match information saved to {output_file_path}")
