import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Load the dataset
file_path = 'eng.csv'  # 请确保数据集路径正确

data = pd.read_csv(file_path)
print("Dataset Loaded Successfully!")
print(data.head())

# 2. Define emotions to predict
target_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# 3. Split data into training and test sets (90% training, 10% test)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
test_data = test_data.sample(20, random_state=42).reset_index(drop=True)  # Select 20 test samples

# 4. Use TF-IDF to vectorize the text column
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

# 5. Train a two-stage model with data augmentation for minority classes
final_predictions_df = test_data[['text']].copy()
true_labels = []
predicted_labels = []

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")
    
    # Define target variable
    y_train = train_data[emotion]
    y_test = test_data[emotion]
    
    # Stage 1: Predict binary outcome (0 vs >0)
    y_train_binary = (y_train > 0).astype(int)
    
    # Resample the data to balance classes
    X_resampled, y_resampled = resample(X_train, y_train_binary, random_state=42)
    
    stage1_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    stage1_model.fit(X_resampled, y_resampled)
    y_pred_binary = stage1_model.predict(X_test)
    
    # Stage 2: Predict intensities (1, 2, 3) for non-zero samples
    X_train_stage2 = X_train[y_train > 0]
    y_train_stage2 = y_train[y_train > 0]
    X_test_stage2 = X_test[y_pred_binary > 0]

    if X_test_stage2.shape[0] > 0:
        stage2_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        stage2_model.fit(X_train_stage2, y_train_stage2)
        y_pred_stage2 = stage2_model.predict(X_test_stage2)
        
        # Combine results
        final_predictions = np.zeros(len(y_test), dtype=int)
        final_predictions[y_pred_binary > 0] = y_pred_stage2.astype(int)
    else:
        final_predictions = np.zeros(len(y_test), dtype=int)
    
    # Append true and predicted labels
    true_labels.append(y_test.values.astype(int))
    predicted_labels.append(final_predictions.astype(int))

# Combine true and predicted labels for display
final_predictions_df['True_Labels'] = list(map(list, zip(*true_labels)))
final_predictions_df['Predicted_Labels'] = list(map(list, zip(*predicted_labels)))

# 6. Display final results
print("\nFinal comparison of predictions and true labels for all emotions:")
for i, row in final_predictions_df.iterrows():
    print(f"Text: {row['text']}\nTrue Labels: {[int(label) for label in row['True_Labels']]}\nPredicted Labels: {[int(label) for label in row['Predicted_Labels']]}\n")
